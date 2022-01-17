import os

import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import tensorflow_datasets as tfds

import torch

gpu1_path = "/home/edupuis@inl34.ec-lyon.fr/.onnx/datasets"
ecl_path = "~/datasets"
docker_path = "/datasets"
dataset_path = docker_path if not os.environ.get('GPU_ENV') else \
    gpu1_path if os.environ['GPU_ENV'] == "gpu1" else ecl_path
print(dataset_path)


def get_imagenet(batch_size=1024, train=False, preprocess=None, IMAGE_SHAPE=(224, 224, 3)):
    imagenet, info = tfds.load(
        'imagenet2012',
        data_dir=dataset_path,
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
        download=False,
    )
    # extract info
    train_data, val_data = imagenet['train'], imagenet['validation']
    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['validation'].num_examples
    shape = info.features["image"].shape
    classes = info.features["label"].num_classes
    print(num_train_examples, num_test_examples, IMAGE_SHAPE, classes)

    if not preprocess:
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        preprocess = lambda image: (image - image_mean) / image_std

    if preprocess == tf.keras.applications.inception_v3.preprocess_input \
            or preprocess == tf.keras.applications.efficientnet.preprocess_input:
        print("Use Inception V3 preprocessing")

        def validation_image_preprocess(image, label):
            height = width = 224
            central_fraction = 0.875
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.central_crop(image, central_fraction=central_fraction)
            image = tf.expand_dims(image, 0)
            image = tf.image.resize(image, [height, width])
            image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            image.set_shape([height, width, 3])
            return image, label

    elif preprocess == tf.keras.applications.mobilenet_v2.preprocess_input \
            or preprocess == tf.keras.applications.resnet_v2.preprocess_input:
        print("Use Mobilenet preprocessing")

        def validation_image_preprocess(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (256, 256))
            image = tf.image.central_crop(image, 0.875)
            image = preprocess(image)
            return image, label
    else:
        print("Use VGG preprocessing")

        def validation_image_preprocess(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
            image = preprocess(image)
            return image, label

    def train_image_preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess(image)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(image, IMAGE_SHAPE)
        image = tf.image.resize_with_crop_or_pad(image, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        return image, label

    # Map preprocessing for validation dataset
    val_data_preprocessed = val_data.map(validation_image_preprocess, num_parallel_calls=4
                                         ).batch(batch_size
                                                 ).prefetch(tf.data.AUTOTUNE
                                                            )

    # Map preprocessing for training dataset
    train_data_preprocessed = train_data.map(train_image_preprocess, num_parallel_calls=tf.data.AUTOTUNE
                                             ).batch(batch_size
                                                     ).prefetch(tf.data.AUTOTUNE)

    # Extract labels
    if not train:
        return val_data_preprocessed
    return train_data_preprocessed, val_data_preprocessed, num_train_examples


def get_cifar(batch_size=64, train=False):
    dataset, metadata = tfds.load('cifar10', data_dir=dataset_path, as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    def norm(image, label):
        return tf.cast(image, tf.float32) / 255, label

    # Extract relevant data
    num_train_examples = metadata.splits["train"].num_examples
    num_test_examples = metadata.splits["test"].num_examples
    shape = metadata.features["image"].shape
    classes = metadata.features["label"].num_classes
    print(num_train_examples, num_test_examples, shape, classes)

    train_dataset = train_dataset.map(norm).cache().repeat().shuffle(num_train_examples).batch(batch_size)
    test_dataset = test_dataset.map(norm).cache().batch(batch_size)
    if not train:
        return test_dataset
    return train_dataset, test_dataset, num_train_examples


def get_mnist(batch_size=64, train=False):
    # Dataset
    dataset, metadata = tfds.load(
        'mnist',
        data_dir=dataset_path if not os.environ.get('DSDIR') else os.environ.get('SCRATCH'),
        as_supervised=True,
        with_info=True)

    train_dataset, test_dataset = dataset['train'], dataset['test']

    def norm(image, label):
        return tf.cast(image, tf.float32) / 255, label

    num_train_examples = metadata.splits["train"].num_examples
    num_test_examples = metadata.splits["test"].num_examples
    shape = metadata.features["image"].shape

    train_dataset = train_dataset.map(norm).cache().repeat().shuffle(num_train_examples).batch(batch_size)
    test_dataset = test_dataset.map(norm).cache().batch(batch_size)
    if not train:
        return test_dataset
    return train_dataset, test_dataset, num_train_examples


def get_dataset(dataset_name, train=False, batch_size=256, preprocess=None, IMAGE_SHAPE=(224, 224, 3)):
    print(dataset_name)
    if dataset_name == "CIFAR10":
        return get_cifar(batch_size=batch_size, train=train)
    elif dataset_name == "MNIST":
        return get_mnist(batch_size=batch_size, train=train)
    elif dataset_name == "IMAGENET":
        return get_imagenet(batch_size=batch_size, train=train, preprocess=preprocess, IMAGE_SHAPE=IMAGE_SHAPE)
    elif dataset_name == "IMAGENET_self":
        return get_imagenet_tf_models(train=train)
    else:
        raise ImportError


def get_official_imagenet_tf_models(model_dir=None, train=False, model_method=None, base_epoch=None,
                                    inceptionV3_preprocess=False):
    import sys
    sys.path.append("tf_models")
    from official.vision.image_classification import classifier_trainer
    from official.vision.image_classification import preprocessing

    preprocessing.inceptionV3_preprocess = inceptionV3_preprocess

    # Get params
    params_path = os.path.join(model_dir, "params.yaml")
    if os.path.exists(params_path):
        params = classifier_trainer.hyperparams.read_yaml_to_params_dict(params_path)
    else:
        params = classifier_trainer._get_params_from_flags(
            {
                'data_dir': dataset_path,
                'model_type': 'resnet',
                'dataset': 'imagenet',
                'model_dir': model_dir,
            }
        )

    # Edit params
    params.train_dataset.data_dir = params.validation_dataset.data_dir = dataset_path
    params.model_dir = model_dir
    tf.keras.mixed_precision.set_global_policy("float32")

    # Get trained model
    model, train_dataset, validation_dataset, num_examples = classifier_trainer.train_and_eval(
        strategy_override=None,
        params=params,
        return_dataset=True,
        model_method=model_method,
        base_epoch=base_epoch,
    )

    if not train:
        return validation_dataset
    return model, train_dataset, validation_dataset, num_examples


def get_imagenet_torch(batch_size=256, transform=None):
    import torchvision

    if isinstance(transform, type(None)):
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    if not os.environ.get('DSDIR') is not None:
        imagenet_data = torchvision.datasets.ImageNet(
            root=os.path.join(dataset_path, "downloads", "manual"),
            split='val',
            transform=transform)

        test_dataset = torch.utils.data.DataLoader(
            imagenet_data,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=20,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    else:
        print(f"getting imagenet at {os.environ['DSDIR'] + '/imagenet/RawImages'}")
        imagenet_dataset = torchvision.datasets.ImageNet(
            root=os.environ['DSDIR'] + '/imagenet/RawImages',
            split='val',
            transform=transform)

        drop_last = True  # set to False if it represents important information loss
        num_workers = 20  # adjust number of CPU workers per process
        persistent_workers = True  # set to False if CPU RAM must be released
        pin_memory = True  # optimize CPU to GPU transfers
        non_blocking = True  # activate asynchronism to speed up CPU/GPU transfers
        prefetch_factor = 2  # adjust number of batches to preload

        test_dataset = torch.utils.data.DataLoader(imagenet_dataset,
                                                   # sampler=data_sampler,
                                                   batch_size=batch_size,
                                                   drop_last=drop_last,
                                                   num_workers=num_workers,
                                                   persistent_workers=persistent_workers,
                                                   pin_memory=pin_memory,
                                                   prefetch_factor=prefetch_factor
                                                   )
    return test_dataset
