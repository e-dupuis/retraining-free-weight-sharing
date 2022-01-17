import copy
import gc
import math
import os
import tempfile
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from src import log
from src import dataset as ds

tf.get_logger().setLevel('ERROR')

model_dir = None

class Model(ABC):
    def __init__(self, model, test_dataset, mlperf_category="heavy", scoring_batch_size=64):
        self.model = model
        self.test_dataset = test_dataset
        self.baseline_accuracy, self.scoring_time = self.score_model()
        self.evaluate_list = None
        self.mlperf_category = mlperf_category
        self.scoring_batch_size = scoring_batch_size

    @abstractmethod
    def get_weights_count(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    @staticmethod
    @abstractmethod
    def get_layer_weights(module, force_numpy=False):
        pass

    @staticmethod
    @abstractmethod
    def set_layer_weights(module, weights):
        pass

    @abstractmethod
    def get_layers_of_interests(self):
        pass

    @abstractmethod
    def score_model(self, ds_scale=1):
        pass

    def score_approximation_list(self, k_vector_list, ds_scale=1, verbose=True, origin_str="unspecified", h5_path=None,
                                 scope="layer"):
        from src.approx import clustering
        from src import scoring
        import uuid

        layers_of_interests = self.get_layers_of_interests()

        # Store initial weights of each of the layers
        baseline_weights = self.get_weights()

        # Init values
        models_list = []
        unique_id_list = []
        progress_bar = tqdm(k_vector_list) if verbose else k_vector_list

        if h5_path and os.path.exists(h5_path):
            evaluate_df = pd.read_hdf(h5_path, key="data")
            evaluate_df = evaluate_df[evaluate_df["accuracy_loss"].notnull()]
        else:
            evaluate_df = None

        for i, k_vector in enumerate(progress_bar):
            # Check validity
            assert (len(k_vector) == len(layers_of_interests))

            # Generate unique id
            uid = uuid.uuid4()

            # Reset weights and apply clustering
            self.set_weights(baseline_weights)

            inertia_vect, clustering_time = clustering.cluster_approx_network(
                self,
                layers_of_interests,
                k_vector,
                scope=scope,
            )

            # Add approximated model to list
            if self.scoring_batch_size > 0:
                models_list.append(copy.deepcopy(self.model))
                unique_id_list.append(uid)
                accuracy = scoring_time = None
            else:
                tic = datetime.now()
                accuracy = self.evaluate_list(self.test_dataset, self.model, ds_scale=ds_scale)
                scoring_time = (datetime.now() - tic).total_seconds()

            # Store data
            evaluate_df = log.append_data(
                evaluate_df,
                {
                    'unique_id': uid,
                    'k_vector': [list(k_vector)],
                    'inertia_vector': [list(inertia_vect)],
                    'compression_rate': -1 * scoring.evaluate_model_compression(
                        self,
                        k_vector,
                        is_clustered=True) if not None in k_vector else -1,
                    'clustering_size': -1 * scoring.evaluate_model_compression(
                        self,
                        k_vector,
                        is_clustered=True,
                        ratio=False) if not None in k_vector else -1,
                    'clustering_time': clustering_time.total_seconds(),
                    'origin': origin_str,
                    'accuracy_loss': None if self.scoring_batch_size > 0 else self.baseline_accuracy - accuracy,
                    'accuracy': accuracy,
                    'scoring_time': scoring_time,
                }
            )

            # Trigger model list evaluation
            if len(models_list) and ((i + 1) % self.scoring_batch_size == 0 or i == len(k_vector_list) - 1):
                # Approximate Model List Scoring
                tic = datetime.now()
                accuracy_list = self.evaluate_list(self.test_dataset, models_list, ds_scale=ds_scale)
                scoring_time = datetime.now() - tic
                scoring_time = scoring_time.total_seconds()

                # Update evaluation dataframe
                evaluate_df.set_index("unique_id", inplace=True)
                evaluate_df.loc[unique_id_list, 'accuracy_loss'] = self.baseline_accuracy - accuracy_list
                evaluate_df.loc[unique_id_list, 'accuracy'] = accuracy_list
                evaluate_df.loc[unique_id_list, 'scoring_time'] = scoring_time / len(accuracy_list)

                # Reset lists
                gc.collect()
                for approx_model in models_list:
                    del approx_model
                models_list = []
                unique_id_list = []

                # log & save data
                if not isinstance(h5_path, type(None)):
                    evaluate_df.to_hdf(h5_path, key="data")

        evaluate_df['accuracy_loss'] = evaluate_df['accuracy_loss'].astype(float)
        evaluate_df['accuracy'] = evaluate_df['accuracy'].astype(float)
        assert evaluate_df['accuracy_loss'].dtype == 'float'
        assert evaluate_df['accuracy'].dtype == 'float'
        if not isinstance(h5_path, type(None)):
            evaluate_df.to_hdf(h5_path, key="data")
        return evaluate_df


class TensorflowModel(Model):
    def get_weights_count(self):
        return [w.size for w in self.get_weights()]

    def __init__(self, model, test_dataset, mlperf_category="heavy", scoring_batch_size=64):
        super(TensorflowModel, self).__init__(model, test_dataset, mlperf_category, scoring_batch_size)
        from src import scoring
        self.evaluate_list = scoring.tf_evaluate

    @staticmethod
    def get_layer_weights(module, force_numpy=False):
        if "use_bias" in module.get_config() and module.get_config()['use_bias'] or \
                isinstance(module.get_weights(), list):
            return module.get_weights()[0]
        else:
            return module.get_weights()

    @staticmethod
    def set_layer_weights(module: tf.keras.layers.Layer, weights):
        if "use_bias" in module.get_config() and module.get_config()['use_bias'] or \
                isinstance(module.get_weights(), list):
            if isinstance(weights, list) and not isinstance(weights, np.ndarray):
                try:
                    module.set_weights(weights)
                except TypeError:
                    module.set_weights([w.detach().cpu().numpy() for w in weights])
            else:
                try:
                    module.set_weights(
                        [weights] if len(module.get_weights()) == 1 else [weights, module.get_weights()[1]])
                except TypeError:
                    module.set_weights(
                        [weights.detach().cpu().numpy()] if len(module.get_weights()) == 1 else [
                            weights.detach().cpu().numpy(), module.get_weights()[1]])

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        try:
            self.model.set_weights(weights)
        except TypeError:
            self.model.set_weights([w.detach().cpu().numpy() for w in weights])

    def get_layers_of_interests(self):
        return get_tf_layers_of_interests(self.model, None)

    def score_model(self, ds_scale=1):
        from src import scoring
        tic = datetime.now()
        return scoring.tf_evaluate(self.test_dataset, self.model, ds_scale=ds_scale), datetime.now() - tic


class TorchModel(Model):
    def get_weights_count(self):
        return [torch.numel(w) for w in self.get_weights()]

    def __init__(self, model, test_dataset, mlperf_category="heavy", scoring_batch_size=64):
        super(TorchModel, self).__init__(model, test_dataset, mlperf_category, scoring_batch_size)
        from src import scoring
        self.evaluate_list = scoring.pytorch_evaluate

    def get_weights(self, force_numpy=False):
        result = [layer.weight if not force_numpy else layer.weight.detach().cpu().numpy() for layer in
                  self.get_layers_of_interests()]
        return result

    @torch.no_grad()
    def set_weights(self, weights):
        i = 0
        for layer in self.get_layers_of_interests():
            try:
                assert (layer.weight.shape == weights[i].shape)
                layer.weight.data = weights[i]
            except TypeError:
                assert (layer.weight.cpu().detach().numpy().shape == weights[i].shape)
                layer.weight.data = torch.Tensor(weights[i]).cuda()
            i += 1

    @staticmethod
    def get_layer_weights(module, force_numpy=False):
        return module.weight if not force_numpy else module.weight.detach().cpu().numpy()

    @staticmethod
    @torch.no_grad()
    def set_layer_weights(module, weights):
        try:
            module.weight = torch.nn.parameter.Parameter(weights)
        except TypeError:
            module.weight = torch.nn.parameter.Parameter(torch.Tensor(weights).cuda())

    def get_layers_of_interests(self):
        list = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                layer.name = name
                list.append(layer)
        return list

    def score_model(self, ds_scale=1):
        from src import scoring
        tic = datetime.now()
        return scoring.pytorch_evaluate(self.test_dataset, self.model, ds_scale=ds_scale), datetime.now() - tic


def get_model(model_name, evaluate=True, get_train=False) -> (Model, float, datetime):
    global model_dir
    # Save paths
    trained_model_path = "trained_models"
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    save_path = model_name

    import fnmatch
    if model_name == "lenet_mnist":
        dataset = "MNIST"
        model_define = lenet
        input_model, test_dataset = get_self_trained(dataset, model_define, trained_model_path, save_path, get_train)
        input_model = TensorflowModel(input_model, test_dataset, scoring_batch_size=0)
    elif model_name == "resnet50v2_cifar10":
        dataset = "CIFAR10"
        model_define = resnet50v2_cifar
        input_model, test_dataset = get_self_trained(dataset, model_define, trained_model_path, save_path, get_train)
        input_model = TensorflowModel(input_model, test_dataset)
    elif model_name == "resnet18v2_cifar10":
        dataset = "CIFAR10"
        model_define = resnet18v2_cifar
        input_model, test_dataset = get_self_trained(dataset, model_define, trained_model_path, save_path, get_train)
        input_model = TensorflowModel(input_model, test_dataset)
    elif model_name == "resnet18v2_plain_cifar10":
        dataset = "CIFAR10"
        model_define = resnet18v2_plain_cifar
        input_model, test_dataset = get_self_trained(dataset, model_define, trained_model_path, save_path, get_train)
        input_model = TensorflowModel(input_model, test_dataset)

    elif model_name == "mobilenetv1_imagenet":
        dataset_name = "IMAGENET"
        shape = (224, 224, 3)
        input_model = tf.keras.applications.MobileNet()
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        test_dataset = ds.get_dataset(dataset_name, train=get_train, preprocess=preprocess, IMAGE_SHAPE=shape)
        input_model = TensorflowModel(input_model, test_dataset, mlperf_category="light")
    elif model_name == "mobilenetv2_imagenet":
        dataset_name = "IMAGENET"
        shape = (224, 224, 3)
        cnn = tf.keras.applications.MobileNetV2(input_shape=shape)
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        test_dataset = ds.get_dataset(dataset_name, train=get_train, preprocess=preprocess, IMAGE_SHAPE=shape)
        input_model = TensorflowModel(cnn, test_dataset, mlperf_category="light")
    elif model_name == "mobilenetv2_imagenet_pytorch":
        cnn = torchvision.models.mobilenet_v2(pretrained=True).cuda()
        test_dataset = ds.get_imagenet_torch()
        input_model = TorchModel(cnn, test_dataset, mlperf_category="light")
    elif model_name == "mobilenetv3_small_imagenet":
        dataset_name = "IMAGENET"
        shape = (224, 224, 3)
        input_model = tf.keras.applications.MobileNetV3Small(input_shape=shape)
        preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
        test_dataset = ds.get_dataset(dataset_name, train=get_train, preprocess=preprocess, IMAGE_SHAPE=shape)
        input_model = TensorflowModel(input_model, test_dataset, mlperf_category="light")
    elif model_name == "mobilenetv3_large_imagenet":
        dataset_name = "IMAGENET"
        shape = (224, 224, 3)
        input_model = tf.keras.applications.MobileNetV3Large(input_shape=shape)
        preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
        test_dataset = ds.get_dataset(dataset_name, train=get_train, preprocess=preprocess, IMAGE_SHAPE=shape)
        input_model = TensorflowModel(input_model, test_dataset, mlperf_category="light")
    elif model_name == "mobilenetv3_small_imagenet_pytorch":
        input_model = torchvision.models.mobilenet_v3_small(pretrained=True).cuda()
        test_dataset = ds.get_imagenet_torch()
        input_model = TorchModel(input_model, test_dataset, mlperf_category="light")
    elif model_name == "mobilenetv3_large_imagenet_pytorch":
        input_model = torchvision.models.mobilenet_v3_large(pretrained=True).cuda()
        test_dataset = ds.get_imagenet_torch()
        input_model = TorchModel(input_model, test_dataset, mlperf_category="light")
    elif model_name == "efficientnetb0_imagenet":
        dataset_name = "IMAGENET"
        shape = (224, 224, 3)
        input_model = tf.keras.applications.EfficientNetB0(input_shape=shape)
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        test_dataset = ds.get_dataset(dataset_name, train=get_train, preprocess=preprocess, IMAGE_SHAPE=shape)
        input_model = TensorflowModel(input_model, test_dataset)
    elif fnmatch.fnmatch(model_name, "efficientnetb?_imagenet_pytorch"):
        from efficientnet_pytorch import EfficientNet
        import PIL

        if "b0" in model_name:
            model_name = 'EfficientNet-B0'
            batch_size = 512
        elif "b1" in model_name:
            model_name = 'EfficientNet-B1'
            batch_size = 512
        elif "b2" in model_name:
            model_name = 'EfficientNet-B2'
            batch_size = 256
        elif "b3" in model_name:
            model_name = 'EfficientNet-B3'
            batch_size = 256
        elif "b4" in model_name:
            model_name = 'EfficientNet-B4'
            batch_size = 256
        elif "b5" in model_name:
            model_name = 'EfficientNet-B5'
            batch_size = 256
        elif "b6" in model_name:
            model_name = 'EfficientNet-B6'
            batch_size = 128
        elif "b7" in model_name:
            model_name = 'EfficientNet-B7'
            batch_size = 64
        else:
            raise ValueError(f"Impossible to find matching Efficientnet for {model_name}")

        input_model = EfficientNet.from_pretrained(model_name.lower())
        image_size = EfficientNet.get_image_size(model_name.lower())
        print(model_name, batch_size, image_size, batch_size * image_size ** 2 / 1e9)
        input_model.to('cuda')
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size, PIL.Image.BICUBIC),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        test_dataset = ds.get_imagenet_torch(batch_size, transform)
        input_model = TorchModel(input_model, test_dataset, mlperf_category="light")
    elif model_name == "resnet50v2_imagenet":
        dataset_name = "IMAGENET"
        shape = (224, 224, 3)
        input_model = tf.keras.applications.ResNet50V2(input_shape=shape)
        preprocess = tf.keras.applications.resnet_v2.preprocess_input
        test_dataset = ds.get_dataset(dataset_name, train=get_train, preprocess=preprocess, IMAGE_SHAPE=shape)
        input_model = TensorflowModel(input_model, test_dataset)
    elif model_name == "resnet50_imagenet_pytorch":
        input_model = torchvision.models.resnet50(pretrained=True).cuda()
        test_dataset = ds.get_imagenet_torch()
        input_model = TorchModel(input_model, test_dataset)
    elif model_name == model_name == "inceptionv3_imagenet_pytorch":
        input_model = torchvision.models.inception_v3(pretrained=True).cuda()
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(299),
            torchvision.transforms.CenterCrop(299),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        test_dataset = ds.get_imagenet_torch(transform=transform)
        input_model = TorchModel(input_model, test_dataset)
    elif model_name == "inceptionv1_imagenet_pytorch":
        input_model = torchvision.models.googlenet(pretrained=True).cuda()
        test_dataset = ds.get_imagenet_torch()
        input_model = TorchModel(input_model, test_dataset)
    else:
        err = "Unknown model: {}".format(model_name)
        raise ImportError(err)

    if evaluate:
        print(f"testing model {model_name}")
        accuracy, scoring_time = input_model.score_model()
        return input_model, accuracy, scoring_time
    return input_model


def save_and_compute_gz_size(model, extension=".h5"):
    if 'torch' in str(type(model)):
        _, file = tempfile.mkstemp()
        torch.save(model.state_dict(), file)
    else:
        _, file = tempfile.mkstemp(extension)
        if extension == ".tflite":
            with open(file, 'wb') as f:
                f.write(model)
        else:
            tf.keras.models.save_model(model, file, include_optimizer=False)
    return get_gzipped_model_size(file)


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)


def get_self_trained(dataset, model_define, model_path, save_path, get_train=False):
    try:
        input_model, test_dataset = get_model_from_path(os.path.join(model_path, save_path), dataset)
    except:
        # Define model
        input_model = model_define()

        # Compile model
        input_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        # Get training DS
        batch_size = 128
        train_dataset, test_dataset, num_train_examples = ds.get_dataset(dataset, train=True, batch_size=batch_size)

        # Training & Benchmark
        tic = log.toc()
        input_model.fit(train_dataset, validation_data=test_dataset, epochs=30,
                        steps_per_epoch=math.ceil(num_train_examples / batch_size))
        print("training time: ", log.toc(tic))

        # Save in model path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        input_model.save(os.path.join(model_path, save_path), save_format="tf")
    if get_train:
        return input_model, test_dataset, train_dataset, num_train_examples
    return input_model, test_dataset


def get_model_from_path(model_path, dataset=None):
    input_model = tf.keras.models.load_model(model_path)
    input_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    if dataset:
        test_dataset = ds.get_dataset(dataset)
        return input_model, test_dataset
    else:
        return input_model


def get_model_from_log(log_path):
    model_name = [x for x in os.listdir(log_path) if ".pb" in x][0]
    if not model_name:
        raise ImportError("No model found in {}".format(log_path))
    return get_model_from_path(os.path.join(log_path, model_name))


def lenet():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=120, activation='relu'),
        tf.keras.layers.Dense(units=84, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax'),
    ])


def resnet50v2_cifar():
    shape = (32, 32, 3)
    classes = 10
    model = tf.keras.applications.ResNet50V2(
        weights=None,
        input_shape=shape,
        classes=classes,
    )
    return model


def resnet18v2_cifar():
    shape = (None, 32, 32, 3)
    classes = 10

    class BottleNeck(tf.keras.layers.Layer):
        def __init__(self, filter_num, stride=1):
            super(BottleNeck, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding='same')
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(3, 3),
                                                strides=stride,
                                                padding='same')
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding='same')
            self.bn3 = tf.keras.layers.BatchNormalization()

            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())

        def call(self, inputs, training=None, **kwargs):
            residual = self.downsample(inputs)

            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = self.conv2(x)
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)
            x = self.conv3(x)
            x = self.bn3(x, training=training)

            output = tf.nn.relu(tf.keras.layers.add([residual, x]))

            return output

    def make_bottleneck_layer(filter_num, blocks, stride=1):
        res_block = tf.keras.Sequential()
        res_block.add(BottleNeck(filter_num, stride=stride))

        for _ in range(1, blocks):
            res_block.add(BottleNeck(filter_num, stride=1))

        return res_block

    class ResNetTypeII(
        tf.keras.Model):  # https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
        def __init__(self, layer_params):
            super(ResNetTypeII, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(7, 7),
                                                strides=2,
                                                padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                   strides=2,
                                                   padding="same")
            self.layer1 = make_bottleneck_layer(filter_num=64,
                                                blocks=layer_params[0])
            self.layer2 = make_bottleneck_layer(filter_num=128,
                                                blocks=layer_params[1],
                                                stride=2)
            self.layer3 = make_bottleneck_layer(filter_num=256,
                                                blocks=layer_params[2],
                                                stride=2)
            self.layer4 = make_bottleneck_layer(filter_num=512,
                                                blocks=layer_params[3],
                                                stride=2)

            self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
            self.fc = tf.keras.layers.Dense(units=classes, activation=tf.keras.activations.softmax)

        def call(self, inputs, training=None, mask=None):
            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = self.pool1(x)
            x = self.layer1(x, training=training)
            x = self.layer2(x, training=training)
            x = self.layer3(x, training=training)
            x = self.layer4(x, training=training)
            x = self.avgpool(x)
            output = self.fc(x)

            return output

    def resnet_18():
        return ResNetTypeII(layer_params=[1, 1, 1, 1])

    def resnet_34():
        return ResNetTypeII(layer_params=[2, 2, 2, 2])

    model = resnet_18()

    model.build(input_shape=shape)
    return model


def resnet18v2_plain_cifar():
    shape = (None, 32, 32, 3)
    classes = 10

    def ResNetTypeII(layer_params):
        # Input layer
        resnet = tf.keras.Sequential()
        resnet.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", activation="relu"))
        resnet.add(tf.keras.layers.BatchNormalization())
        resnet.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"))

        def create_bottleneck_layer(resnet, filter_num, stride):
            resnet.add(tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride))
            resnet.add(tf.keras.layers.BatchNormalization())

            resnet.add(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding='same',
                                              activation="relu"))
            resnet.add(tf.keras.layers.BatchNormalization())
            resnet.add(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same',
                                              activation="relu"))
            resnet.add(tf.keras.layers.BatchNormalization())
            resnet.add(tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding='same',
                                              activation="relu"))
            resnet.add(tf.keras.layers.BatchNormalization())

        # Residual blocks
        def add_bottleneck_layer(resnet, filter_num, blocks, stride=1):
            create_bottleneck_layer(resnet, filter_num, stride=stride)

            for _ in range(1, blocks):
                create_bottleneck_layer(resnet, filter_num, stride=1)

        # add_bottleneck_layer(resnet=resnet, filter_num=64, blocks=layer_params[0])
        # add_bottleneck_layer(resnet=resnet, filter_num=128, blocks=layer_params[1], stride=2)
        # add_bottleneck_layer(resnet=resnet, filter_num=256, blocks=layer_params[2], stride=2)
        # add_bottleneck_layer(resnet=resnet, filter_num=512, blocks=layer_params[3], stride=2)

        # Output Layer
        resnet.add(tf.keras.layers.GlobalAveragePooling2D())
        resnet.add(tf.keras.layers.Dense(units=classes, activation=tf.keras.activations.softmax))
        return resnet

    def resnet_18():
        return ResNetTypeII(layer_params=[1, 1, 1, 1])

    def resnet_34():
        return ResNetTypeII(layer_params=[2, 2, 2, 2])

    model = resnet_18()
    model.build(input_shape=shape)
    return model


def get_tf_layers_of_interests(module, list):
    fc_conv_layer_list = list if list else []
    sub_layer_list = module.layers if hasattr(module, "layers") else module._layers if hasattr(
        module,
        "_layers") else None
    for layer in sub_layer_list:
        if isinstance(layer, tf.keras.layers.Conv2D) or \
                isinstance(layer, tf.keras.layers.Dense) or \
                (
                        "layer" in layer.get_config() and "class_name" in layer.get_config()["layer"] and
                        (
                                layer.get_config()["layer"]["class_name"] == "Conv2D" or
                                layer.get_config()["layer"]["class_name"] == "Dense"
                        )
                ):
            fc_conv_layer_list.append(layer)
        elif hasattr(layer, "layers") or "BottleNeck" in type(layer).__name__:
            get_tf_layers_of_interests(module, fc_conv_layer_list)
        # else:
        # print("no", layer)
    return fc_conv_layer_list
