import os.path

import pytest
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import sys

sys.path.insert(1, '../')
from src import model
from src.optimization import layer_exploration, combination

model_list = [
    ("lenet_mnist", 0.98, "heavy"),
    ("resnet50v2_cifar10", 0.72, "heavy"),
    ("resnet18v2_cifar10", 0.73, "heavy"),
    ("resnet18v2_plain_cifar10", 0.54, "heavy"),
    ("mobilenetv1_imagenet", 0.71, "light"),
    ("mobilenetv1_0.25_imagenet_n2d2", .55, "light"),
    ("mobilenetv2_imagenet_pytorch", 0.71, "light"),
    ("mobilenetv3_small_imagenet", 0.67, "light"),
    ("mobilenetv3_large_imagenet", 0.73, "light"),
    ("mobilenetv3_small_imagenet_pytorch", 0.67, "light"),
    ("mobilenetv3_large_imagenet_pytorch", 0.73, "light"),
    ("efficientnetb0_imagenet", 0.76, "light"),
    ("efficientnetb0_imagenet_pytorch", 0.76, "light"),
    ("efficientnetb1_imagenet_pytorch", 0.78, "light"),
    ("efficientnetb2_imagenet_pytorch", 0.79, "heavy"),
    ("efficientnetb3_imagenet_pytorch", 0.81, "heavy"),
    ("resnet50_imagenet_pytorch", 0.76, "heavy"),
    ("resnet50v2_imagenet", 0.69, "heavy"),
    ("resnet50v2_imagenet_self", 0.76, "heavy"),
    ("inceptionv1_imagenet_pytorch", 0.69, "heavy"),
    ("inceptionv3_imagenet_pytorch", 0.77, "heavy"),
]


@pytest.mark.parametrize('param', model_list)
def test_full(param):
    model_name, desired_accuracy, category = param
    cnn, accuracy, time = model.get_model(model_name)
    assert accuracy >= desired_accuracy, f"{model_name} baseline accuracy issue {accuracy} < {desired_accuracy}"
    max_acl = 2 / 100 if category == "light" else 1 / 100

    print(cnn.get_layers_of_interests())

    filename = f"/tmp/test/{model_name}"
    if not os.path.exists(filename):
        os.makedirs(filename)
    else:
        import shutil
        shutil.rmtree(filename)
        os.makedirs(filename)

    # Layer exploration
    k_list, layer_exploration_df = layer_exploration.layer_exploration(cnn, filename, max_acl, time, max_bit_range=10,
                                                                       ds_scale=0.1, log_scale=True)

    assert not k_list.empty, f"{k_list}"
    assert not layer_exploration_df.empty, f"{layer_exploration_df}"

    # Combination Exploration
    exploration_df = combination.random_sampling_exploration(
        cnn,
        k_list,
        filename,
        max_samples=50,
        ds_scale=0.1,
    )

    assert not exploration_df.empty, f"{exploration_df}"


@pytest.mark.parametrize('param', model_list)
def test_scoring_list(param):
    model_name, desired_accuracy, category = param
    cnn, accuracy, time = model.get_model(model_name)

    evaluation_df = cnn.score_approximation_list([[50 + i for _ in cnn.get_layers_of_interests()] for i in range(5)],
                                                 ds_scale=0.1, verbose=False, origin_str="test",
                                                 h5_path="/tmp/data.csv")

    print(evaluation_df.accuracy_loss)
