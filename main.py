# %%
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import mlflow

from src import scoring, log, model, utils, plot
from src.optimization import combination, layer_exploration

print(f"Tensorflow version={tf.__version__}")
tf.get_logger().setLevel('INFO')


# %%
def explore_cnn(log_file, optimization, cnn_name="resnet18v2_cifar10", N_random_sampling=2000,
                with_local_optimization=True, ds_scale=1):
    # %% Inputs
    # Get model & test dataset
    input_model, original_accuracy, scoring_time = model.get_model(cnn_name)
    layers_vect = input_model.get_layers_of_interests()
    max_acl = 99 / 100 if input_model.mlperf_category == "heavy" else 98 / 100

    # Compute Baseline size
    original_size = scoring.evaluate_model_compression(input_model, layers_vect, ratio=False)

    # Log metrics to mlflow
    mlflow.log_params({
        "mlperf_category": input_model.mlperf_category,
    })
    weights_count = sum(input_model.get_weights_count())
    mlflow.log_metrics({
        "baseline_accuracy": original_accuracy,
        "baseline_weight_size": original_size,
        "baseline_weight_count": weights_count,
        "baseline_scoring_time": scoring_time.total_seconds(),
    })

    original_size = scoring.evaluate_model_compression(input_model, layers_vect, ratio=False)
    print(
        f"Targeting model {cnn_name} ({input_model.mlperf_category}) with {len(layers_vect)} layers and {weights_count} weights")
    print(f"top1 accuracy: {original_accuracy:.2%} scoring_time: {scoring_time}")

    pipeline = [(original_accuracy, original_size, "original")]
    print(scoring.evaluate_model_compression(
        input_model=input_model,
        ratio=True,
    )
    )

    if optimization == "two_step_optimization":
        # %% Layer Exploration
        if with_local_optimization:
            max_bit_range = 10
            k_list, layer_exploration_df = layer_exploration.layer_exploration(input_model, log_file, max_acl,
                                                                               scoring_time,
                                                                               max_bit_range=max_bit_range,
                                                                               ds_scale=ds_scale, log_scale=True)
            num_combination = np.prod(k_list.apply(len).values.astype(float))
            print(
                f"Combinations before {(max_bit_range ** 2) ** len(input_model.get_layers_of_interests()):.2e},"
                f" and after layer-wise {num_combination :.2e}")
            assert num_combination > 0
        else:
            k_list = None
            num_combination = None

        # %% Create layers plot
        plot.plot_layer_exploration(layers_vect, layer_exploration_df, k_list, save_path=log_file)

        # %% Combination Exploration
        if num_combination > 5e4:
            from doepy import build
            exploration_df = combination.try_all_sampling(
                input_model,
                k_list,
                log_file,
                num_samples=N_random_sampling,
                ds_scale=0.1,
            )
        else:
            from doepy import build
            exploration_df = combination.search_space_sampling(
                input_model,
                k_list,
                log_file,
                num_samples=int(num_combination),
                sampling_method=build.full_fact,
                ds_scale=0.1,
            )

        # Meta heuristic optimization
        population_df = combination.population_expansion(
            input_model=input_model,
            layer_df=layer_exploration_df,
            k_list=k_list,
            var_select="inertia",
            population_df=exploration_df,
            log_file=log_file,
            ds_scale=ds_scale,
            max_acl=max_acl,
            real_evaluation=True,
            time_bounded=False,
            sampling=None
        )

        exploration_df = population_df

        plot.plot_population_expansion_result(exploration_df, max_acl, log_file)


    # %% Pareto optimal selection of candidates
    # exploration_df = exploration_df.drop(columns="scoring_time")
    exploration_df["pareto"] = utils.find_pareto_frontiers(exploration_df, ["compression_rate", "accuracy_loss"],
                                                           n_front=1)
    pareto_exploration_df = exploration_df[exploration_df["pareto"] == 0]
    plot.plot_pareto_frontier(pareto_exploration_df, exploration_df, log_file, max_acl)

    # Score optimal
    weight_sharing_exploration_df = input_model.score_approximation_list(pareto_exploration_df.k_vector.values,
                                                                         ds_scale=None, verbose=True,
                                                                         origin_str="final_scoring")

    # Re-select optimals
    weight_sharing_exploration_df["pareto"] = utils.find_pareto_frontiers(weight_sharing_exploration_df,
                                                                          ["compression_rate", "accuracy_loss"],
                                                                          n_front=1)
    pareto_exploration_df = weight_sharing_exploration_df[weight_sharing_exploration_df["pareto"] == 0]

    # Extract data
    clustered_accuracy = pareto_exploration_df["accuracy"].values
    clustered_size = pareto_exploration_df["clustering_size"].values
    pipeline.append((clustered_accuracy, clustered_size, "clustered"))

    # Log metrics to mlflow
    for i in range(len(clustered_accuracy)):
        mlflow.log_metrics({
            f"weight_sharing_accuracy_{i}": pareto_exploration_df["accuracy"].values[i],
            f"weight_sharing_weight_size_{i}": pareto_exploration_df["clustering_size"].values[i],
            f"weight_sharing_cr_{i}": pareto_exploration_df["compression_rate"].values[i],
            f"weight_sharing_al_{i}": pareto_exploration_df["accuracy_loss"].values[i]
        }, step=i)
    print(pipeline)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description='Execute local exploration for the specified network')
    parser.add_argument('--network_dataset', type=str, default="lenet_mnist",
                        help='The name of the network '
                             '\n|lenet_mnist'
                             '\n|resnet18v2_cifar10'
                             '\n|resnet50v2_cifar10'
                             '\n|mobilenetv2_imagenet'
                             '\n|resnet18v2_plain_cifar10',
                        )
    parser.add_argument('--log_file', default=None, help='Path to the log folder')
    parser.add_argument('--num_samples', default=None, type=int,
                        help='Number of samples combinations for random subsampling')
    parser.add_argument('--mode', default="exploration", type=str, help="exploration or benchmark")
    parser.add_argument('--with_local_optimization', default=1, type=int,
                        help='Apply local optimization before combination')
    parser.add_argument('--optimization', default="two_step_optimization", type=str,
                        help='choose between \"two_step_optimization\" or \"bayesian_optimization\"')

    parser.add_argument('--dataset_scale', default=1., type=float,
                        help='proportion of the test dataset used for scoring model')

    args = parser.parse_args()
    if not args.log_file:
        args.log_file = log.get_log_file(args.network_dataset)
    if not os.path.exists(args.log_file):
        os.makedirs(args.log_file)

    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "exploration":

        mlflow.set_experiment("weight-sharing")
        mlflow.start_run()
        mlflow.log_params({
            "log_file": args.log_file,
            "cnn_name": args.network_dataset,
            "N_random_sampling": args.num_samples,
            "with_local_optimization": args.with_local_optimization,
            "ds_scale": args.dataset_scale,
            "optimization": args.optimization,
        })

        explore_cnn(log_file=args.log_file, optimization=args.optimization, cnn_name=args.network_dataset,
                    N_random_sampling=args.num_samples, with_local_optimization=args.with_local_optimization,
                    ds_scale=args.dataset_scale)
        print("log_artifacts")
        mlflow.log_artifact(args.log_file)
        mlflow.end_run()

    else:
        print(args)
