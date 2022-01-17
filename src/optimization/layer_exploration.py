import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import model


def layer_exploration(input_model: model.Model, log_file, max_acl, scoring_time, max_bit_range=8, min_k=2, ds_scale=1,
                      log_scale=False):
    # %% try to find result il log file
    layer_exploration_name = "layer_exploration"
    h5_path = os.path.join(log_file, f"{layer_exploration_name}.h5")
    if os.path.exists(h5_path):
        print("load layer exploration from {}".format(h5_path))
        layer_exploration_df = pd.read_hdf(h5_path, index_col=0, key="data")
    else:
        layer_exploration_df = None

    layers_vect = input_model.get_layers_of_interests()

    # %% Compute number of candidate
    total_time = 1 * 60 * 60
    n_layer = len(layers_vect)
    bit_range = int(np.ceil((total_time / scoring_time.total_seconds())) / n_layer)
    bit_range = max_bit_range
    local_exploration_scoring_count = 0
    max_k = 2 ** ((bit_range - 1) + int(np.log2(min_k)))
    print("resolved bit_range: {}".format(bit_range))
    print(f"range[{min_k}, {max_k}]")

    # %% Explore candidate for each layer
    for i, layer in enumerate(tqdm(layers_vect, total=len(layers_vect))):
        k_vector_list = []
        if log_scale:
            k_range = np.logspace(start=1, stop=max_bit_range, num=max_bit_range ** 2, base=2)
            k_range = list(dict.fromkeys(k_range.astype(int)))
        else:
            k_range = range(min_k, min(input_model.get_layer_weights(layer, force_numpy=True).size, max_k))

        for k in k_range:
            k_integer = int(k)
            if isinstance(layer_exploration_df, pd.DataFrame) and layer_exploration_df[
                (layer_exploration_df.layer == layer.name) &
                (layer_exploration_df.k == k_integer)
            ].size > 0:
                continue

            # if already explored
            if k_integer > input_model.get_layer_weights(layer, force_numpy=True).size:
                continue

            k_vector = [None for _ in input_model.get_layers_of_interests()]
            k_vector[i] = k_integer
            k_vector_list.append(k_vector)

        # Score approximate network
        if k_vector_list:
            current_layer_exploration_df = input_model.score_approximation_list(k_vector_list, ds_scale=ds_scale,
                                                                                verbose=False,
                                                                                origin_str="layer_exploration")

            # retro-compatibility
            current_layer_exploration_df["k"] = current_layer_exploration_df.k_vector.apply(
                lambda x: next((val for val in x if val)))
            current_layer_exploration_df["inertia"] = current_layer_exploration_df.inertia_vector.apply(sum)
            current_layer_exploration_df["layer"] = layer.name

            # Concat
            layer_exploration_df = pd.concat(
                [
                    layer_exploration_df,
                    current_layer_exploration_df
                ]
            )

            # log & save data
            layer_exploration_df.to_hdf(h5_path, key="data")
            local_exploration_scoring_count += len(k_vector_list)
    print("local optimization complexity: {}".format(local_exploration_scoring_count))

    # %% Filter pareto efficient candidate by looking at the number of bit required for indexing shared weights
    k_list = []
    iteration = 0
    while len(k_list) < len(layers_vect):
        filtered_layer_exploration = layer_exploration_df[
            layer_exploration_df.accuracy > max_acl * input_model.baseline_accuracy].copy()
        for layer in layers_vect:
            if filtered_layer_exploration[filtered_layer_exploration.layer == layer.name].size == 0:
                print(f"Layer {layer.name} has no candidate with accuracy < {max_acl:.2%}\n"
                      f"Smallest accuracy loss is {layer_exploration_df[layer_exploration_df.layer == layer.name].accuracy_loss.min():.2%}")
        filtered_layer_exploration = filtered_layer_exploration.reset_index(drop=True)
        filtered_layer_exploration["nb_bit"] = np.ceil(np.log2(filtered_layer_exploration.k))
        # Group by
        grouped = filtered_layer_exploration.groupby(["layer", "nb_bit"])
        # Select best candidate based on accuracy loss
        candidates_layers = filtered_layer_exploration.loc[grouped.accuracy_loss.idxmin().values][
            ["layer", "nb_bit", "k", "accuracy_loss"]]
        # Convert candidate to a list of k
        k_list = candidates_layers.groupby("layer")["k"].apply(list)

        print(f"list of K for {max_acl:.2%} affordable accuracy: {len(k_list)}")

        if len(k_list) < len(layers_vect):
            if iteration < 4:
                iteration += 1
            else:
                break
            max_acl *= 0.9
            print(f"Some layer don't have any candidate under contraints, use {max_acl:.2%} as affordable accuracy")

    # Resolve number of combination
    combination = 1
    for a in k_list:
        combination *= len(a)
    return k_list, filtered_layer_exploration
