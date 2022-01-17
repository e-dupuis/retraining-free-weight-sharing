import os
from datetime import datetime

import numpy as np
import torch
from fast_pytorch_kmeans import KMeans
from src import model


@torch.no_grad()
def kmeans_torch(layer_weights, number_cluster):
    kmeans = KMeans(n_clusters=number_cluster, mode='euclidean', verbose=0)
    weights = layer_weights if isinstance(layer_weights, torch.Tensor) else torch.tensor(layer_weights, device='cuda')
    init_vector = torch.linspace(start=float(torch.min(weights)), end=float(torch.max(weights)),
                                 steps=number_cluster).reshape(-1, 1).cuda()
    assignments = kmeans.fit_predict(
        weights.reshape(-1, 1),
        centroids=init_vector
    )
    centroids = kmeans.centroids
    inertia = torch.sum(torch.square(centroids[assignments] - weights.reshape(-1, 1))).detach().cpu().numpy()
    return assignments, centroids, inertia


@torch.no_grad()
def apply_1d_kmeans(layer_weights, number_cluster, force_cpu=False, anti_zero_drift=False, scope="layer"):
    if anti_zero_drift:
        non_zeros = np.nonzero(layer_weights)
        # print('non zero   size ratio=', 1 - layer_weights[non_zeros].size / layer_weights.size)
        layer_weights[non_zeros], inertia = apply_1d_kmeans(
            layer_weights[non_zeros], number_cluster - 1, force_cpu=force_cpu, anti_zero_drift=False
        )
        return layer_weights, inertia
    assert (isinstance(number_cluster, (
        int, np.int, np.int32, np.int64,
        torch.IntType))), f"Number of cluster type ({type(number_cluster)})is not valid"

    if scope == "layer":
        assert (0 < number_cluster <= len(
            layer_weights.reshape(-1, 1))), "Number of cluster value is not valid 0 < {} < {}".format(
            number_cluster,
            len(layer_weights.reshape(-1, 1)))

        if number_cluster == 1:
            clusterized = np.full(layer_weights.shape, np.average(layer_weights))
            inertia = float(np.sum(np.power((layer_weights - clusterized).flatten(), 2)))
            return clusterized, inertia

        assignments, centroids, inertia = kmeans_torch(layer_weights, number_cluster)
        return centroids[assignments].reshape(layer_weights.shape), inertia

    elif scope == "channel":
        inertia_list = []
        for i, channel in enumerate(layer_weights):
            if not 0 < number_cluster <= len(channel.reshape(-1, 1)):
                print(f"Number of cluster too large for channel with num elements={len(channel.reshape(-1, 1))}")
                return apply_1d_kmeans(
                    layer_weights, number_cluster, force_cpu=force_cpu, anti_zero_drift=False, scope="layer")
            assignments, centroids, inertia = kmeans_torch(channel, number_cluster)
            channel = centroids[assignments].reshape(channel.shape)
            inertia_list.append(inertia)
            layer_weights[i] = channel if isinstance(layer_weights,
                                                     torch.Tensor) and layer_weights.is_cuda else channel.cpu().numpy()
        return layer_weights, sum(inertia_list)

    elif scope == "kernel":
        inertia_list = []
        for i, channel in enumerate(layer_weights):
            for j, kernel in enumerate(channel):
                if not 0 < number_cluster <= len(kernel.reshape(-1, 1)):
                    print(f"Number of cluster too large for kernel with num elements={len(kernel.reshape(-1, 1))}")
                    return apply_1d_kmeans(
                        layer_weights, number_cluster, force_cpu=force_cpu, anti_zero_drift=False, scope="channel")
                assignments, centroids, inertia = kmeans_torch(kernel, number_cluster)
                kernel = centroids[assignments].reshape(kernel.shape)
                channel[j] = kernel if isinstance(channel,
                                                  torch.Tensor) and "cuda" in channel.is_cuda else kernel.cpu().numpy()
                inertia_list.append(inertia)
            layer_weights[i] = channel
        return layer_weights, sum(inertia_list)


@torch.no_grad()
def cluster_approx_layer(input_model: model.Model, layer, k=16, anti_zero_drift=False, scope="layer"):
    # Extract weights and apply clustering
    tic = datetime.now()
    weights = input_model.get_layer_weights(layer)
    clustered_weights, inertia = apply_1d_kmeans(weights, k, anti_zero_drift=anti_zero_drift, scope=scope)
    time = datetime.now() - tic

    # Set Weights (& Biases)
    input_model.set_layer_weights(layer, clustered_weights)
    return float(inertia), time


@torch.no_grad()
def cluster_approx_network(input_model, layers_vector, k_vector, anti_zero_drift=False, scope="layer"):
    assert (len(layers_vector) == len(k_vector))
    inertia_vect = []
    tic = datetime.now()
    for layer, k in zip(layers_vector, k_vector):
        if isinstance(k, type(None)):
            continue
        # Approximate layer
        inertia, time = cluster_approx_layer(input_model, layer, k, anti_zero_drift=anti_zero_drift, scope=scope)
        inertia_vect.append(inertia)
    clustering_time = datetime.now() - tic
    return inertia_vect, clustering_time
