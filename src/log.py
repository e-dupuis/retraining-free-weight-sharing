from datetime import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

start_time = datetime.now()


def log_data(layer, distance, mean, toc_scoring, toc_clustering, data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(
            columns=[
                'layer_name',
                'distance',
                'mean',
                'num_weights',
                'scoring_time',
                'clustering_time',
            ]
        )
    data = data.append(
        {
            'layer_name': layer.name,
            'distance': distance,
            'mean': mean,
            'num_weights': np.array(layer.get_weights()).size if not layer.get_config()['use_bias'] else
            np.array(layer.get_weights())[0].size,
            'scoring_time': toc_scoring,
            'clustering_time': toc_clustering,
        }, ignore_index=True
    )
    return data


def plot(data_global_scoring, data_local_scoring):
    # All for local
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False)
    axes[0].plot('layer_name', 'distance', label='local distance(/mean)', data=data_local_scoring, marker='o',
                 color="blue", alpha=0.3)
    axes[1].plot('layer_name', 'num_weights', label='num_weights', data=data_local_scoring, marker='o', color="orange",
                 alpha=0.3)
    axes[2].plot('layer_name', 'scoring_time', label='scoring_time', data=data_local_scoring, marker='o', color="red",
                 alpha=0.3)
    axes[3].plot('layer_name', 'clustering_time', label='clustering_time', data=data_local_scoring, marker='o',
                 color="green", alpha=0.3)
    axes[0].title.set_text('Results from local scoring')
    for axe in axes:
        axe.legend()
    plt.show()
    # All for global
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False)
    axes[0].plot('layer_name', 'distance', label='global distance', data=data_global_scoring, marker='o', color="blue",
                 alpha=0.3)
    axes[1].plot('layer_name', 'num_weights', label='num_weights', data=data_global_scoring, marker='o', color="orange",
                 alpha=0.3)
    axes[2].plot('layer_name', 'scoring_time', label='scoring_time', data=data_global_scoring, marker='o', color="red",
                 alpha=0.3)
    axes[3].plot('layer_name', 'clustering_time', label='clustering_time', data=data_global_scoring, marker='o',
                 color="green", alpha=0.3)
    axes[0].title.set_text('Results from global scoring')
    for axe in axes:
        axe.legend()
    plt.show()
    # Global and local distances
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    y1 = np.concatenate(
        (
            np.array(data_local_scoring['distance'])[:-1] / (np.array(data_local_scoring['mean'])[:-1] ** 2),
            [
                np.array(data_local_scoring['distance'])[-1] / np.array(data_local_scoring['mean'])[-1]
            ]
        )
    )
    x1 = np.array(data_local_scoring['layer_name'])
    axes[0].plot(x1, y1, label='Local_distance (l2_norm)', marker='o', color="blue", alpha=0.3)
    axes[1].plot('layer_name', 'distance', label='Global distance (l2_norm)', data=data_global_scoring, marker='o',
                 color="orange",
                 alpha=0.3)
    axes[0].title.set_text('Global and local distances')
    for axe in axes:
        axe.legend()
    plt.show()
    # scoring time for local and global
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    axes[0].plot('layer_name', 'scoring_time', label='Local', data=data_local_scoring, marker='o',
                 color="blue", alpha=0.3)
    axes[1].plot('layer_name', 'scoring_time', label='Global', data=data_global_scoring, marker='o',
                 color="orange",
                 alpha=0.3)
    axes[0].title.set_text('Scoring time for local and global')
    for axe in axes:
        axe.legend()
    plt.show()

start_time
def log(data):
    print(tabulate(data, headers='keys', tablefmt='psql'))


def get_log_file(prefix=None):
    global log_file
    if not log_file:
        global start_time
        log_file = f"logs/{prefix}/log_{str(start_time)}"
        if not os.path.exists(log_file):
            os.makedirs(log_file)
    return log_file


log_file = None


def append_data(data, data_dict):
    return pd.concat(
        [
            data, pd.DataFrame(
            data_dict, index=[0]
        )
        ], ignore_index=True
    )


def toc(tic=None):
    if tic is None:
        return datetime.now()
    return datetime.now() - tic
