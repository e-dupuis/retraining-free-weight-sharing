import os

import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
else:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import pandas as pd

def plot_quantization_study(quantization_exploration_df, log_file):
    ax = quantization_exploration_df.plot(x="clustering_CR", y="clustering_accuracy_loss", label="Clustering, 32b",
                                          kind="scatter", color="blue")
    quantization_exploration_df.plot(x="quantization_CR", y="quantization_accuracy_loss", label="Quantization, 8b",
                                     ax=ax, kind="scatter", color="orange")
    plt.xlabel("Compression rate")
    plt.ylabel("Accuracy loss")
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])
    plt.title("Contribution of the weight sharing to the compression")
    plt.savefig(os.path.join(log_file, "quantization_exploration.svg"))


def plot_pareto_frontier(pareto_exploration_df, exploration_df, log_file, max_acl):
    ax = exploration_df.plot.scatter(
        x="compression_rate", y="accuracy_loss", label="explored combination",
        style='+', alpha=0.2, color="blue")
    pareto_exploration_df.plot.scatter(
        x="compression_rate", y="accuracy_loss", label="Pareto combination",
        style='x', ax=ax, color="green")

    plt.ylim(exploration_df["accuracy_loss"].min() if exploration_df["accuracy_loss"].min() else 0, max_acl)
    plt.xlabel("Compression rate")
    plt.ylabel("Accuracy loss")
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])
    title = "Combination Exploration Result"
    plt.title(title)
    plt.savefig(os.path.join(log_file, f"{title}.svg"))


def plot_pipeline_result(param, log_file, max_acl):
    fig, ax = plt.subplots()

    # Data & labels
    for accuracy, size, label in param:
        ax.plot(size, accuracy, '+', label=label)
    plt.axhline(param[0][0] - max_acl, linestyle='--', label="affordable_accuracy_loss")
    ax.set_xlabel("Storage Size")
    ax.set_ylabel("top1 Accuracy")
    title = "Compression pipeline Result"
    plt.title(title)
    plt.legend()

    # Grid & ticks
    ax.set_ylim(param[0][0] * 90 / 100, param[0][0] * 101 / 100)
    ax.yaxis.set_major_formatter('{x:.1%}')

    mkfunc = lambda x: '%1.1fM' % (x * 1e-6) if x >= 1e6 else '%1.1fK' % (x * 1e-3) if x >= 1e3 else '%1.1f' % x
    ax.set_xticklabels([mkfunc(x) for x in ax.get_xticks()])

    # plt.tight_layout()
    plt.savefig(os.path.join(log_file, f"{title}.svg"))
    pd.DataFrame(param).to_csv(os.path.join(log_file, f"{title}.csv"))

def plot_layer_exploration(layer_vect, layer_exploration_df, k_list, save_path):
    save_dir = os.path.join(save_path, "layer_exploration")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for layer_id, layer in enumerate(layer_vect):
        K = layer_exploration_df[layer_exploration_df.layer == layer.name].k.values
        accuracy_loss = layer_exploration_df[layer_exploration_df.layer == layer.name].accuracy_loss.values
        inertia = layer_exploration_df[layer_exploration_df.layer == layer.name].accuracy_loss.values
        normalized_inertia = (inertia - min(inertia)) / (max(inertia) - min(inertia))
        selected_K = k_list[layer.name]

        plt.figure()
        plt.plot(K, accuracy_loss, '+', label="Accuracy Loss")
        plt.plot(K, normalized_inertia, 'x', label="normalized inertia")
        for i, xc in enumerate(selected_K):
            plt.axvline(x=xc, color="green", linestyle="--", alpha=0.5, label=f"k{i + 1}")

        title = f"layer exploration {layer.name}"
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{layer_id:02d} {title}.svg"))
        plt.close()


# %% Pareto optimal selection of candidates
def plot_population_expansion_result(final_exploration_df, max_acl, log_file):
    ax = final_exploration_df.plot.scatter(
        x="compression_rate", y="accuracy_loss", label="random_sampling",
        style='+', alpha=0.2, color="blue")

    final_exploration_df[final_exploration_df.origin == "nsgaII_expansion"].plot.scatter(
        x="compression_rate", y="accuracy_loss", label="nsgaII_expansion",
        style='x', ax=ax, color="orange", alpha=0.5)

    plt.ylim(final_exploration_df["accuracy_loss"].min(), 10 * max_acl)
    plt.xlabel("Compression rate")
    plt.ylabel("Accuracy loss")
    ax.yaxis.set_major_formatter('{x:.1%}')

    title = "Combination Exploration Result with Linear Regression and NSGA II for expansion"
    plt.title(title)
    plt.show()
    plt.savefig(os.path.join(log_file, f"{title}.svg"))
