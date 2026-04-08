import matplotlib.pyplot as plt
from config.plotting_config import PlottingConfig
from scipy.stats import gaussian_kde

datasetnames = ["CTC", "MDN", "GMM"]
config = PlottingConfig()


def plot_energy_relaxation(stats):
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.plot(
        stats["timestep"],
        stats["T_trans_mean"],
        label="Translational Energy",
    )
    ax.plot(
        stats["timestep"],
        stats["T_rot_mean"],
        label="Rotational Energy",
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=config.label_fontsize,
        fontweight=config.label_fontweight,
    )
    ax.ticklabel_format(style="sci", scilimits=(0, 0))
    ax.set_ylabel(
        "Energy [K]",
        fontsize=config.label_fontsize,
        fontweight=config.label_fontweight,
    )
    ax.set_title(
        "Energy Relaxation Over Time",
        fontsize=config.title_fontsize,
        fontweight=config.title_fontweight,
    )

    ax.legend(fontsize=config.legend_fontsize)

    return fig, ax


def plot_loss_history(train_loss_history, val_loss_history, dataset_name):
    plt.figure(figsize=config.figsize)
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel(
        "Epoch",
        fontsize=config.label_fontsize,
        fontweight=config.label_fontweight,
    )
    plt.ylabel(
        "Loss",
        fontsize=config.label_fontsize,
        fontweight=config.label_fontweight,
    )
    plt.legend(fontsize=config.legend_fontsize)
    plt.tight_layout()
    plt.show()


def compute_density_per_point(x, y):
    data = [x, y]
    kde = gaussian_kde(data)
    return kde(data)


def plot_density_scatter(datasets: dict):
    """ "Plots a scatter plot of the CTC, MDN and GMM datapoints colored by their density."""
    # TODO: use seaborn kdeplot for quicker density estimation and plotting

    datasetnames = list(datasets.keys())
    datasetnames.remove("inputs")

    figsize_x = config.figsize[0] * len(datasetnames)
    figsize_y = config.figsize[1] * 2
    fig, ax = plt.subplots(2, len(datasetnames), figsize=(figsize_x, figsize_y))

    for i in range(2):
        x = datasets["inputs"][:, i]
        for j, dataset_name in enumerate(datasetnames):
            y = datasets[dataset_name][:, i]
            density = compute_density_per_point(x, y)
            ax[i, j].scatter(
                x,
                y,
                c=density,
                cmap="viridis",
                s=config.scatter_point_size,
                alpha=config.scatter_alpha,
            )
            ax[i, j].set_xlim(0, 1)
            ax[i, j].set_ylim(0, 1)
            ax[0, j].set_title(
                f"{dataset_name}",
                fontsize=config.title_fontsize,
                fontweight=config.title_fontweight,
            )
            ax[i, j].set_xlabel(
                "eta_tr" if i == 0 else "eta_rot_A",
                fontsize=config.label_fontsize,
                fontweight=config.label_fontweight,
            )
            ax[i, 0].set_ylabel(
                "eta_trp" if i == 0 else "eta_rot_Ap",
                fontsize=config.label_fontsize,
                fontweight=config.label_fontweight,
            )
    plt.tight_layout()
    plt.show()


def plot_histogram(datasets: dict):
    """Plots histograms of the CTC, MDN and GMM datasets for both output variables."""
    fig, ax = plt.subplots(
        1, 2, figsize=(config.figsize[0] * 2, config.figsize[1])
    )
    for i, dataset_name in enumerate(datasetnames):
        ax[0].hist(
            datasets[dataset_name][:, 0],
            bins=config.bin_count,
            density=config.hist_density,
            alpha=config.hist_alpha,
            label=f"{dataset_name}",
        )
        ax[0].vlines(
            x=[0.0, 1.0],
            ymin=0,
            ymax=ax[0].get_ylim()[1],
            color="black",
            linestyle="--",
            alpha=0.5,
        )
        ax[1].hist(
            datasets[dataset_name][:, 1],
            bins=config.bin_count,
            density=config.hist_density,
            alpha=config.hist_alpha,
            label=f"{dataset_name}",
        )
        ax[1].vlines(
            x=[0.0, 1.0],
            ymin=0,
            ymax=ax[1].get_ylim()[1],
            color="black",
            linestyle="--",
            alpha=0.5,
        )
        ax[0].set_xlabel(
            "Value",
            fontsize=config.label_fontsize,
            fontweight=config.label_fontweight,
        )
        ax[0].set_ylabel(
            "Density",
            fontsize=config.label_fontsize,
            fontweight=config.label_fontweight,
        )
        ax[1].set_xlabel(
            "Value",
            fontsize=config.label_fontsize,
            fontweight=config.label_fontweight,
        )
        ax[1].set_ylabel(
            "Density",
            fontsize=config.label_fontsize,
            fontweight=config.label_fontweight,
        )
        ax[0].set_title(
            r"$\epsilon_{tr}$",
            fontsize=config.title_fontsize,
            fontweight=config.title_fontweight,
        )
        ax[1].set_title(
            r"$\epsilon_{rot,A}$",
            fontsize=config.title_fontsize,
            fontweight=config.title_fontweight,
        )
        ax[0].legend(fontsize=config.legend_fontsize)
        ax[1].legend(fontsize=config.legend_fontsize)
    plt.tight_layout()
    plt.show()