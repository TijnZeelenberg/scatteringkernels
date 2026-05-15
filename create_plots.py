"""Compare CTC, MDN, and GMM scattering distributions side-by-side.

Loads a CTC collision dataset, samples from a trained MDN and a freshly-fit
GMM at the same input points, then renders density-scatter + histogram
comparison plots and reports KL divergences.
"""

from __future__ import annotations

import paths
from analysis.kl_divergence import kl_divergence
from config.experiment_config import ExperimentConfig
from config.plotting_config import PlottingConfig
from machinelearning.gmm import GaussianMixtureModel
from machinelearning.mdn import MixtureDensityNetwork
from utils.helpers import load_dataset
from visualization.plot import plot_density_scatter, plot_histogram

import matplotlib.pyplot as plt


def main(
    dataset_path: str = "data/O2O2_collisions.csv",
    mdn_model_path: str = "results/models/mdn_O2O2.pth",
    config: ExperimentConfig | None = None,
):
    config = config or ExperimentConfig()
    inputs, outputs = load_dataset(dataset_path)

    mdn = MixtureDensityNetwork(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        num_mixtures=config.num_mixtures,
        hidden_dim=config.hidden_dim,
        randomseed=config.random_seed,
    )
    mdn.load_model(mdn_model_path)
    mdn_samples = mdn.sample(x=inputs)

    gmm = GaussianMixtureModel(
        n_components=config.gmm_n_components, covariance_type=config.gmm_covariance_type
    )
    gmm.fit(outputs)
    gmm_samples = gmm.sample(num_samples=config.num_samples)

    datasets = {
        "inputs": inputs[:, 1:],
        "CTC": outputs,
        "MDN": mdn_samples,
        "GMM": gmm_samples,
    }

    plotting_config = PlottingConfig()
    fig_scatter, axes = plt.subplots(
        2, 3, figsize=(plotting_config.figsize[0] * 3, plotting_config.figsize[1] * 2)
    )
    plot_density_scatter(axes, datasets)
    fig_scatter.tight_layout()
    fig_scatter.savefig(paths.plot_path("ctc_vs_mdn_vs_gmm_scatter.png"), dpi=150)

    plot_histogram(datasets)
    plt.savefig(paths.plot_path("ctc_vs_mdn_vs_gmm_histogram.png"), dpi=150)

    kl_ctc_mdn = kl_divergence(datasets["CTC"][:, 0], datasets["MDN"][:, 0])
    kl_ctc_gmm = kl_divergence(datasets["CTC"][:, 0], datasets["GMM"][:, 0])
    print(f"KL Divergence between CTC and MDN: {kl_ctc_mdn:.4f}")
    print(f"KL Divergence between CTC and GMM: {kl_ctc_gmm:.4f}")


if __name__ == "__main__":
    main()
