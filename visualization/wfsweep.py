"""Run DSMC energy-relaxation experiments across a weighting-factor sweep.

`run_wf_sweep_experiments` is the reusable function: given a model kind and
the directory tag of a previously-trained sweep, it loads each model, runs a
DSMC relaxation, and plots a 3x3 grid comparing to SPARTA. Both this script
and `visualization/betamdn_wfsweep.py` call it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import paths
from experiments.energy_relaxation import (
    SimulationParams,
    load_beta_mdn,
    load_mdn,
    load_sparta_reference,
    run_relaxation,
)
from physics.species import Species


DEFAULT_WEIGHTS: tuple[float, ...] = (0.25, 0.5, 1, 2, 3, 4, 5, 6, 7)


def run_wf_sweep_experiments(
    kind: str,
    tag: str,
    *,
    species: Species,
    sparta_path: str,
    trainseed: int | None = None,
    weights: Iterable[float] = DEFAULT_WEIGHTS,
    params: SimulationParams | None = None,
    output_path: str | Path | None = None,
):
    """Load each model in a wf sweep, run DSMC, plot a comparison grid.

    Args:
        kind: "mdn" or "beta_mdn".
        tag: the wf-sweep directory tag, e.g. "H2_400000_dataseed42".
        species: gas species parameters.
        sparta_path: SPARTA reference data file.
        trainseed: when set, load models from the `trainseed<N>/` subdirectory.
        weights: weighting factors that have models on disk.
        params: simulation parameters (default suited for H2 relaxation).
        output_path: figure save path (default: alongside the trained models).
    """
    weights = list(weights)
    sim_params: SimulationParams = (
        params
        if params is not None
        else SimulationParams(
            nr_steps=150,
            trans_temperature=3000.0,
            rot_temperature=1000.0,
            randomseed=42,
            grid_cells=(5, 5, 5),
        )
    )
    sparta = load_sparta_reference(sparta_path)

    loader = load_beta_mdn if kind.startswith("beta") else load_mdn
    model_label = "Beta MDN" if kind.startswith("beta") else "MDN"

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes_flat = np.asarray(axes).flatten()

    for i, wf in tqdm(
        enumerate(weights),
        desc=f"Running {model_label} wf sweep simulations",
        unit="simulation",
        total=len(weights),
    ):
        ax = axes_flat[i]
        model_path = paths.wf_sweep_model_path(kind, tag, wf, trainseed=trainseed)
        model = loader(model_path, randomseed=sim_params.randomseed)
        stats = run_relaxation(species, model, params=sim_params)

        ax.plot(
            stats["timestep"],
            stats["T_trans_mean"],
            label=rf"$T_{{trans}}$ {model_label}",
        )
        ax.plot(
            stats["timestep"], stats["T_rot_mean"], label=rf"$T_{{rot}}$ {model_label}"
        )
        ax.plot(
            sparta["t"],
            sparta["T_trans"],
            linestyle="--",
            color="red",
            label=r"$T_{trans}$ SPARTA",
        )
        ax.plot(
            sparta["t"],
            sparta["T_rot"],
            linestyle="--",
            color="blue",
            label=r"$T_{rot}$ SPARTA",
        )

        ax.set_title(f"wf = {wf}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [s]", fontsize=9)
        ax.set_ylabel("Temperature [K]", fontsize=9)
        ax.ticklabel_format(style="sci", scilimits=(-2, 3))
        ax.set_ylim(20, 450)
        ax.grid(True)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"{species.name} Energy Relaxation — {model_label} Weighting Factor Sweep — Randomseed {sim_params.randomseed}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path is None:
        output_path = (
            paths.wf_sweep_dir(kind, tag, trainseed)
            / f"{species.name}_{kind}_wfsweep.png"
        )
    fig.savefig(paths.ensure_parent(output_path), dpi=300)
    return fig


if __name__ == "__main__":
    run_wf_sweep_experiments(
        kind="mdn",
        tag="H2_400000_dataseed42",
        trainseed=42,
        species=Species.H2(),
        sparta_path="data/sparta_H2_energy_relaxationVHS_zinv0151.dat",
    )
    plt.show()
