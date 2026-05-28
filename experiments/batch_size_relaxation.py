"""H2 energy-relaxation sweep over MDN batch-size variants."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paths
from config.plotting_config import PlottingConfig
from experiments.energy_relaxation import (
    SimulationParams,
    load_bl_reference,
    load_mdn,
    run_relaxation,
)
from physics.species import Species

BATCH_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192, 10000, 16384]
MODEL_DIR = "results/models/mdn/batch_size"
BL_PATH = "data/ml-dsmc/BL/bl_H2_energy_relaxation.dat"


def main(
    output_path: str | None = None,
    nr_steps: int = 150,
    randomseed: int = 1,
    d: float = 10.1e-10,
    zrot_bl: float = 5.0,
    zrot_mdn: float = 5.0 / 2.5,
    ylim: tuple[float, float] = (100.0, 300.0),
):
    species = replace(
        Species.H2(),
        diameter=d,
        zrot_bl=zrot_bl,
        zrot_mdn=zrot_mdn,
    )
    params = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=300.0,
        rot_temperature=100.0,
        randomseed=randomseed,
        grid_cells=(5, 5, 5),
        box_size=1.0e-7,
        dt=1.0e-11,
    )

    bl = load_bl_reference(BL_PATH)

    pc = PlottingConfig()
    fig, axes = plt.subplots(3, 3, figsize=(2 * pc.figsize[0], 2 * pc.figsize[1]))

    for ax, bs in zip(axes.flat, BATCH_SIZES):
        model_path = f"{MODEL_DIR}/mdn_H2_b{bs}.pth"
        print(f"\n--- batch size {bs} ---")
        model = load_mdn(model_path, randomseed=randomseed)
        stats = run_relaxation(species, model, params=params)

        t = stats["timestep"]
        ax.plot(t, stats["T_trans_mean"], color="tab:blue", label=r"$T_{trans}$ MDN")
        ax.plot(t, stats["T_rot_mean"], color="tab:orange", label=r"$T_{rot}$ MDN")
        ax.plot(
            bl["timestep"],
            bl["T_trans_mean"],
            color="tab:blue",
            linestyle="--",
            label=r"$T_{trans}$ BL",
        )
        ax.plot(
            bl["timestep"],
            bl["T_rot_mean"],
            color="tab:orange",
            linestyle="--",
            label=r"$T_{rot}$ BL",
        )

        ax.set_title(f"batch size = {bs}", fontsize=pc.label_fontsize)
        ax.set_xlabel("Time [s]", fontsize=pc.label_fontsize - 2)
        ax.set_ylabel("Temperature [K]", fontsize=pc.label_fontsize - 2)
        ax.ticklabel_format(style="sci", scilimits=(-2, 3))
        ax.set_ylim(*ylim)
        ax.grid()
        ax.legend(fontsize=pc.legend_fontsize - 1)

    fig.tight_layout()

    out = output_path or paths.plot_path("H2_energy_relaxation_batch_size_sweep.png")
    out = paths.ensure_parent(out)
    fig.savefig(out, dpi=300)
    print(f"\nSaved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
