"""Plot energy-relaxation results for all six impact-parameter MDN models."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig
from experiments.energy_relaxation import (
    SimulationParams,
    load_mdn,
    run_relaxation,
)
from physics.species import Species

BFAC_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
YLIM = (100.0, 300.0)

params = SimulationParams(
    nr_steps=150,
    trans_temperature=300.0,
    rot_temperature=100.0,
    randomseed=1,
    grid_cells=(5, 5, 5),
    box_size=1.0e-7,
    dt=1.0e-11,
)
species = replace(Species.H2(), diameter=10.1e-10, zrot_bl=5.0, zrot_mdn=5.0 / 2.5)

pc = PlottingConfig()
fig, axes = plt.subplots(2, 3, figsize=(3 * pc.figsize[0], 2 * pc.figsize[1]))

for ax, bfac in zip(axes.flat, BFAC_VALUES):
    bfac_tag = str(bfac).replace(".", "_")
    model_path = f"results/models/mdn/impactparam/mdn_H2_b{bfac_tag}.pth"
    model = load_mdn(model_path, randomseed=params.randomseed)
    stats = run_relaxation(species, model, params=params)

    ax.plot(stats["timestep"], stats["T_trans_mean"], label=r"$T_{trans}$ MDN")
    ax.plot(stats["timestep"], stats["T_rot_mean"], label=r"$T_{rot}$ MDN")

    ax.set_title(f"$b_{{fac}} = {bfac}$", fontsize=pc.label_fontsize)
    ax.set_xlabel("Time [s]", fontsize=pc.label_fontsize)
    ax.set_ylabel("Temperature [K]", fontsize=pc.label_fontsize)
    ax.ticklabel_format(style="sci", scilimits=(-2, 3))
    ax.set_ylim(*YLIM)
    ax.grid()
    ax.legend(fontsize=pc.legend_fontsize)

fig.tight_layout()
out = paths.plot_path("mdn_impactparam_relaxation2.png")
fig.savefig(out, dpi=300)
print(f"Saved to {out}")
plt.show()
