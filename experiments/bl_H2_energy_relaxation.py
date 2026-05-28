"""Generate BL-DSMC energy-relaxation trace for H2 and save to disk."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import paths
from experiments.energy_relaxation import SimulationParams, run_relaxation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.species import Species

OUTPUT_PATH = "data/ml-dsmc/BL/bl_H2_energy_relaxation.dat"


def main(
    output_path: str = OUTPUT_PATH,
    nr_steps: int = 150,
    randomseed: int = 1,
    d: float = 10.1e-10,
    zrot_bl: float = 5.0,
):
    species = replace(
        Species.H2(),
        diameter=d,
        zrot_bl=zrot_bl,
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

    model = borgnakke_larssen_model(randomseed=randomseed)
    stats = run_relaxation(species, model, params=params)

    steps = np.arange(nr_steps)
    data = np.column_stack([
        steps,
        stats["timestep"],
        stats["T_trans_mean"],
        stats["T_rot_mean"],
    ])

    out = paths.ensure_parent(output_path)
    np.savetxt(
        out,
        data,
        header="step time T_trans T_rot",
        comments="",
        fmt=["%.0f", "%.6e", "%.6f", "%.6f"],
    )
    print(f"Saved BL energy relaxation data to {out}")


if __name__ == "__main__":
    main()
