"""Generate BL-DSMC energy-relaxation trace for H2 and save to disk."""

from __future__ import annotations
import numpy as np

import paths
from experiments.energy_relaxation import SimulationParams, run_relaxation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.species import Species

OUTPUT_PATH = "data/ml-dsmc/bl/h2_energy_relaxation.dat"


def main(
    nr_steps: int = 400,
    output_path: str = OUTPUT_PATH,
):
    species = Species.H2()

    params = SimulationParams(nr_steps=nr_steps)

    model = borgnakke_larssen_model()
    stats = run_relaxation(species, model, params=params)

    steps = np.arange(nr_steps)
    data = np.column_stack(
        [
            steps,
            stats["timestep"],
            stats["T_trans_mean"],
            stats["T_rot_mean"],
        ]
    )

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
