"""H2 energy-relaxation experiment: MDN vs Borgnakke-Larssen vs SPARTA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import paths
from experiments.energy_relaxation import (
    SimulationParams,
    load_mdn,
    load_lammps_reference,
    load_sparta_reference,
    load_bl_reference,
    plot_relaxation_comparison,
    print_relaxation_table,
    run_relaxation_comparison,
)
from physics.species import Species

MDN_CONVERGENT = "results/h2/models/mdn/best_model_mdn_H2_bs2000_bmax1_6.pth"


def main(
    mdn_model_path: str = MDN_CONVERGENT,
    sparta_path: str = "data/sparta/h2_energy_relaxation.dat",
    lammps_path: str = "data/lammps/h2_energy_relaxation.dat",
    bl_path="data/ml-dsmc/bl/h2_energy_relaxation.dat",
    output_path: str | None = None,
    randomseed: int = 1,
):
    species = Species.H2()
    params = SimulationParams(nr_steps=400)

    model_tag = Path(mdn_model_path).stem  # e.g. mdn_H2_wf7

    models: dict[str, object] = {
        "MDN (ML-DSMC)": load_mdn(mdn_model_path, randomseed=randomseed),
    }

    results = run_relaxation_comparison(
        species,
        models,
        params=params,
    )
    sparta = load_sparta_reference(sparta_path)
    lammps = load_lammps_reference(lammps_path)
    bl = load_bl_reference(bl_path)

    print_relaxation_table(
        results,
        sparta,
        rot_temperature_initial=params.rot_temperature,
        lammps=lammps,
        bl=bl,
    )

    out_path: str | paths.Path = output_path or paths.plot_path(
        f"H2_energy_relaxation_{model_tag}.png"
    )
    plot_relaxation_comparison(
        results, sparta, lammps=lammps, bl=bl, ylim=(100.0, 300.0), output_path=out_path
    )
    plt.show()


if __name__ == "__main__":
    main()
