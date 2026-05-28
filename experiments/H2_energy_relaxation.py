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
from dataclasses import replace
from physics.species import Species

MDN_CONVERGENT = "results/models/mdn/mdn_H2_b1_55_bs10000Erelmax2000.pth"


def main(
    mdn_model_path: str = MDN_CONVERGENT,
    sparta_path: str = "data/sparta/sparta_H2_energy_relaxationTtr3000_Trot1000.dat",
    lammps_path: str = "data/lammps/lammps_H2_energy_relaxation.dat",
    bl_path="data/ml-dsmc/BL/bl_H2_energy_relaxation.dat",
    output_path: str | None = None,
    nr_steps: int = 150,
    randomseed: int = 1,
    d=10.1e-10,
    zrot_bl=5.0,
    zrot_mdn=5.0 / 2.5,
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
