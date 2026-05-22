"""Compare DB-lambda sweep models on the H2 energy-relaxation experiment."""

from __future__ import annotations

import matplotlib.pyplot as plt
from dataclasses import replace

import paths
from experiments.energy_relaxation import (
    SimulationParams,
    load_bl_reference,
    load_lammps_reference,
    load_mdn,
    load_sparta_reference,
    plot_relaxation_comparison,
    print_relaxation_table,
    run_relaxation_comparison,
)
from physics.species import Species

_D_CLASSICAL_MD = 10.1e-10
_ZROT_CLASSICAL_MD = 5.0

LAMBDA_MODELS = {
    "MDN db=0.5": paths.model_path("mdn", "mdn_H2_Etr20k_Erot15k_Teq2200_db05"),
    "MDN db=1.0": paths.model_path("mdn", "mdn_H2_Etr20k_Erot15k_Teq2200_db10"),
    "MDN db=5.0": paths.model_path("mdn", "mdn_H2_Etr20k_Erot15k_Teq2200_db50"),
}


def main(
    sparta_path: str = "sparta/output/sparta_H2_energy_relaxationTtr3000_Trot1000.dat",
    lammps_path: str = "lammps/output/lammps_H2_energy_relaxation.dat",
    bl_path: str = "data/bl_H2_energy_relaxation.dat",
    output_path: str | None = None,
    nr_steps: int = 1000,
    randomseed: int = 1,
):
    species = replace(
        Species.H2(),
        diameter=_D_CLASSICAL_MD,
        zrot_bl=_ZROT_CLASSICAL_MD,
        zrot_mdn=_ZROT_CLASSICAL_MD,
    )
    params = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=3000.0,
        rot_temperature=1000.0,
        randomseed=randomseed,
        grid_cells=(5, 5, 5),
        box_size=1.0e-7,
        dt=1.0e-12,
    )

    models = {
        label: load_mdn(path, randomseed=randomseed)
        for label, path in LAMBDA_MODELS.items()
    }
    results = run_relaxation_comparison(species, models, params=params)
    results["BL (ML-DSMC)"] = load_bl_reference(bl_path)

    sparta = load_sparta_reference(sparta_path)
    lammps = load_lammps_reference(lammps_path)

    print_relaxation_table(
        results, sparta, rot_temperature_initial=params.rot_temperature, lammps=lammps
    )

    out_path = output_path or paths.plot_path("H2_lambdasweep_relaxation.png")
    plot_relaxation_comparison(
        results, sparta, lammps=lammps, ylim=(1000.0, 3000.0), output_path=out_path
    )
    plt.show()


if __name__ == "__main__":
    main()
