"""H2 energy-relaxation experiment: MDN vs Borgnakke-Larssen vs SPARTA."""

from __future__ import annotations

import matplotlib.pyplot as plt

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
from dataclasses import replace

from physics.species import Species

# Calibrated to classical LJ/rigid-rotor MD (matching LAMMPS tau_rot = 2.35 ps):
#   d_eff = 10.1 Å  (≈ LJ cutoff radius, sets the correct VHS collision rate)
#   zrot  = 2.0     (classical rigid-rotor: ~2 collisions to thermalize rotation)
_D_CLASSICAL_MD = 10.1e-10
_ZROT_CLASSICAL_MD = 5.0


def main(
    mdn_model_path: str = "results/models/mdn/mdn_H2_Etr20k_Erot15k_Teq2200_db01.pth",
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

    models: dict[str, object] = {
        "MDN (ML-DSMC)": load_mdn(mdn_model_path, randomseed=randomseed),
    }
    results = run_relaxation_comparison(species, models, params=params)
    results["BL (ML-DSMC)"] = load_bl_reference(bl_path)
    sparta = load_sparta_reference(sparta_path)
    lammps = load_lammps_reference(lammps_path)

    print_relaxation_table(
        results, sparta, rot_temperature_initial=params.rot_temperature, lammps=lammps
    )

    out_path: str | paths.Path = output_path or paths.plot_path(
        "H2_energy_relaxation.png"
    )
    plot_relaxation_comparison(
        results, sparta, lammps=lammps, ylim=(1000.0, 3000.0), output_path=out_path
    )
    plt.show()


if __name__ == "__main__":
    main()
