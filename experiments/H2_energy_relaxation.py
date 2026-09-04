"""H2 energy-relaxation experiment: MDN vs Borgnakke-Larssen vs SPARTA."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

import paths
from experiments.energy_relaxation import (
    SimulationParams,
    load_lammps_reference,
    load_mdn,
    load_sparta_reference,
    plot_relaxation_comparison,
    print_relaxation_table,
    run_relaxation_comparison,
)
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.collision_logger import CollisionLogger
from physics.species import Species

MDN_CONVERGENT = "results/models/mdn/mdn_H2_uniform_hiddim8_mix20.pth"


def main(
    mdn_model_path: str = MDN_CONVERGENT,
    sparta_path: str = "data/sparta/sparta_H2_energy_relaxationTtr3000_Trot1000.dat",
    lammps_path: str = "data/lammps/lammps_H2_energy_relaxation.dat",
    output_path: str | None = None,
    nr_steps: int = 100,
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
        trans_temperature=3000.0,
        rot_temperature=1000.0,
        randomseed=randomseed,
        grid_cells=(5, 5, 5),
        box_size=1.0e-7,
        dt=1.0e-11,
    )

    model_tag = Path(mdn_model_path).stem  # e.g. mdn_H2_wf7

    models: dict[str, object] = {
        "BL (ML-DSMC)": borgnakke_larssen_model(randomseed=randomseed),
        "MDN (ML-DSMC)": load_mdn(mdn_model_path, randomseed=randomseed),
    }

    log_suffix_by_label = {
        "BL (ML-DSMC)": "BL",
        "MDN (ML-DSMC)": model_tag,
    }

    def make_logger(label: str) -> CollisionLogger:
        return CollisionLogger(
            output_path=paths.log_path(
                f"H2_energy_relaxation_{log_suffix_by_label[label]}.npz"
            ),
            snapshot_every=100,
            training_caps_K={"E_trans_max_K": 20100.0, "E_rot_max_K": 15000.0},
        )

    results = run_relaxation_comparison(
        species, models, params=params, collision_logger_factory=make_logger
    )
    sparta = load_sparta_reference(sparta_path)
    lammps = load_lammps_reference(lammps_path)

    print_relaxation_table(
        results, sparta, rot_temperature_initial=params.rot_temperature, lammps=lammps
    )

    out_path: str | paths.Path = output_path or paths.plot_path(
        f"H2_energy_relaxation_{model_tag}.png"
    )
    plot_relaxation_comparison(
        results, sparta, lammps=lammps, ylim=(1000.0, 3000.0), output_path=out_path
    )
    plt.show()


if __name__ == "__main__":
    main()
