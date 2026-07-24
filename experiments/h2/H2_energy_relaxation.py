"""H2 energy-relaxation experiment: MDN vs Borgnakke-Larssen vs SPARTA."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    _attach_clamp_counter,
    _print_clamp_rates,
)
from physics.species import Species
from config.experiment_config import ExperimentConfig

config = ExperimentConfig()

best_model_path = f"results/h2/models/mdn/best_model_bs{config.batch_size}_ngauss{config.num_mixtures}.pth"


def main(
    mdn_model_path: str = best_model_path,
    sparta_path: str = "data/sparta/h2_energy_relaxation.dat",
    lammps_path: str = "data/lammps/h2_energy_relaxation.dat",
    bl_path="data/ml-dsmc/bl/h2_energy_relaxation.dat",
    output_path: str | None = None,
    randomseed: int = 1,
):
    species = Species.H2()
    params = SimulationParams(nr_steps=1500)

    model_tag = Path(mdn_model_path).stem  # e.g. mdn_H2_wf7

    mdn_model = load_mdn(mdn_model_path, randomseed=randomseed)
    models: dict[str, object] = {
        "MDN (ML-DSMC)": mdn_model,
    }

    # Tally how often the MDN's raw eta_tr'/eta_rot' samples land outside [0, 1]
    # and get clamped by batch_collide (mdn.py). model.sample returns the raw,
    # pre-clip values, so wrapping it captures every clamp without touching the
    # engine or the model definition.
    clamp_counts = _attach_clamp_counter(mdn_model)

    results = run_relaxation_comparison(
        species,
        models,
        params=params,
    )

    _print_clamp_rates(clamp_counts)

    mdn_stats = results["MDN (ML-DSMC)"]
    dtype = np.dtype(
        [("timestep", float), ("T_trans_mean", float), ("T_rot_mean", float)]
    )
    arr = np.empty(len(mdn_stats["timestep"]), dtype=dtype)
    arr["timestep"] = mdn_stats["timestep"]
    arr["T_trans_mean"] = mdn_stats["T_trans_mean"]
    arr["T_rot_mean"] = mdn_stats["T_rot_mean"]
    npy_out = paths.ensure_parent(
        "data/ml-dsmc/mdn/h2/best_model_relaxation_clamptest.npy"
    )
    np.save(npy_out, arr)
    print(f"Saved MDN relaxation data to {npy_out}")

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
    # plt.show()


if __name__ == "__main__":
    main()
