"""O2 equilibrium-invariance experiment: start both T_trans and T_rot at the
equilibrium temperature (220 K) and check the MDN kernel leaves them there.

This is the equilibrium counterpart to ``O2_energy_relaxation.py``: instead of
relaxing from a non-equilibrium split (T_trans=300, T_rot=100) toward 220 K, the
gas starts *already* equilibrated. A reversible kernel should be invariant here
(both temperatures stay flat at 220 K); any systematic drift exposes a bias in
the learned one-shot map under recursive NTC-weighted application.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paths
from experiments.energy_relaxation import (
    SimulationParams,
    load_mdn,
    plot_relaxation_comparison,
    print_relaxation_table,
    run_relaxation_comparison,
    _attach_clamp_counter,
    _print_clamp_rates,
)
from physics.species import Species
from config.experiment_config import ExperimentConfig

config = ExperimentConfig()

MDN_CONVERGENT = f"results/o2/models/mdn/best_model_bs{config.batch_size}_ngauss{config.num_mixtures}.pth"

# DOF-weighted equilibrium of the relaxation experiment (3 trans + 2 rot DOF):
# (3 * 300 + 2 * 100) / 5 = 220 K.
EQUILIBRIUM_TEMPERATURE = 220.0


def main(
    mdn_model_path: str = MDN_CONVERGENT,
    output_path: str | None = None,
    randomseed: int = 1,
):
    species = Species.O2()
    params = SimulationParams(
        nr_steps=1000,
        trans_temperature=EQUILIBRIUM_TEMPERATURE,
        rot_temperature=EQUILIBRIUM_TEMPERATURE,
    )

    model_tag = Path(mdn_model_path).stem

    mdn_model = load_mdn(mdn_model_path, randomseed=randomseed)
    models: dict[str, object] = {
        "MDN (ML-DSMC)": mdn_model,
    }

    # Tally how often the MDN's raw eta_tr'/eta_rot' samples land outside [0, 1]
    # and get clamped by batch_collide (mdn.py); see O2_energy_relaxation.py.
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
        "data/ml-dsmc/mdn/o2/best_model_equilibrium_invariance.npy"
    )
    # np.save(npy_out, arr)
    # print(f"Saved MDN equilibrium-invariance data to {npy_out}")

    print_relaxation_table(
        results,
        sparta=None,
        rot_temperature_initial=params.rot_temperature,
    )

    out_path: str | paths.Path = output_path or paths.plot_path(
        f"O2_equilibrium_invariance_{model_tag}.png"
    )
    plot_relaxation_comparison(
        results, sparta=None, ylim=(100.0, 300.0), output_path=out_path
    )
    plt.show()


if __name__ == "__main__":
    main()
