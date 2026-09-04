"""Train an MDN on the example dataset and run it inside the DSMC.

One command that exercises stages 2 and 3 of the pipeline on the committed
`examples/` artifacts, so a fresh clone can verify the code runs without
generating any data first:

    uv run python -m scripts.run_example              # train, then simulate
    uv run python -m scripts.run_example --dsmc-only  # skip training

The DSMC is run twice — once with the Borgnakke-Larssen baseline, once with the
MDN — so the two collision models can be compared directly. Both should relax
T_trans down and T_rot up towards the same equipartition temperature while
conserving total energy.
"""

from __future__ import annotations

import argparse
import os

# Pin to the CPU before anything imports torch, so this runs the same way on a
# clone with no GPU or a mismatched torch wheel. Export the variable to override.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import paths
from config.experiment_config import ExperimentConfig
from experiments.energy_relaxation import (
    SimulationParams,
    load_mdn,
    run_relaxation,
)
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.species import Species
from scripts.make_example_data import EXAMPLE_DATASET, EXAMPLE_MODEL
from training.core import train_collision_model

# a short run: enough steps to show the relaxation, few enough to finish in seconds
EXAMPLE_PARAMS = SimulationParams(
    N_sim=2000,
    N_real=20000,
    nr_steps=50,
    grid_cells=(3, 3, 3),
)


def train() -> str:
    """Fit a Gaussian MDN to the example dataset; return the checkpoint path."""
    config = ExperimentConfig()
    outputpath = paths.model_path("mdn", "example/mdn_example")
    _, train_hist, val_hist = train_collision_model(
        "mdn",
        datapath=EXAMPLE_DATASET,
        outputpath=outputpath,
        epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        patience=config.patience,
    )
    print(f"  train NLL: {train_hist[0]:.3f} -> {train_hist[-1]:.3f}")
    print(f"    val NLL: {val_hist[0]:.3f} -> {val_hist[-1]:.3f}")
    return str(outputpath)


def simulate(label: str, collision_model) -> None:
    """Run one DSMC relaxation and print its start/end temperatures."""
    stats = run_relaxation(Species.H2(), collision_model, params=EXAMPLE_PARAMS)
    e0, e1 = stats["total_energy"][0], stats["total_energy"][-1]
    print(
        f"  {label:<20} "
        f"T_trans {stats['T_trans_mean'][0]:6.1f} -> {stats['T_trans_mean'][-1]:6.1f} K   "
        f"T_rot {stats['T_rot_mean'][0]:6.1f} -> {stats['T_rot_mean'][-1]:6.1f} K   "
        f"energy drift {abs(e1 - e0) / e0:.1e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsmc-only",
        action="store_true",
        help="skip training and use the committed example model",
    )
    args = parser.parse_args()

    if args.dsmc_only:
        model_path = str(EXAMPLE_MODEL)
    else:
        print("=== Training an MDN on examples/ ===")
        model_path = train()

    print("\n=== DSMC energy relaxation ===")
    print(
        f"  {EXAMPLE_PARAMS.N_sim} particles, {EXAMPLE_PARAMS.nr_steps} steps, "
        f"T_trans {EXAMPLE_PARAMS.trans_temperature:.0f} K / "
        f"T_rot {EXAMPLE_PARAMS.rot_temperature:.0f} K at t=0"
    )
    simulate("Borgnakke-Larssen", borgnakke_larssen_model(randomseed=1))
    simulate("MDN", load_mdn(model_path))

    # Both models redistribute energy between 3 translational and 2 rotational
    # degrees of freedom, so both relax towards the same equipartition value.
    t_eq = (
        3 * EXAMPLE_PARAMS.trans_temperature + 2 * EXAMPLE_PARAMS.rot_temperature
    ) / 5
    print(f"\n  equipartition temperature for 3+2 DOF: {t_eq:.1f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
