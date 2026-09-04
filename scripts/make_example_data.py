"""Generate the tracked example dataset and model in `examples/`.

The project has three stages:

    1. `ctc_adjusted/` generates collision data with a classical trajectory
       simulation (Numba, parallel)
    2. `training/` fits a mixture density network to that data
    3. `physics/dsmc.py` runs a DSMC simulation with the fitted kernel plugged
       in as its collision model

`data/` and `results/` are untracked, so a fresh clone can run none of it. This
script produces the two small artifacts in `examples/` that are committed to fix
that: a 5000-collision CTC dataset and an MDN trained on it. Together they let
someone clone the repository and run a DSMC experiment immediately.

Both are deliberately tiny. 5000 collisions is three orders of magnitude below
the datasets used for the thesis results, and the DSMC experiments run at 300 K
while the dataset is sampled up to E_rel/k_B = 10^4 K, so the example model is
extrapolating throughout. It demonstrates that the pipeline runs; it is not a
result.

    uv run python -m scripts.make_example_data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Pin to the CPU before anything imports torch, so this runs the same way on a
# clone with no GPU or a mismatched torch wheel. Export the variable to override.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

import paths
from config.experiment_config import ExperimentConfig
from training.core import train_collision_model

EXAMPLES_DIR = paths.PROJECT_ROOT / "examples"
EXAMPLE_DATASET = EXAMPLES_DIR / "ctc_H2_ncoll5000.npy"
EXAMPLE_MODEL = EXAMPLES_DIR / "mdn_H2_ncoll5000.pth"

N_COLLISIONS = 5000

# stage 1: CTC data generation


def generate_ctc_dataset(n_collisions: int = N_COLLISIONS) -> np.ndarray:
    """Run `ctc_adjusted.ctc_h2` and return its (N, 6) column layout.

    Columns are (Etr, Erot1, Erot2, Etr', Erot1', Erot2'), energies as E / k_B
    in Kelvin — what `training.data_prep` expects.

    Sampling settings (`dist`, `T_eq`, `E_rel_max`, `bfac`, `seed`) are read off
    the module rather than chosen here, so this is the shipped generator at a
    smaller collision count and nothing else.
    """
    from ctc_adjusted import ctc_h2

    print(
        f"Generating {n_collisions} CTC collisions "
        f"(dist={ctc_h2.dist}, E_rel_max={ctc_h2.E_rel_max:.0f} K, bfac={ctc_h2.bfac})"
    )
    print("Compiling the numba kernel (first run only) ...")
    ctc_h2._run_chunk(
        0, 1, ctc_h2._DIST_IDS[ctc_h2.dist], ctc_h2.T_eq, ctc_h2.E_rel_max
    )

    raw = ctc_h2.run_all_collisions(
        n_collisions,
        chunk_size=1000,
        seed=ctc_h2.seed,
        dist=ctc_h2.dist,
        T_eq=ctc_h2.T_eq,
        E_rel_max=ctc_h2.E_rel_max,
    )
    # Same column selection as ctc_h2's own __main__ block.
    return np.column_stack(
        [raw[:, 10], raw[:, 6], raw[:, 7], raw[:, 11], raw[:, 8], raw[:, 9]]
    )


def describe_dataset(data: np.ndarray) -> None:
    """Print the dataset properties that matter downstream."""
    pre = data[:, 0:3].sum(axis=1)
    post = data[:, 3:6].sum(axis=1)
    ratio = data[:, 0] / data[:, 3]
    elastic = (ratio > 0.99) & (ratio < 1.01)
    print(f"  shape: {data.shape}  (Etr, Er1, Er2, Etr', Er1', Er2') [K]")
    print(f"  Etot:  {pre.min():.0f} - {pre.max():.0f} K  (mean {pre.mean():.0f} K)")
    print(f"  elastic fraction: {elastic.mean():.3f}")
    # CTC trajectories conserve energy only to integrator accuracy. data_prep
    # normalizes pre- and post-states separately, so this does not propagate.
    print(f"  max per-collision energy drift: {np.abs((post - pre) / pre).max():.2e}")


# stage 2: MDN training


def train_example_model(datapath: Path, outputpath: Path):
    """Fit a Gaussian MDN to the example dataset using the project defaults."""
    config = ExperimentConfig()
    model, train_hist, val_hist = train_collision_model(
        "mdn",
        datapath=datapath,
        outputpath=outputpath,
        epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        patience=config.patience,
    )
    print(f"  train NLL: {train_hist[0]:.4f} -> {train_hist[-1]:.4f}")
    print(f"    val NLL: {val_hist[0]:.4f} -> {val_hist[-1]:.4f}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-collisions",
        type=int,
        default=N_COLLISIONS,
        help=f"number of CTC collisions to generate (default {N_COLLISIONS})",
    )
    args = parser.parse_args()

    paths.ensure_dir(EXAMPLES_DIR)

    print("=== 1/2  CTC collision data ===")
    data = generate_ctc_dataset(args.n_collisions)
    np.save(EXAMPLE_DATASET, data)
    print(f"Wrote {EXAMPLE_DATASET}")
    describe_dataset(data)

    print("\n=== 2/2  MDN training ===")
    train_example_model(EXAMPLE_DATASET, EXAMPLE_MODEL)
    print(f"Wrote {EXAMPLE_MODEL}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
