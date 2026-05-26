"""Train one Gaussian MDN per E_rel input distribution.

Datasets are expected in data/distributions/ and are matched by distribution
name (uniform / mb / ntc). Generate them with:

    # in ctc_adjusted/ctc_h2_multiple_collisions_numba.py set dist = 'uniform' | 'mb' | 'ntc'
    python ctc_adjusted/ctc_h2_multiple_collisions_numba.py

Models are saved to results/models/mdn/Etrans_distribution/{dist}.pth.
No sample reweighting is applied — all collisions are weighted equally.
"""

from __future__ import annotations

from pathlib import Path

import paths
from training.core import train_mdn

DIST_DIR = paths.DATA_DIR / "distributions"
OUTPUT_DIR = paths.MDN_DIR / "Etrans_distribution"

DISTRIBUTIONS = ("_uniform_", "_mb_", "_ntc_")

# Defaults from training/trainer.py
EPOCHS = 100
BATCH_SIZE = 2048
LR = 2.0e-4
PATIENCE = 100


def _find_dataset(dist: str) -> Path:
    matches = list(DIST_DIR.glob(f"*_{dist}_*")) + list(DIST_DIR.glob(f"*{dist}*"))
    matches = list({p for p in matches if p.suffix == ".npy"})
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No dataset found for dist='{dist}' in {DIST_DIR}. "
            f"Run ctc_h2_multiple_collisions_numba.py with dist='{dist}'."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple datasets matched dist='{dist}' in {DIST_DIR}: {matches}. "
            f"Remove the ambiguous files and keep one per distribution."
        )
    return matches[0]


def run_dist_sweep() -> dict[str, Path]:
    saved: dict[str, Path] = {}
    for dist in DISTRIBUTIONS:
        datapath = _find_dataset(dist)
        outputpath = paths.ensure_parent(OUTPUT_DIR / f"{dist.replace('_', '')}.pth")
        print(f"\n{'=' * 60}")
        print(f"Distribution: {dist}  |  dataset: {datapath.name}")
        print(f"{'=' * 60}")
        train_mdn(
            datapath=str(datapath),
            outputpath=str(outputpath),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            patience=PATIENCE,
            wf=None,
        )
        saved[dist] = outputpath
    return saved


if __name__ == "__main__":
    saved = run_dist_sweep()
    print("\nAll models saved:")
    for dist, path in saved.items():
        print(f"  {dist:8s} -> {path}")
