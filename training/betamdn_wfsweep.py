"""Beta-MDN weighting-factor sweep — thin wrapper around `run_wf_sweep`."""

from __future__ import annotations

import paths
from training.wfsweep import run_wf_sweep


if __name__ == "__main__":
    datapath = paths.DATA_DIR / "H2H2_collisions_numba_b1_0_400000_seed42.npy"
    run_wf_sweep("beta_mdn", datapath, tag="H2_400000")
