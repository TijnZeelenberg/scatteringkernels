"""Train a Beta Mixture Density Network on a CTC collision dataset.

Real implementation lives in `training.core`; this module re-exports the
function for backwards compatibility and provides a runnable `__main__`.
"""

import paths
from training.core import train_beta_mdn

__all__ = ["train_beta_mdn"]


if __name__ == "__main__":
    datapath = paths.DATA_DIR / "H2H2_collisions_numba_b1_0_400000_seed42.npy"
    outputpath = paths.model_path("beta_mdn", "H2H2v1")
    train_beta_mdn(
        datapath=str(datapath),
        outputpath=str(outputpath),
        epochs=100,
        batch_size=128,
        lr=2.0e-4,
        wf=10.0,
        patience=50,
        showplots=True,
    )
