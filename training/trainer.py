"""Train a Gaussian Mixture Density Network on a CTC collision dataset.

Real implementation lives in `training.core`; this module re-exports the
function for backwards compatibility and provides a runnable `__main__`.
"""

from training.core import train_mdn

import paths

__all__ = ["train_mdn"]


if __name__ == "__main__":
    datapath = paths.DATA_DIR / "H2H2_collisions_numba_b1_0_Etr20k_Erot15k_400000_seed42.npy"
    outputpath = paths.model_path("mdn", "mdn_H2_Etr20k_Erot15k_Teq2200")
    train_mdn(
        datapath=str(datapath),
        outputpath=str(outputpath),
        epochs=100,
        batch_size=128,
        lr=2.0e-4,
        T_eq=2200.0,
        patience=100,
        showplots=True,
    )
