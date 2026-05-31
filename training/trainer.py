"""Train a Gaussian Mixture Density Network on a CTC collision dataset.

Real implementation lives in `training.core`; this module re-exports the
function for backwards compatibility and provides a runnable `__main__`.
"""

from training.core import train_mdn

import paths

__all__ = ["train_mdn"]


if __name__ == "__main__":
    datapath = (
        paths.DATA_DIR
        / "ctc/h2/impactparam/Erelmax10000/H2_collisions_b1_6_uniform_Erelmax10000_ncoll1000000_seed42.npy"
    )
    outputpath = paths.ensure_parent(
        paths.RESULTS_DIR
        / "h2"
        / "models"
        / "mdn"
        / "best_model_mdn_H2_bs2000_bmax1_6.pth"
    )
    train_mdn(
        datapath=str(datapath),
        outputpath=str(outputpath),
        epochs=200,
        batch_size=2000,
        lr=1.0e-4,
        patience=200,
        showplots=True,
    )
