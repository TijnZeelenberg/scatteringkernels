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
        / "ctc/H2/impactparam/Erelmax6000/H2_collisions_b1_5_uniform_Erelmax6000_ncoll1000000_seed42.npy"
    )
    outputpath = paths.model_path("mdn", "mdn_H2_b1_5_bs10000Erelmax6000.pth")
    train_mdn(
        datapath=str(datapath),
        outputpath=str(outputpath),
        epochs=200,
        batch_size=10000,
        lr=1.0e-4,
        patience=200,
        showplots=True,
    )
