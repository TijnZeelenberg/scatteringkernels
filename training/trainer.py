"""Train a Gaussian Mixture Density Network on a CTC collision dataset.

Real implementation lives in `training.core`; this module re-exports the
function for backwards compatibility and provides a runnable `__main__`.
"""

import paths
from training.core import train_mdn

__all__ = ["train_mdn"]


if __name__ == "__main__":
    datapath = (
        paths.DATA_DIR
        / "ctc/H2/impactparam/H2_collisions_b1_0_uniform_Erelmax10000_ncoll800000_seed42.npy"
    )
    outputpath = paths.model_path("mdn", "mdn_H2_uniform_hiddim8_mix20")
    train_mdn(
        datapath=str(datapath),
        outputpath=str(outputpath),
        epochs=100,
        batch_size=10000,
        lr=1.0e-4,
        patience=100,
        showplots=True,
    )
