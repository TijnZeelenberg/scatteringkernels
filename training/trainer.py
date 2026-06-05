"""Train a Gaussian Mixture Density Network on a CTC collision dataset.

Real implementation lives in `training.core`; this module re-exports the
function for backwards compatibility and provides a runnable `__main__`.
"""

from training.core import train_mdn
from config.experiment_config import ExperimentConfig
import paths

config = ExperimentConfig()

__all__ = ["train_mdn"]


if __name__ == "__main__":
    datapath = (
        paths.DATA_DIR
        / "ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax1_5.npy"
    )
    outputpath = paths.ensure_parent(
        paths.RESULTS_DIR / "o2" / "models" / "mdn" / "best_model.pth"
    )
    train_mdn(
        datapath=str(datapath),
        outputpath=str(outputpath),
        epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        patience=config.patience,
        showplots=True,
    )
