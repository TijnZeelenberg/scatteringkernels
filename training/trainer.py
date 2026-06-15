"""Train a Gaussian Mixture Density Network on a CTC collision dataset.

Real implementation lives in `training.core`; this module re-exports the
function for backwards compatibility and provides a runnable `__main__`.
"""

from training.core import train_mdn
from config.experiment_config import ExperimentConfig
import paths

config = ExperimentConfig()

__all__ = ["train_mdn"]

h2_datapath = "data/ctc/h2/impactparam/Erelmax10000/H2_collisions_b1_6_uniform_Erelmax10000_ncoll1000000_seed42.npy"
o2_datapath = "data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax1_5.npy"

_datapaths = {"h2": h2_datapath, "o2": o2_datapath}

# One-time script: add a batch_size=750 model to the existing batch_size
# sweep results of H2 and O2. Output paths match the locations that
# visualization/create_plots.py reads for the sweep.
_sweep_outputpaths = {
    "h2": paths.RESULTS_DIR
    / "h2"
    / "models"
    / "mdn"
    / "batch_size"
    / "Erelmax10000"
    / "b1_6"
    / "n_epochs300"
    / "mdn_H2_bs750.pth",
    "o2": paths.RESULTS_DIR
    / "o2"
    / "models"
    / "mdn"
    / "batch_size"
    / "mdn_O2_bs750.pth",
}

if __name__ == "__main__":
    for gas in ["h2", "o2"]:
        datapath = _datapaths[gas]
        outputpath = paths.ensure_parent(_sweep_outputpaths[gas])
        train_mdn(
            datapath=str(datapath),
            outputpath=str(outputpath),
            epochs=config.num_epochs,
            batch_size=750,
            lr=config.learning_rate,
            patience=config.patience,
            showplots=True,
        )
