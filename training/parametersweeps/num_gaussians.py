"""Train one MDN per number of Gaussian mixtures.

Reads the default H2 dataset and trains a standard MDN for each value of
num_mixtures in the sweep.

Output: results/models/h2/mdn/num_gaussians/mdn_H2_ng{n}.pth
"""

from config.experiment_config import ExperimentConfig
from training.core import train_collision_model
import paths

config = ExperimentConfig()

# Sweep settings
num_gaussians_sweep = [1, 3, 5, 8, 10, 12, 15, 18, 20]

bfac_tag = str(config.bfac_h2).replace(".", "_")
dataset = (
    paths.DATA_DIR
    / "ctc/h2/impactparam/Erelmax10000"
    / f"H2_collisions_b{bfac_tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"
)
OUTPUT_DIR = paths.H2_MDN_DIR / "num_gaussians"

for ng in num_gaussians_sweep:
    config.num_mixtures = ng
    outputpath = OUTPUT_DIR / f"mdn_H2_ng{ng}.pth"

    print(f"\n{'=' * 60}")
    print(f"num_gaussians={ng}  →  {outputpath.name}")
    print(f"{'=' * 60}")

    train_collision_model(
        kind="mdn",
        datapath=dataset,
        outputpath=outputpath,
        epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        wf=None,
        patience=config.patience,
        config=config,
    )

print("\nDone. All models saved to:", OUTPUT_DIR)
