"""Train one MDN per number of Gaussian mixtures.

Reads the default O2 dataset and trains a standard MDN for each value of
num_mixtures in the sweep.

Output: results/models/mdn/num_gaussians/o2/mdn_O2_ng{n}.pth
"""

from config.experiment_config import ExperimentConfig
from training.core import train_collision_model
import paths

config = ExperimentConfig()

# Sweep settings
num_gaussians_sweep = [1, 3, 5, 8, 10, 12, 15, 18, 20]

bfac_tag = str(config.bfac_o2).replace(".", "_")
dataset = (
    paths.DATA_DIR
    / "ctc/o2/impactparam/Erelmax10000"
    / f"O2_collisions_uniform_bmax{bfac_tag}.npy"
)
OUTPUT_DIR = paths.O2_MDN_DIR / "num_gaussians"

for ng in num_gaussians_sweep:
    config.num_mixtures = ng
    outputpath = OUTPUT_DIR / f"mdn_O2_ng{ng}.pth"

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
