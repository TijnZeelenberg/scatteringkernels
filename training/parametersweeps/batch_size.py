"""Train an mdn for each batch size in the batch size sweep.

Reads datasets produced by ctc_adjusted/ctc_h2_impactparamsweep.py and trains
a standard MDN (config from ExperimentConfig) for each bfac value.

Output: results/models/mdn/impactparam/mdn_H2_b{bfac_tag}.pth
"""

from config.experiment_config import ExperimentConfig
from training.core import train_collision_model
import paths

config = ExperimentConfig()

# Sweep settings
batch_sizes = [1000, 2000, 5000, 10000, 12500, 15625]
ncoll = 1000000
seed = 42
dist_tag = "uniform_Erelmax10000"  # distribution tag

bfac = config.bfac
bfac_tag = str(bfac).replace(".", "_")
dataset = (
    paths.DATA_DIR
    / "ctc/H2/impactparam/Erelmax10000"
    / f"H2_collisions_b{bfac_tag}_{dist_tag}_ncoll{ncoll}_seed{seed}.npy"
)
OUTPUT_DIR = paths.MDN_DIR / "batch_size/Erelmax10000"

for bs in batch_sizes:
    bs_tag = str(bs)

    outputpath = OUTPUT_DIR / f"mdn_H2_b{bs_tag}.pth"

    print(f"\n{'=' * 60}")
    print(f"batch_size={bs}  →  {outputpath.name}")
    print(f"{'=' * 60}")

    train_collision_model(
        kind="mdn",
        datapath=dataset,
        outputpath=outputpath,
        epochs=config.num_epochs,
        batch_size=bs,
        lr=config.learning_rate,
        wf=None,
        patience=config.patience,
        config=config,
    )

print("\nDone. All models saved to:", OUTPUT_DIR)
