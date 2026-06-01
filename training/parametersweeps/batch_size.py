"""Train an mdn for each batch size in the batch size sweep.

Reads datasets produced by ctc_adjusted/ctc_h2_impactparamsweep.py and trains
a standard MDN (config from ExperimentConfig) for each batch size value.

Output: results/models/h2/mdn/batch_size/mdn_H2_bs{bs_tag}.pth
"""

from config.experiment_config import ExperimentConfig
from training.core import train_collision_model
import paths

config = ExperimentConfig()

# Sweep settings
batch_sizes = [1000, 2000, 5000, 10000, 12500, 15625]

bfac_tag = str(config.bfac_h2).replace(".", "_")
dataset = (
    paths.DATA_DIR
    / "ctc/h2/impactparam/Erelmax10000"
    / f"H2_collisions_b{bfac_tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"
)
OUTPUT_DIR = paths.H2_MDN_DIR / "batch_size"

for bs in batch_sizes:
    bs_tag = str(bs)

    outputpath = OUTPUT_DIR / f"mdn_H2_bs{bs_tag}.pth"

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
