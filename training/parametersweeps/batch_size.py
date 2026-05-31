"""Train an mdn for each batch size in the batch size sweep.

Reads datasets produced by ctc_adjusted/ctc_o2_impactparamsweep.py and trains
a standard MDN (config from ExperimentConfig) for each batch size value.

Output: results/models/mdn/batch_size/o2/Erelmax10000/mdn_O2_b{bs_tag}.pth
"""

from config.experiment_config import ExperimentConfig
from training.core import train_collision_model
import paths

config = ExperimentConfig()

# Sweep settings
batch_sizes = [1000, 2000, 5000, 10000, 12500, 15625]

bfac_tag = str(config.bfac_o2).replace(".", "_")
dataset = (
    paths.DATA_DIR
    / "ctc/o2/impactparam/Erelmax10000"
    / f"O2_collisions_uniform_bmax{bfac_tag}.npy"
)
OUTPUT_DIR = paths.O2_MDN_DIR / "batch_size"

for bs in batch_sizes:
    bs_tag = str(bs)

    outputpath = OUTPUT_DIR / f"mdn_O2_bs{bs_tag}.pth"

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
