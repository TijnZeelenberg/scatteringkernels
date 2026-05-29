"""Train one MDN per impact-parameter value from the bfac sweep datasets.

Reads datasets produced by ctc_adjusted/ctc_h2_impactparamsweep.py and trains
a standard MDN (config from ExperimentConfig) for each bfac value.

Output: results/models/mdn/impactparam/mdn_H2_b{bfac_tag}.pth
"""

from tqdm import tqdm

from config.experiment_config import ExperimentConfig
from training.core import train_collision_model
import paths

# ---------------------------------------------------------------------------
# Sweep settings — must match ctc_adjusted/ctc_h2_impactparamsweep.py
# ---------------------------------------------------------------------------
bfac_sweep = [
    1.6,
    1.7,
    1.8,
]  # impact-parameter sweep values
ncoll = 1000000
seed = 42
dist_tag = "uniform_Erelmax10000"  # distribution tag

OUTPUT_DIR = paths.ensure_dir(paths.MDN_DIR / "impactparam/Erelmax10000")

config = ExperimentConfig()

for bfac in tqdm(bfac_sweep, desc="bfac sweep", unit="model"):
    bfac_tag = str(bfac).replace(".", "_")
    dataset = (
        paths.DATA_DIR
        / "ctc/H2/impactparam"
        / f"H2_collisions_b{bfac_tag}_{dist_tag}_ncoll{ncoll}_seed{seed}.npy"
    )

    if not dataset.exists():
        print(f"[SKIP] Dataset not found: {dataset}")
        continue

    outputpath = OUTPUT_DIR / f"mdn_H2_b{bfac_tag}.pth"

    print(f"\n{'=' * 60}")
    print(f"bfac={bfac}  →  {outputpath.name}")
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
