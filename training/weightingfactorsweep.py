import matplotlib.pyplot as plt
from training.trainer import train_mdn
from config.experiment_config import ExperimentConfig
from tqdm import tqdm


config = ExperimentConfig()
datapath = "data/O2O2_collisions_uniform.npy"
weights = [0.25, 0.5, 1, 2, 4, 8]
for wf in tqdm(
    weights, desc="Training models with different weighting factors", unit="weight"
):
    outputpath = (
        f"results/models/weightsensitivity/mdn_O2_wf{str(wf).replace('.', '_')}.pth"
    )
    train_mdn(
        datapath,
        outputpath,
        epochs=100,
        batch_size=128,
        lr=2.00e-4,
        wf=wf,
        patience=50,
        showplots=False,
    )
