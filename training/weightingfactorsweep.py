from training.trainer import train_mdn
from config.experiment_config import ExperimentConfig
from tqdm import tqdm


config = ExperimentConfig()
datapath = "data/H2H2_collisions_numba_b1_0_200000_seed41.npy"
weights = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7]
for wf in tqdm(
    weights, desc="Training models with different weighting factors", unit="weight"
):
    outputpath = f"results/models/weightsensitivity/O2_200000_dataseed41/trainseed42/mdn_H2_wf{str(wf).replace('.', '_')}.pth"
    train_mdn(
        datapath,
        outputpath,
        epochs=100,
        batch_size=128,
        lr=2.00e-4,
        wf=wf,
        patience=100,
        showplots=False,
    )
