from machinelearning.mdn import MixtureDensityNetwork
from config.experiment_config import ExperimentConfig
from config.plotting_config import PlottingConfig
import numpy as np
import torch
import matplotlib.pyplot as plt


def train_mdn(datapath, outputpath, epochs: int, batch_size: int, lr: float, wf: float = 1, patience: int = 30, showplots=False, pretrained_path: str | None = None):

    print(f"Training on dataset: {datapath}")

    # Load and check dataset
    if ".npy" in datapath:
        data = np.load(datapath)
        if data.shape[1] != 6:
            raise ValueError(
                f"Expected dataset with 6 columns (Etr, Erot1, Erot2, Etr', Erot1', Erot2'), but got {data.shape[1]} columns"
            )
    elif ".csv" in datapath:
        data = np.loadtxt(datapath, delimiter=",", skiprows=1)
        if data.shape[1] != 6:
            raise ValueError(
                f"Expected dataset with 6 columns (Etr, Erot1, Erot2, Etr', Erot1', Erot2'), but got {data.shape[1]} columns"
            )
    else:
        raise ValueError(f"Unsupported file format for dataset: {datapath}")
    print(f"Dataset contains {data.shape[0]} rows")

    # eta_trans_rel = E_trans_rel/ E_total, eta_rot_A = E_rot_A / (E_rot_A + E_rot_B)
    # Convert to variable set E_c, \eta_trans, \eta_rot_A
    inputdata = np.zeros((data.shape[0], 3))
    inputdata[:, 0] = np.sum(data[:, 0:3], axis=1)  # total energy
    inputdata[:, 1] = (
        data[:, 0] / inputdata[:, 0]
    )  # fraction of the total energy that is translational energy
    inputdata[:, 2] = data[:, 1] / np.sum(
        data[:, 1:3], axis=1
    )  # fraction of the total rotational energy that belongs to molecule A

    outputdata = np.zeros((data.shape[0], 2))
    outputdata[:, 0] = data[:, 3] / np.sum(data[:, 3:6], axis=1)
    outputdata[:, 1] = data[:, 4] / np.sum(data[:, 4:6], axis=1)

    X = torch.tensor(inputdata, dtype=torch.float32)
    y = torch.tensor(outputdata, dtype=torch.float32)

    # Weigh training samples according to translational energy (faster molecules are more likely to collide)
    E_rel_trans_pre = data[:, 0]
    sample_weights = E_rel_trans_pre * wf
    sample_weights = sample_weights / sample_weights.sum()
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    # Initialize model and training parameters
    config = ExperimentConfig()
    model = MixtureDensityNetwork(
        input_dim=3,
        output_dim=2,
        num_mixtures=config.num_mixtures,
        hidden_dim=config.hidden_dim,
        randomseed=config.random_seed,
    )
    if pretrained_path is not None:
        model.load_model(pretrained_path)
        print(f"Loaded pretrained weights from: {pretrained_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    # Train the model
    train_loader, val_loader = model.create_dataloaders(
        X,
        y,
        batch_size=batch_size,
        shuffle=config.shuffle,
        trainval_split=config.trainval_split,
        random_seed=config.random_seed,
        weights=sample_weights,
    )
    train_loss_history, val_loss_history = model.train_model(
        train_loader,
        val_loader,
        optimizer,
        num_epochs=epochs,
        lr=lr,
        patience=patience,
        scheduler=None,
    )

    # Save the trained model
    model.save_model(outputpath)
    print(f"Model saved to: {outputpath}")

    if showplots:
        # Plot training and validation loss histories
        plottingconfig = PlottingConfig()
        plt.figure(figsize=(plottingconfig.figsize))
        plt.plot(train_loss_history, label="Training Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel(
            "Epoch",
            fontsize=plottingconfig.label_fontsize,
            fontweight=plottingconfig.label_fontweight,
        )
        plt.ylabel(
            "Loss",
            fontsize=plottingconfig.label_fontsize,
            fontweight=plottingconfig.label_fontweight,
        )
        plt.legend(fontsize=plottingconfig.legend_fontsize)
        plt.show()

if __name__ == "__main__":
    config = ExperimentConfig()
    datapath = "data/H2H2_collisions.npy"
    outputpath = "results/models/mdn_H2H2v2.pth"
    train_mdn(datapath, outputpath, epochs=100, batch_size=128, lr=2.00e-4, wf=10.0, patience=50, showplots=True)
