from machinelearning.mdn import MixtureDensityNetwork
from config.experiment_config import ExperimentConfig
from config.plotting_config import PlottingConfig
import numpy as np
import torch
import matplotlib.pyplot as plt

DATASETS = ["data/H2H2_collisionsV2.csv", "data/O2O2_collisions.csv"]

for dataset in DATASETS:
    print(f"Training on dataset: {dataset}")

    # Load dataset
    data = np.loadtxt(dataset, delimiter=",", skiprows=1)
    print(f"Dataset contains {data.shape[0]} rows")

    # Convert to variable set E_c, \eta_trans, \eta_rot_A
    inputdata = np.zeros((data.shape[0], 3))
    inputdata[:, 0] = np.sum(data[:, 0:3], axis=1) # total energy
    inputdata[:, 1] = data[:, 0] / inputdata[:, 0] # fraction of the total energy that is translational energy
    inputdata[:, 2] = data[:, 1] / np.sum(data[:, 1:3], axis=1) # fraction of the total rotational energy that belongs to molecule A

    outputdata = np.zeros((data.shape[0], 2))
    outputdata[:, 0] = data[:, 3] / np.sum(data[:, 3:6], axis=1)
    outputdata[:, 1] = data[:, 4] / np.sum(data[:, 4:6], axis=1)

    X = torch.tensor(inputdata, dtype=torch.float32)
    y = torch.tensor(outputdata, dtype=torch.float32)

    # Initialize model and training parameters
    config = ExperimentConfig()
    model = MixtureDensityNetwork(
        input_dim=3,
        output_dim=2,
        num_mixtures=config.num_mixtures,
        hidden_dim=config.hidden_dim,
        randomseed=config.random_seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model
    train_loader, val_loader = model.create_dataloaders(
        X,
        y,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        trainval_split=config.trainval_split,
        random_seed=config.random_seed,
    )
    train_loss_history, val_loss_history = model.train_model(
        train_loader,
        val_loader,
        optimizer,
        num_epochs=config.num_epochs,
        lr=config.learning_rate,
    )

    # Save the trained model
    if "H2H2" in dataset:
        model.save_model("results/models/mdn_H2H2V2.pth")
    elif "O2O2" in dataset:
        model.save_model("results/models/mdn_O2O2.pth")

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
    if "H2H2" in dataset:
        plt.savefig("results/plots/H2H2_loss_history.png")
    elif "O2O2" in dataset:
        plt.savefig("results/plots/O2O2_loss_history.png")
    plt.close()

