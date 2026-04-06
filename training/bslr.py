from machinelearning.mdn import MixtureDensityNetwork
from config.experiment_config import ExperimentConfig
from config.plotting_config import PlottingConfig
import numpy as np
import torch
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# ── Storage for all trial loss histories ──────────────────────────────────────
all_results = []  # list of dicts: {label, train_loss, val_loss}

# ── Dataset loading (done once) ───────────────────────────────────────────────
DATASETS = ["data/H2H2_collisionsV2.csv"]
for dataset in DATASETS:
    print(f"Training on dataset: {dataset}")
    data = np.loadtxt(dataset, delimiter=",", skiprows=1)
    print(f"Dataset contains {data.shape[0]} rows")

    # Convert to variable set E_c, eta_trans, eta_rot_A
    inputdata = np.zeros((data.shape[0], 3))
    inputdata[:, 0] = np.sum(data[:, 0:3], axis=1)
    inputdata[:, 1] = data[:, 0] / inputdata[:, 0]
    inputdata[:, 2] = data[:, 1] / np.sum(data[:, 1:3], axis=1)

    outputdata = np.zeros((data.shape[0], 2))
    outputdata[:, 0] = data[:, 3] / np.sum(data[:, 3:6], axis=1)
    outputdata[:, 1] = data[:, 4] / np.sum(data[:, 4:6], axis=1)

    X = torch.tensor(inputdata, dtype=torch.float32)
    y = torch.tensor(outputdata, dtype=torch.float32)

    # ── Hyperopt search space ─────────────────────────────────────────────
    search_space = {
        "batch_size": hp.choice("batch_size", [32, 64, 128, 256]),
        "learning_rate": hp.loguniform(
            "learning_rate", np.log(1e-6), np.log(1e-3)
        ),  # centered around 1e-5
    }

    def objective(params):
        batch_size = params["batch_size"]
        lr = params["learning_rate"]
        label = f"bs={batch_size}, lr={lr:.2e}"
        print(f"\n{'=' * 60}")
        print(f"Trial: {label}")
        print(f"{'=' * 60}")

        config = ExperimentConfig()

        model = MixtureDensityNetwork(
            input_dim=3,
            output_dim=2,
            num_mixtures=config.num_mixtures,
            hidden_dim=config.hidden_dim,
            randomseed=config.random_seed,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader, val_loader = model.create_dataloaders(
            X,
            y,
            batch_size=batch_size,
            shuffle=config.shuffle,
            trainval_split=config.trainval_split,
            random_seed=config.random_seed,
        )

        train_loss_history, val_loss_history = model.train_model(
            train_loader,
            val_loader,
            optimizer,
            num_epochs=config.num_epochs,
            lr=lr,
        )

        # Store histories for plotting
        all_results.append(
            {
                "label": label,
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
                "batch_size": batch_size,
                "lr": lr,
            }
        )

        # Hyperopt minimises this value — use final validation loss
        best_val_loss = min(val_loss_history)
        return {"loss": best_val_loss, "status": STATUS_OK}

    # ── Run Hyperopt ──────────────────────────────────────────────────────
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=12,  # adjust as needed
        trials=trials,
    )

    # Resolve hp.choice index back to actual value
    batch_size_options = [32, 64, 128, 256]
    print(f"\nBest hyperparameters:")
    print(f"  batch_size  = {batch_size_options[best['batch_size']]}")
    print(f"  learning_rate = {best['learning_rate']:.2e}")

    # ── Plot all loss curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    for result in all_results:
        axes[0].plot(result["train_loss"], label=result["label"], alpha=0.8)
        axes[1].plot(result["val_loss"], label=result["label"], alpha=0.8)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Hyperopt MDN Hyperparameter Search — Loss Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig("hyperopt_loss_curves.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("Plot saved to hyperopt_loss_curves.png")
