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
DATASETS = ["data/filtered/O2O2_collisions.csv"]
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
        "learning_rate": hp.loguniform("learning_rate", np.log(5e-5), np.log(1e-3)),
        "num_mixtures": hp.choice("num_mixtures", [4, 8, 12, 16, 24]),
        "hidden_dim": hp.choice("hidden_dim", [64, 128, 256, 512]),
        "dropout": hp.uniform("dropout", 0.0, 0.3),
    }

    NUM_EPOCHS = 75

    def objective(params):
        batch_size = params["batch_size"]
        lr = params["learning_rate"]
        num_mix = params["num_mixtures"]
        hid_dim = params["hidden_dim"]
        dropout = params["dropout"]
        label = f"bs={batch_size}, lr={lr:.2e}, mix={num_mix}, hid={hid_dim}, do={dropout:.2f}"
        print(f"\n{'=' * 60}")
        print(f"Trial {len(all_results) + 1}/50: {label}")
        print(f"{'=' * 60}")

        config = ExperimentConfig()

        model = MixtureDensityNetwork(
            input_dim=3,
            output_dim=2,
            num_mixtures=num_mix,
            hidden_dim=hid_dim,
            randomseed=config.random_seed,
            dropout=dropout,
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
            num_epochs=NUM_EPOCHS,
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
                "num_mixtures": num_mix,
                "hidden_dim": hid_dim,
                "dropout": dropout,
            }
        )

        # Hyperopt minimises this value — use best validation loss
        best_val_loss = min(val_loss_history)
        return {"loss": best_val_loss, "status": STATUS_OK}

    # ── Run Hyperopt ──────────────────────────────────────────────────────
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
    )

    # Resolve hp.choice indices back to actual values
    batch_size_options = [32, 64, 128, 256]
    num_mixtures_options = [4, 8, 12, 16, 24]
    hidden_dim_options = [64, 128, 256, 512]
    print(f"\nBest hyperparameters:")
    print(f"  batch_size    = {batch_size_options[best['batch_size']]}")
    print(f"  learning_rate = {best['learning_rate']:.2e}")
    print(f"  num_mixtures  = {num_mixtures_options[best['num_mixtures']]}")
    print(f"  hidden_dim    = {hidden_dim_options[best['hidden_dim']]}")
    print(f"  dropout       = {best['dropout']:.3f}")

    # ── Sort results by best validation loss ──────────────────────────────
    sorted_results = sorted(all_results, key=lambda r: min(r["val_loss"]))

    # ── Plot top 10 loss curves (50 would be unreadable) ──────────────────
    top_n = 10
    top_results = sorted_results[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

    for result in top_results:
        axes[0].plot(result["train_loss"], label=result["label"], alpha=0.8)
        axes[1].plot(result["val_loss"], label=result["label"], alpha=0.8)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=6, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=6, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Hyperopt MDN Search — Top {top_n} of {len(all_results)} Trials (300 epochs)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("hyperopt_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to hyperopt_loss_curves.png (showing top {top_n} trials)")

    # ── Print top 10 results table ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"Top {top_n} trials by best validation loss:")
    print(f"{'=' * 80}")
    print(f"{'Rank':<5} {'Val Loss':<10} {'BS':<5} {'LR':<12} {'Mix':<5} {'Hid':<5} {'Drop':<6}")
    print(f"{'-' * 48}")
    for i, r in enumerate(top_results):
        print(
            f"{i + 1:<5} {min(r['val_loss']):<10.4f} {r['batch_size']:<5} "
            f"{r['lr']:<12.2e} {r['num_mixtures']:<5} {r['hidden_dim']:<5} {r['dropout']:<6.2f}"
        )
