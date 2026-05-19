"""Unified collision-model trainer.

The MDN and BetaMDN training loops were almost identical — same data prep, same
dataloader construction, same Adam + early-stopping recipe. This module exposes
a single `train_collision_model` entry point that takes a `model_kind` argument
and dispatches to the right class. Thin convenience wrappers `train_mdn` and
`train_beta_mdn` exist for callers who prefer the explicit form.

Trained model paths automatically get their parent directories created via
`paths.ensure_parent`, so you can call this with an output path several levels
deep and it will Just Work even on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch

from config.experiment_config import ExperimentConfig
from config.plotting_config import PlottingConfig
from machinelearning.beta_mdn import BetaMixtureDensityNetwork
from machinelearning.mdn import MixtureDensityNetwork
import paths
from training.data_prep import load_and_prepare


ModelKind = Literal["mdn", "beta_mdn"]


def _build_model(kind: str, config: ExperimentConfig):
    if kind == "mdn":
        return MixtureDensityNetwork(
            input_dim=3,
            output_dim=2,
            num_mixtures=config.num_mixtures,
            hidden_dim=config.hidden_dim,
            randomseed=config.random_seed,
        )
    if kind == "beta_mdn":
        return BetaMixtureDensityNetwork(
            input_dim=3,
            output_dim=2,
            num_mixtures=config.num_mixtures,
            hidden_dim=config.hidden_dim,
            randomseed=config.random_seed,
        )
    raise ValueError(f"Unknown model kind: {kind!r}")


def train_collision_model(
    kind: ModelKind,
    datapath: str | Path,
    outputpath: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    wf: float = 1.0,
    T_eq: float | None = None,
    patience: int = 30,
    showplots: bool = False,
    pretrained_path: str | Path | None = None,
    config: ExperimentConfig | None = None,
    db_lambda: float = 0.0,
):
    """Train an MDN-style collision model on a CTC dataset.

    Args:
        kind: "mdn" (Gaussian MDN) or "beta_mdn" (Beta MDN).
        datapath: input collision dataset (.npy or .csv).
        outputpath: destination .pth path; parent dirs are created automatically.
        epochs: maximum training epochs.
        batch_size: minibatch size.
        lr: Adam learning rate.
        wf: weighting-factor exponent (legacy polynomial knob). Ignored when
            `T_eq` is set.
        T_eq: equilibrium temperature in K. When set, sample weights use the
            exact NTC importance ratio √E_trans · exp(−E_total/T_eq) instead
            of the polynomial `wf` weighting — see `training/data_prep.py`.
        patience: early-stopping patience in epochs.
        showplots: if True, show a training/validation loss curve.
        pretrained_path: if given, load these weights before training.
        config: override the default ExperimentConfig (random seed, hidden_dim, ...).
        db_lambda: detailed-balance penalty strength (MDN only — Beta MDN
            currently ignores it). When > 0, forces the kernel to satisfy the
            equilibrium DB relation; data-prep skips time-reversal augmentation
            since the loss already evaluates both directions per pair.

    Returns:
        (model, train_loss_history, val_loss_history)
    """
    config = config or ExperimentConfig()
    outputpath = paths.ensure_parent(outputpath)

    if T_eq is not None:
        print(f"Training {kind} on dataset: {datapath}  (NTC importance weight, T_eq={T_eq} K)")
    else:
        print(f"Training {kind} on dataset: {datapath}  (polynomial weight, wf={wf})")
    if db_lambda > 0.0 and kind != "mdn":
        raise NotImplementedError(
            f"DB penalty is only wired up for kind='mdn', got kind={kind!r}"
        )
    # When the DB penalty is on, the loss itself evaluates both directions per
    # batch — augmenting the data with time-reversed rows would just double the
    # cost and halve effective batch size. So we skip augmentation in that case.
    augment = db_lambda == 0.0
    if db_lambda > 0.0:
        print(f"DB penalty enabled: λ={db_lambda} (time-reversal augmentation disabled)")
    X, y, sample_weights, raw = load_and_prepare(
        datapath, wf=wf, T_eq=T_eq, augment=augment
    )
    print(f"Dataset contains {raw.shape[0]} rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = _build_model(kind, config)
    if pretrained_path is not None:
        model.load_model(str(pretrained_path))
        print(f"Loaded pretrained weights from: {pretrained_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader = model.create_dataloaders(
        X,
        y,
        batch_size=batch_size,
        shuffle=config.shuffle,
        trainval_split=config.trainval_split,
        random_seed=config.random_seed,
        weights=sample_weights,
    )

    model = model.to(device)
    model._cast_normalization_tensors()

    train_kwargs: dict = {"num_epochs": epochs, "patience": patience}
    if kind == "mdn":
        train_kwargs["db_lambda"] = db_lambda
    train_hist, val_hist = model.train_model(
        train_loader,
        val_loader,
        optimizer,
        **train_kwargs,
    )

    model.save_model(str(outputpath))
    print(f"Model saved to: {outputpath}")

    if showplots:
        _plot_loss_history(train_hist, val_hist)

    return model, train_hist, val_hist


def _plot_loss_history(train_hist, val_hist):
    pc = PlottingConfig()
    plt.figure(figsize=pc.figsize)
    plt.plot(train_hist, label="Training Loss")
    plt.plot(val_hist, label="Validation Loss")
    plt.xlabel("Epoch", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight)
    plt.ylabel("Loss", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight)
    plt.legend(fontsize=pc.legend_fontsize)
    plt.show()


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers
# ---------------------------------------------------------------------------


def train_mdn(datapath, outputpath, **kwargs):
    """Train a Gaussian MDN. See `train_collision_model` for kwargs."""
    return train_collision_model("mdn", datapath, outputpath, **kwargs)


def train_beta_mdn(datapath, outputpath, **kwargs):
    """Train a Beta MDN. See `train_collision_model` for kwargs."""
    return train_collision_model("beta_mdn", datapath, outputpath, **kwargs)
