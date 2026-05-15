"""Train collision models across a range of NTC weighting factors.

`run_wf_sweep` is the reusable function — both this module and
`training.betamdn_wfsweep` call it. Run this file directly to reproduce the
Gaussian-MDN sweep with the defaults that match the rest of the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tqdm import tqdm

import paths
from config.experiment_config import ExperimentConfig
from training.core import train_collision_model


DEFAULT_WEIGHTS: tuple[float, ...] = (0.25, 0.5, 1, 2, 3, 4, 5, 6, 7)


def run_wf_sweep(
    kind: str,
    datapath: str | Path,
    tag: str,
    weights: Iterable[float] = DEFAULT_WEIGHTS,
    *,
    trainseed: int | None = None,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 2.0e-4,
    patience: int = 100,
    showplots: bool = False,
) -> dict[float, Path]:
    """Train one model per weighting factor in `weights`.

    Args:
        kind: "mdn" or "beta_mdn".
        datapath: input collision dataset.
        tag: short identifier used to namespace the output directory, e.g.
             "H2_400000_dataseed42".
        weights: iterable of NTC weighting-factor exponents to sweep over.
        trainseed: when set, overrides the ExperimentConfig random seed and
            saves models under a `trainseed<N>/` subdirectory of the sweep.
        epochs, batch_size, lr, patience, showplots: forwarded to the trainer.

    Returns:
        Mapping from wf -> saved model path.
    """
    config: ExperimentConfig | None = None
    if trainseed is not None:
        config = ExperimentConfig()
        config.random_seed = trainseed

    saved: dict[float, Path] = {}
    weights = list(weights)
    for wf in tqdm(
        weights, desc=f"Training {kind} across wf", unit="weight"
    ):
        outputpath = paths.wf_sweep_model_path(kind, tag, wf, trainseed=trainseed)
        train_collision_model(
            kind=kind,  # type: ignore[arg-type]
            datapath=str(datapath),
            outputpath=str(outputpath),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            wf=wf,
            patience=patience,
            showplots=showplots,
            config=config,
        )
        saved[wf] = outputpath
    return saved


if __name__ == "__main__":
    datapath = paths.DATA_DIR / "H2H2_collisions_numba_b1_0_400000_seed42.npy"
    run_wf_sweep("mdn", datapath, tag="H2_400000_dataseed42")
