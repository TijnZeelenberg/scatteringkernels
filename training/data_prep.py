"""Shared collision-dataset preparation.

Both MDN and BetaMDN training paths take a CTC `.npy`/`.csv` collision dataset
with columns (Etr, Erot1, Erot2, Etr', Erot1', Erot2') and convert it to:

    X (Etot, eta_tr, eta_rot_A)        - 3-d input features
    y (eta_tr_post, eta_rot_A_post)    - 2-d output fractions
    sample_weights                     - importance weights for training (or None)

Two sample-weighting schemes are available:

  * `wf` (polynomial)   →  w_i ∝ E_trans,i**wf
        The ad-hoc knob used in the original codebase. Approximates the NTC
        bias DSMC applies at inference time, but as a polynomial only.

  * `uniform_weights=True`  →  weights=None (unweighted NLL, every sample equal)

Energies are stored in CTC datasets as E / k_B in Kelvin (see the numba
generator in `ctc_adjusted/`).
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


EXPECTED_COLS = 6


def load_collision_dataset(datapath: str | Path):
    """Load a CTC collision dataset and return the raw (N, 6) array.

    Raises a clear error if the file doesn't have the expected layout.
    """
    datapath = str(datapath)
    if datapath.endswith(".npy"):
        data = np.load(datapath)
    elif datapath.endswith(".csv"):
        data = np.loadtxt(datapath, delimiter=",", skiprows=1)
    else:
        raise ValueError(f"Unsupported file format for dataset: {datapath}")

    if data.shape[1] != EXPECTED_COLS:
        raise ValueError(
            f"Expected dataset with {EXPECTED_COLS} columns "
            f"(Etr, Erot1, Erot2, Etr', Erot1', Erot2'), got {data.shape[1]}"
        )
    return data


def polynomial_weight(data: np.ndarray, wf: float) -> np.ndarray:
    """Original `wf` knob: w_i ∝ E_trans,i**wf, normalised to sum to 1."""
    w = data[:, 0] ** wf
    return w / w.sum()


def prepare_training_tensors(
    data: np.ndarray,
    wf: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert raw collision data to (X, y, sample_weights) tensors.

    Args:
        data: (N, 6) array as returned by `load_collision_dataset`.
        wf: polynomial weighting exponent: w_i ∝ E_trans,i**wf. When None
            (default), every sample is weighted equally (unweighted NLL).

    Returns:
        X: (N, 3) tensor of (Etot, eta_tr, eta_rot_A).
        y: (N, 2) tensor of (eta_tr_post, eta_rot_A_post).
        weights: (N,) tensor of per-sample weights, or None when `wf` is None.
    """
    n = data.shape[0]
    X = np.zeros((n, 3))
    X[:, 0] = np.sum(data[:, 0:3], axis=1)              # E_total
    X[:, 1] = data[:, 0] / X[:, 0]                       # eta_tr
    X[:, 2] = data[:, 1] / np.sum(data[:, 1:3], axis=1)  # eta_rot_A

    y = np.zeros((n, 2))
    y[:, 0] = data[:, 3] / np.sum(data[:, 3:6], axis=1)
    y[:, 1] = data[:, 4] / np.sum(data[:, 4:6], axis=1)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    if wf is None:
        return X_t, y_t, None

    weights = polynomial_weight(data, wf=wf)
    return X_t, y_t, torch.tensor(weights, dtype=torch.float32)


def load_and_prepare(
    datapath: str | Path,
    wf: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, np.ndarray]:
    """Convenience: load + convert in one step. Also returns the raw array.

    When `wf` is None (default), training uses an unweighted loss. Pass a
    float to apply polynomial importance weighting w_i ∝ E_trans,i**wf.
    """
    raw = load_collision_dataset(datapath)
    X, y, weights = prepare_training_tensors(raw, wf=wf)
    return X, y, weights, raw
