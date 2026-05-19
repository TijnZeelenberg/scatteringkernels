"""Shared collision-dataset preparation.

Both MDN and BetaMDN training paths take a CTC `.npy`/`.csv` collision dataset
with columns (Etr, Erot1, Erot2, Etr', Erot1', Erot2') and convert it to:

    X (Etot, eta_tr, eta_rot_A)        - 3-d input features
    y (eta_tr_post, eta_rot_A_post)    - 2-d output fractions
    sample_weights                     - importance weights for training

Two sample-weighting schemes are available:

  * `wf` (polynomial)   →  w_i ∝ E_trans,i**wf
        The ad-hoc knob used in the original codebase. Approximates the NTC
        bias DSMC applies at inference time, but as a polynomial only —
        it can't reproduce the high-energy exponential cutoff of the true
        importance ratio, which is why the optimal `wf` is random-seed
        sensitive and why some sweeps fail to converge in DSMC.

  * `T_eq` (NTC importance ratio)  →  w_i ∝ √E_trans,i · exp(−E_total,i / T_eq)
        The exact importance ratio between CTC's uniform sampling and the
        |g| × Maxwell-Boltzmann distribution DSMC actually feeds the kernel
        under NTC selection at temperature T_eq. Setting T_eq replaces the
        polynomial knob with the principled correction — no per-seed tuning
        needed.

Energies are stored in CTC datasets as E / k_B in Kelvin (see the numba
generator in `ctc_adjusted/`), and T_eq is also in K, so the dimensionless
exponent collapses to `E_total_K / T_eq`.
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


def time_reverse_augment(data: np.ndarray) -> np.ndarray:
    """Augment a CTC collision dataset with its time-reversed counterpart.

    Classical CTC dynamics are time-reversal symmetric: every recorded
    transition (E_tr, E_rot1, E_rot2) → (E_tr', E_rot1', E_rot2') is matched
    by an equally valid reverse trajectory (E_tr', E_rot1', E_rot2') →
    (E_tr, E_rot1, E_rot2). Stacking both directions into the training set
    forces the MDN to learn a kernel that respects detailed balance — which a
    vanilla NLL-trained MDN otherwise breaks, producing the equipartition
    bias we see in DSMC relaxation experiments. Bonus: doubles the effective
    sample count.
    """
    reversed_data = np.concatenate([data[:, 3:6], data[:, 0:3]], axis=1)
    return np.concatenate([data, reversed_data], axis=0)


def polynomial_weight(data: np.ndarray, wf: float) -> np.ndarray:
    """Original `wf` knob: w_i ∝ E_trans,i**wf, normalised to sum to 1."""
    w = data[:, 0] ** wf
    return w / w.sum()


def ntc_importance_weight(data: np.ndarray, T_eq: float) -> np.ndarray:
    """Exact NTC importance ratio for uniform-CTC → NTC-at-T_eq.

    DSMC under NTC accepts a candidate pair with probability proportional to
    |v_i − v_j|, on top of a Maxwell-Boltzmann velocity distribution at the
    equilibrium temperature `T_eq`. Marginalised onto the kernel's input
    variables, the resulting density is

        p_NTC(E_trans)  ∝  √E_trans · exp(−E_trans / T_eq)        (Γ(3/2, T_eq))
        p_NTC(E_rot_i)  ∝  exp(−E_rot_i / T_eq)                   (2-DOF Boltzmann)

    CTC's training data was generated with uniform sampling on E_trans and
    each E_rot_i, so p_CTC is constant. The importance ratio is therefore

        w_i  ∝  √E_trans,i · exp(−E_total,i / T_eq)

    which has a *polynomial* √E_trans part **and** an *exponential* cutoff at
    high E_total. The `wf` polynomial knob can only reproduce the first part —
    the missing exp(-E_total/T_eq) cutoff is the reason it's seed-fragile.

    Args:
        data: raw collision array, shape (N, 6). data[:, 0] = E_trans in K,
            data[:, 0:3] sums to E_total in K.
        T_eq: equilibrium temperature in K (the same K the energies are in).

    Returns:
        Normalised weights, shape (N,).
    """
    if T_eq <= 0:
        raise ValueError(f"T_eq must be positive, got {T_eq!r}")
    E_trans = np.maximum(data[:, 0], 0.0)
    E_total = data[:, 0:3].sum(axis=1)
    log_w = 0.5 * np.log(E_trans + 1e-30) - E_total / T_eq
    log_w -= log_w.max()  # numerical stability before exponentiating
    w = np.exp(log_w)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(
            f"NTC weights collapsed to zero at T_eq={T_eq} K — the dataset's E_total "
            "range is far from T_eq. Pick a T_eq in the regime your data covers."
        )
    return w / s


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS = (Σw)^2 / Σw^2. Tells you how many "real" samples the
    weighted loss is averaging over — useful sanity check when picking T_eq."""
    s1 = float(weights.sum())
    s2 = float((weights ** 2).sum())
    return s1 * s1 / s2 if s2 > 0 else 0.0


def prepare_training_tensors(
    data: np.ndarray,
    wf: float = 1.0,
    T_eq: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw collision data to (X, y, sample_weights) tensors.

    Args:
        data: (N, 6) array as returned by `load_collision_dataset`.
        wf: NTC weighting-factor exponent (legacy polynomial knob). Used only
            when `T_eq is None`.
        T_eq: equilibrium temperature in K. When set, sample weights use the
            exact NTC importance ratio (√E_trans · exp(−E_total / T_eq))
            and `wf` is ignored.

    Returns:
        X: (N, 3) tensor of (Etot, eta_tr, eta_rot_A).
        y: (N, 2) tensor of (eta_tr_post, eta_rot_A_post).
        weights: (N,) tensor of per-sample weights.
    """
    n = data.shape[0]
    X = np.zeros((n, 3))
    X[:, 0] = np.sum(data[:, 0:3], axis=1)              # E_total
    X[:, 1] = data[:, 0] / X[:, 0]                       # eta_tr
    X[:, 2] = data[:, 1] / np.sum(data[:, 1:3], axis=1)  # eta_rot_A

    y = np.zeros((n, 2))
    y[:, 0] = data[:, 3] / np.sum(data[:, 3:6], axis=1)
    y[:, 1] = data[:, 4] / np.sum(data[:, 4:6], axis=1)

    if T_eq is not None:
        weights = ntc_importance_weight(data, T_eq=T_eq)
    else:
        weights = polynomial_weight(data, wf=wf)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def load_and_prepare(
    datapath: str | Path,
    wf: float = 1.0,
    T_eq: float | None = None,
    augment: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Convenience: load + convert in one step. Also returns the raw array.

    See `prepare_training_tensors` for `wf` vs `T_eq` semantics. When
    `augment=True` (default), the dataset is doubled with time-reversed
    collisions before tensor conversion — see `time_reverse_augment`. The
    returned `raw` array reflects the (post-augmentation) data that was
    actually used to build the tensors.
    """
    raw = load_collision_dataset(datapath)
    if augment:
        raw_orig_n = len(raw)
        raw = time_reverse_augment(raw)
        print(f"Time-reversal augmentation: {raw_orig_n} -> {len(raw)} rows")
    X, y, weights = prepare_training_tensors(raw, wf=wf, T_eq=T_eq)
    if T_eq is not None:
        ess = effective_sample_size(weights.detach().cpu().numpy())
        print(
            f"NTC importance weights @ T_eq={T_eq} K: "
            f"ESS = {ess:.0f} / {len(weights)} ({ess / len(weights) * 100:.2f}%)"
        )
    return X, y, weights, raw
