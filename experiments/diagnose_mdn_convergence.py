"""Side-by-side diagnostics for two MDN scattering kernels.

Use this when one wf model converges in DSMC and another doesn't, to localize
the failure mode. Compares a "good" and a "bad" model against the CTC ground
truth across four panels:

  1. Out-of-bounds sample rate vs E_total
       Gaussian MDNs sample outside [0,1] and rely on a hard clip in
       `collide()`. A clipping rate that's high (or rising at low E) usually
       indicates components placed near the [0,1] boundary, which creates
       Dirac spikes at the edges and breaks detailed balance.

  2. Mixture components in (eta_tr', eta_rot_A') space
       For one representative input per E bin, plot the K mixture means and
       1-sigma ellipses. Mode collapse (components overlapping) or components
       parked outside [0,1] are obvious failure signals.

  3. Conditional sample scatter at low / mid / high E_total
       Run each model on the same inputs the CTC dataset uses, compare the
       resulting (eta_tr', eta_rot_A') cloud to the CTC truth. Useful for
       eye-balling regime mismatches.

  4. KL divergence to CTC vs E_total
       Per E_total bin, KL(CTC || sampled) for both output dimensions. The
       bin where the bad model loses ground is usually low E.

  5. Mean response curve E[η_tr' | η_tr_in]  ← the DSMC-relevant test
       The fixed point (where the curve crosses y=x) is the equilibrium that
       iterating this kernel drives to. A model can fit the marginal CTC
       distribution well and still place its fixed point in the wrong spot —
       which is exactly the failure mode you see when DSMC doesn't relax.
       What to look for: does the curve cross y=x at the same η_tr as CTC's
       curve does? Is the crossing stable (slope < 1 from above)? If wf3's
       fixed point is at a different η_tr than CTC's, that's the smoking gun.

  6. Conditional spread σ[η' | η_in] + response slope at fixed point
       If the mean response (panel 5) looks fine but DSMC still doesn't
       relax, the problem is likely the *spread*, not the *mean*. A kernel
       with mode collapse has correct mean response but very narrow
       conditional distribution, so iterating it barely moves the state per
       step. The printed slope at the fixed point tells the same story from
       another angle: slope ≈ 1 means marginal stability (vanishingly slow
       relaxation), slope ≈ 0 means strong pull toward equilibrium.

  7. Iterative kernel from thermal equilibrium  ← the most DSMC-like test
       Initialize a particle pool at Maxwell-Boltzmann equilibrium (where
       <η_tr> = 3/7 by equipartition for diatomic pairs), repeatedly pair
       particles and apply each kernel. This is DSMC without the spatial
       structure. A kernel that preserves equilibrium produces flat
       trajectories; one that doesn't shows visible drift over O(100)
       iterations. Failure mode that hides from every marginal in panels
       1-6 still reveals itself here because we're testing the actual
       composition of many kernel applications.

  8. NTC-biased iterative kernel  ← the DSMC selection-bias test
       Same setup as panel 7, but candidate pairs are accepted with
       probability |v_i − v_j| / v_rmax (NTC selection). This over-samples
       high-relative-velocity pairs, just as real DSMC does — matching the
       training-time E_trans^wf weighting. If wf3 stabilises in panel 7 but
       diverges (or stabilises further from equipartition) in panel 8, the
       failure is caused by mismatch between training distribution and the
       NTC-biased inference distribution.

Run:
    python experiments/diagnose_mdn_convergence.py \\
        --good results/models/mdn/weightsensitivity/H2_200000_dataseed41/mdn_H2_wf7.pth \\
        --bad  results/models/mdn/weightsensitivity/H2_200000_dataseed41/mdn_H2_wf3.pth \\
        --dataset data/H2H2_collisions_numba_b1_0_200000_seed41.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

import paths
from analysis.kl_divergence import kl_divergence
from experiments.energy_relaxation import load_mdn
from physics.species import Species
from training.data_prep import load_collision_dataset, prepare_training_tensors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bin_indices(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin a 1-D array. Returns (bin_assignments, bin_edges)."""
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9  # so the max value lands in the last bin
    return np.digitize(values, edges[1:-1]), edges


def _sample_from(model, X: torch.Tensor) -> np.ndarray:
    """Draw one sample per input row from `model`. Returns (N, 2)."""
    with torch.no_grad():
        return model.sample(X).detach().cpu().numpy()


def _forward_params(model, x_row: torch.Tensor):
    """Return (pi, mu_denorm, sigma_denorm) for a single (1, 3) input."""
    model.eval()
    with torch.no_grad():
        x_norm = (x_row - model.input_mean) / model.input_std
        pi, mu, sigma = model.forward(x_norm)
        # de-normalize means/sigmas back to physical [0,1] coordinates
        mu_denorm = mu * model.output_std + model.output_mean
        sigma_denorm = sigma * model.output_std
    return (
        pi.squeeze(0).cpu().numpy(),
        mu_denorm.squeeze(0).cpu().numpy(),
        sigma_denorm.squeeze(0).cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Diagnostic panels
# ---------------------------------------------------------------------------


def plot_clipping_rate(
    models: dict[str, object],
    X: torch.Tensor,
    raw: np.ndarray,
    n_bins: int = 10,
    samples_per_input: int = 1,
):
    """Panel 1: fraction of samples landing outside [0,1] per E_total bin."""
    E_total = raw[:, 0:3].sum(axis=1)
    bin_idx, edges = _bin_indices(E_total, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, model in models.items():
        out_of_bounds_tr = np.zeros(n_bins)
        out_of_bounds_rot = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            samples_per_input_total = []
            for _ in range(samples_per_input):
                samples_per_input_total.append(_sample_from(model, X[mask]))
            samples_array = np.concatenate(samples_per_input_total, axis=0)
            out_of_bounds_tr[b] = np.mean(
                (samples_array[:, 0] < 0) | (samples_array[:, 0] > 1)
            )
            out_of_bounds_rot[b] = np.mean(
                (samples_array[:, 1] < 0) | (samples_array[:, 1] > 1)
            )

        axes[0].plot(centers, 100 * out_of_bounds_tr, marker="o", label=label)
        axes[1].plot(centers, 100 * out_of_bounds_rot, marker="o", label=label)

    for ax, title in zip(
        axes, [r"$\eta'_{tr}$ out of $[0,1]$", r"$\eta'_{rot,A}$ out of $[0,1]$"]
    ):
        ax.set_title(title)
        ax.set_xlabel(r"$E_{total}$ [K]")
        ax.set_ylabel("% of samples clipped")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Panel 1 — Out-of-bounds sampling rate (lower is better)")
    fig.tight_layout()
    return fig


def plot_mixture_components(
    models: dict[str, object],
    X: torch.Tensor,
    raw: np.ndarray,
    n_bins: int = 3,
):
    """Panel 2: mixture means + 1-sigma ellipses for one input per E bin."""
    E_total = raw[:, 0:3].sum(axis=1)
    bin_idx, _ = _bin_indices(E_total, n_bins)

    # Pick one representative input per bin (median E_total within bin)
    rep_inputs = []
    rep_labels = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        E_bin = E_total[mask]
        idx_local = np.argsort(E_bin)[len(E_bin) // 2]
        idx_global = np.where(mask)[0][idx_local]
        rep_inputs.append(X[idx_global : idx_global + 1])
        rep_labels.append(
            f"E≈{E_total[idx_global]:.0f}K, ηₜᵣ={raw[idx_global, 0] / E_total[idx_global]:.2f}"
        )

    fig, axes = plt.subplots(
        len(models), len(rep_inputs), figsize=(4.5 * len(rep_inputs), 4 * len(models))
    )
    axes = np.atleast_2d(axes)
    for r, (label, model) in enumerate(models.items()):
        for c, (xrow, rlabel) in enumerate(zip(rep_inputs, rep_labels)):
            pi, mu, sigma = _forward_params(model, xrow)
            ax = axes[r, c]
            # Background sample cloud for visual context
            samples = _sample_from(model, xrow.repeat(2000, 1))
            ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.15, color="gray")
            # Component ellipses
            for k in range(len(pi)):
                e = Ellipse(
                    xy=(mu[k, 0], mu[k, 1]),
                    width=2 * sigma[k, 0],
                    height=2 * sigma[k, 1],
                    fill=False,
                    edgecolor="C3",
                    lw=1 + 3 * pi[k],
                )
                ax.add_patch(e)
                ax.plot(mu[k, 0], mu[k, 1], "rx", ms=5 + 12 * pi[k])
            ax.axvline(0, color="k", lw=0.5)
            ax.axvline(1, color="k", lw=0.5)
            ax.axhline(0, color="k", lw=0.5)
            ax.axhline(1, color="k", lw=0.5)
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(-0.3, 1.3)
            ax.set_xlabel(r"$\eta'_{tr}$")
            ax.set_ylabel(r"$\eta'_{rot,A}$")
            ax.set_title(f"{label}\n{rlabel}")
            ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Panel 2 — Mixture components (line width ∝ π; rays outside [0,1] = clipping)"
    )
    fig.tight_layout()
    return fig


def plot_conditional_scatter(
    models: dict[str, object],
    X: torch.Tensor,
    raw: np.ndarray,
    y_ctc: np.ndarray,
    n_bins: int = 3,
    max_points: int = 3000,
):
    """Panel 3: scatter of (eta_tr', eta_rot_A') for CTC vs each model, per E bin."""
    E_total = raw[:, 0:3].sum(axis=1)
    bin_idx, edges = _bin_indices(E_total, n_bins)
    cols = ["CTC", *models.keys()]
    fig, axes = plt.subplots(n_bins, len(cols), figsize=(4 * len(cols), 4 * n_bins))
    axes = np.atleast_2d(axes)

    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        sel = np.where(mask)[0]
        if len(sel) > max_points:
            sel = np.random.default_rng(0).choice(sel, max_points, replace=False)

        # CTC truth
        ax = axes[b, 0]
        ax.scatter(y_ctc[sel, 0], y_ctc[sel, 1], s=3, alpha=0.3)
        ax.set_title(f"CTC | E∈[{edges[b]:.0f}, {edges[b + 1]:.0f}] K")

        # Model samples
        for c, (label, model) in enumerate(models.items(), start=1):
            samples = _sample_from(model, X[sel])
            ax = axes[b, c]
            ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.3)
            ax.set_title(f"{label} | E∈[{edges[b]:.0f}, {edges[b + 1]:.0f}] K")

        for ax in axes[b]:
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel(r"$\eta'_{tr}$")
            ax.set_ylabel(r"$\eta'_{rot,A}$")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Panel 3 — Post-collision distribution per E_total bin")
    fig.tight_layout()
    return fig


def plot_kl_vs_energy(
    models: dict[str, object],
    X: torch.Tensor,
    raw: np.ndarray,
    y_ctc: np.ndarray,
    n_bins: int = 8,
):
    """Panel 4: KL(CTC || model_samples) per E_total bin, for each output dimension."""
    E_total = raw[:, 0:3].sum(axis=1)
    bin_idx, edges = _bin_indices(E_total, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, model in models.items():
        kl_tr = np.full(n_bins, np.nan)
        kl_rot = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() < 200:
                continue
            samples = _sample_from(model, X[mask])
            # Clip to mimic what DSMC actually feeds back into the simulation
            samples_clipped = np.clip(samples, 0.0, 1.0)
            try:
                kl_tr[b] = kl_divergence(y_ctc[mask, 0], samples_clipped[:, 0])
                kl_rot[b] = kl_divergence(y_ctc[mask, 1], samples_clipped[:, 1])
            except Exception as ex:
                print(f"  KL failed in bin {b} for {label}: {ex}")
        axes[0].plot(centers, kl_tr, marker="o", label=label)
        axes[1].plot(centers, kl_rot, marker="o", label=label)

    for ax, title in zip(axes, [r"KL on $\eta'_{tr}$", r"KL on $\eta'_{rot,A}$"]):
        ax.set_title(title)
        ax.set_xlabel(r"$E_{total}$ [K]")
        ax.set_ylabel("KL(CTC ‖ model)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Panel 4 — KL divergence vs energy (gap at low E is the smoking gun)")
    fig.tight_layout()
    return fig


def _find_fixed_point(centers: np.ndarray, response: np.ndarray) -> float:
    """Locate the first y=x crossing of a (binned) response curve.

    Uses linear interpolation between the two consecutive bin centres where
    (response - identity) changes sign. Returns NaN if no crossing exists in
    the binned range.
    """
    valid = ~np.isnan(response)
    if valid.sum() < 2:
        return float("nan")
    x = centers[valid]
    diff = response[valid] - x
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return float("nan")
    i = sign_changes[0]
    # Linear interp: diff is zero somewhere between x[i] and x[i+1].
    return float(x[i] - diff[i] * (x[i + 1] - x[i]) / (diff[i + 1] - diff[i]))


def plot_response_curve(
    models: dict[str, object],
    X: torch.Tensor,
    raw: np.ndarray,
    y_ctc: np.ndarray,
    n_bins: int = 25,
    samples_per_input: int = 8,
):
    """Panel 5: mean response E[eta_tr' | eta_tr_in] and E[eta_rot' | eta_rot_in].

    Bins inputs by eta_tr_in (and eta_rot_A_in), draws several samples per
    input from each model, plots the bin-mean of the post-collision fraction
    against the bin centre. CTC's empirical curve uses the dataset's actual
    post-collision fractions.

    The intersection with the y=x identity line is the kernel's fixed point —
    that's where iterating it drives the system. If the bad model's fixed
    point sits at a different x than CTC's, you've found the cause of the
    DSMC failure.
    """
    eta_tr_in = raw[:, 0] / raw[:, 0:3].sum(axis=1)
    eta_rot_A_in = raw[:, 1] / raw[:, 1:3].sum(axis=1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _bin_mean(x_in, y_out):
        bin_idx = np.digitize(x_in, edges[1:-1])
        means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 20:
                means[b] = float(np.mean(y_out[mask]))
        return means

    def _model_samples(model, n_draws):
        """Draw n_draws samples per input row; returns (N * n_draws, 2)."""
        chunks = []
        for _ in range(n_draws):
            chunks.append(_sample_from(model, X))
        return np.clip(np.concatenate(chunks, axis=0), 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fixed_points: dict[str, dict[str, float]] = {}

    def _annotate_fp(ax, fp: float, color: str, name: str):
        if np.isnan(fp):
            return
        ax.axvline(fp, color=color, ls=":", lw=1, alpha=0.7)
        ax.plot(fp, fp, "o", color=color, ms=8, mfc="none", mew=2)
        ax.annotate(
            f"{name}: {fp:.4f}",
            xy=(fp, fp),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

    # --- Translational response curve ---
    ctc_tr = _bin_mean(eta_tr_in, y_ctc[:, 0])
    fp_ctc_tr = _find_fixed_point(centers, ctc_tr)
    fixed_points["CTC"] = {"eta_tr": fp_ctc_tr}
    axes[0].plot(centers, ctc_tr, "k-", lw=2, label="CTC")
    _annotate_fp(axes[0], fp_ctc_tr, "black", "CTC")

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (label, model), color in zip(models.items(), palette):
        s = _model_samples(model, samples_per_input)
        x_tile = np.tile(eta_tr_in, samples_per_input)
        m = _bin_mean(x_tile, s[:, 0])
        fp = _find_fixed_point(centers, m)
        fixed_points[label] = {"eta_tr": fp}
        axes[0].plot(centers, m, marker="o", color=color, label=label)
        _annotate_fp(axes[0], fp, color, label)

    axes[0].plot([0, 1], [0, 1], "r--", lw=1, label="identity ($y=x$)")
    axes[0].set_xlabel(r"$\eta_{tr}$ (pre-collision)")
    axes[0].set_ylabel(r"$E[\eta'_{tr} \mid \eta_{tr}]$ (post-collision)")
    axes[0].set_title(r"Translational response — fixed point at $y=x$ intersection")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # --- Rotational A response curve ---
    ctc_rot = _bin_mean(eta_rot_A_in, y_ctc[:, 1])
    fp_ctc_rot = _find_fixed_point(centers, ctc_rot)
    fixed_points["CTC"]["eta_rot_A"] = fp_ctc_rot
    axes[1].plot(centers, ctc_rot, "k-", lw=2, label="CTC")
    _annotate_fp(axes[1], fp_ctc_rot, "black", "CTC")

    for (label, model), color in zip(models.items(), palette):
        s = _model_samples(model, samples_per_input)
        x_tile = np.tile(eta_rot_A_in, samples_per_input)
        m = _bin_mean(x_tile, s[:, 1])
        fp = _find_fixed_point(centers, m)
        fixed_points[label]["eta_rot_A"] = fp
        axes[1].plot(centers, m, marker="o", color=color, label=label)
        _annotate_fp(axes[1], fp, color, label)

    axes[1].plot([0, 1], [0, 1], "r--", lw=1, label="identity ($y=x$)")
    axes[1].set_xlabel(r"$\eta_{rot,A}$ (pre-collision)")
    axes[1].set_ylabel(r"$E[\eta'_{rot,A} \mid \eta_{rot,A}]$ (post-collision)")
    axes[1].set_title(r"Rotational A response — fixed point at $y=x$ intersection")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    fig.suptitle("Panel 5 — Mean response curves (fixed point = where DSMC settles)")
    fig.tight_layout()

    # --- Print numerical summary ---
    print("\nFixed points (where E[η'] = η, i.e. where the kernel iterates to):")
    print(f"  {'source':<25} {'eta_tr*':>10} {'eta_rot_A*':>14}")
    for src, vals in fixed_points.items():
        tr = vals.get("eta_tr", float("nan"))
        rot = vals.get("eta_rot_A", float("nan"))
        print(f"  {src:<25} {tr:>10.4f} {rot:>14.4f}")
    # Drifts from CTC
    print(
        "\nDrift from CTC fixed point (positive = model fixed point higher than CTC):"
    )
    base_tr = fixed_points["CTC"]["eta_tr"]
    base_rot = fixed_points["CTC"]["eta_rot_A"]
    for src, vals in fixed_points.items():
        if src == "CTC":
            continue
        d_tr = vals["eta_tr"] - base_tr
        d_rot = vals["eta_rot_A"] - base_rot
        print(f"  {src:<25} Δeta_tr* = {d_tr:+.4f}    Δeta_rot_A* = {d_rot:+.4f}")

    return fig


def plot_response_spread(
    models: dict[str, object],
    X: torch.Tensor,
    raw: np.ndarray,
    y_ctc: np.ndarray,
    n_bins: int = 25,
    samples_per_input: int = 16,
):
    """Panel 6: conditional std σ[η' | η_in] and slope at fixed point.

    Tests the mode-collapse hypothesis: a kernel with mode collapse has the
    right mean response but a narrow conditional distribution, so iterating
    it barely moves the state per step and DSMC relaxation stalls.

    Returns the local slope dE[η']/dη_in at each model's fixed point — near 0
    is fast relaxation, near 1 is marginal stability, > 1 is unstable.
    """
    eta_tr_in = raw[:, 0] / raw[:, 0:3].sum(axis=1)
    eta_rot_A_in = raw[:, 1] / raw[:, 1:3].sum(axis=1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _bin_stats(x_in, y_out):
        bin_idx = np.digitize(x_in, edges[1:-1])
        means = np.full(n_bins, np.nan)
        stds = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 20:
                means[b] = float(np.mean(y_out[mask]))
                stds[b] = float(np.std(y_out[mask]))
        return means, stds

    def _model_samples(model, n_draws):
        chunks = []
        for _ in range(n_draws):
            chunks.append(_sample_from(model, X))
        return np.clip(np.concatenate(chunks, axis=0), 0.0, 1.0)

    def _slope_at(x_arr, y_arr, x0):
        """dy/dx at x = x0 via gradient on the binned curve."""
        if np.isnan(x0):
            return float("nan")
        valid = ~np.isnan(y_arr)
        xv = x_arr[valid]
        yv = y_arr[valid]
        if len(xv) < 3 or x0 < xv[0] or x0 > xv[-1]:
            return float("nan")
        return float(np.interp(x0, xv, np.gradient(yv, xv)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    slopes_tr: dict[str, float] = {}
    slopes_rot: dict[str, float] = {}

    # --- Translational ---
    ctc_tr_mean, ctc_tr_std = _bin_stats(eta_tr_in, y_ctc[:, 0])
    fp_ctc_tr = _find_fixed_point(centers, ctc_tr_mean)
    slopes_tr["CTC"] = _slope_at(centers, ctc_tr_mean, fp_ctc_tr)
    axes[0].plot(centers, ctc_tr_std, "k-", lw=2, label="CTC")
    for (label, model), color in zip(models.items(), palette):
        s = _model_samples(model, samples_per_input)
        x_tile = np.tile(eta_tr_in, samples_per_input)
        m, sd = _bin_stats(x_tile, s[:, 0])
        fp = _find_fixed_point(centers, m)
        slopes_tr[label] = _slope_at(centers, m, fp)
        axes[0].plot(centers, sd, marker="o", color=color, label=label)
    axes[0].set_xlabel(r"$\eta_{tr}$ (pre-collision)")
    axes[0].set_ylabel(r"$\sigma[\eta'_{tr} \mid \eta_{tr}]$")
    axes[0].set_title("Translational conditional spread")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)

    # --- Rotational A ---
    ctc_rot_mean, ctc_rot_std = _bin_stats(eta_rot_A_in, y_ctc[:, 1])
    fp_ctc_rot = _find_fixed_point(centers, ctc_rot_mean)
    slopes_rot["CTC"] = _slope_at(centers, ctc_rot_mean, fp_ctc_rot)
    axes[1].plot(centers, ctc_rot_std, "k-", lw=2, label="CTC")
    for (label, model), color in zip(models.items(), palette):
        s = _model_samples(model, samples_per_input)
        x_tile = np.tile(eta_rot_A_in, samples_per_input)
        m, sd = _bin_stats(x_tile, s[:, 1])
        fp = _find_fixed_point(centers, m)
        slopes_rot[label] = _slope_at(centers, m, fp)
        axes[1].plot(centers, sd, marker="o", color=color, label=label)
    axes[1].set_xlabel(r"$\eta_{rot,A}$ (pre-collision)")
    axes[1].set_ylabel(r"$\sigma[\eta'_{rot,A} \mid \eta_{rot,A}]$")
    axes[1].set_title("Rotational A conditional spread")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)

    fig.suptitle(
        "Panel 6 — Conditional spread (mode collapse → tiny σ → slow relaxation)"
    )
    fig.tight_layout()

    # --- Print slope summary ---
    print(
        "\nResponse-curve slope at fixed point (near 0 = fast relaxation, near 1 = marginal):"
    )
    print(f"  {'source':<25} {'slope_tr':>10} {'slope_rot':>12}")
    for src in slopes_tr:
        print(f"  {src:<25} {slopes_tr[src]:>10.3f} {slopes_rot[src]:>12.3f}")

    return fig


def plot_iterative_drift(
    models: dict[str, object],
    *,
    species: Species | None = None,
    T_trans_init: float = 300.0,
    T_rot_init: float = 100.0,
    n_particles: int = 4000,
    n_iters: int = 200,
    randomseed: int = 0,
):
    """Panel 7: iterative kernel application from configurable initial state.

    Initialize a particle pool: translational velocities Maxwell-Boltzmann at
    `T_trans_init`, rotational energies exponential with mean k * T_rot_init.
    Each iteration pairs particles randomly and applies the kernel's
    batch_collide. Tracks <η_tr> and <η_rot_A> across the pair population.

    Setting T_trans_init = T_rot_init = T_eq tests whether the kernel
    preserves thermal equilibrium (drift = detailed-balance violation).
    Setting them to the energy-relaxation experiment's initial conditions
    (e.g. 300 K / 100 K) tests whether the kernel actually relaxes the system
    toward equipartition — which is the DSMC failure mode under inspection.

    For diatomic pairs (3 trans + 2 rot DOF each), the eventual equilibrium
    at fixed total energy lies at T_eq = (3 T_trans + 2 T_rot) / 5 and
    <η_tr> = 3/7 ≈ 0.4286, <η_rot_A> = 0.5.
    """
    sim_species: Species = species if species is not None else Species.H2()

    rng = np.random.default_rng(randomseed)
    kB = 1.380649e-23
    m = sim_species.mass
    zrot = sim_species.zrot_mdn

    T_eq = (3.0 * T_trans_init + 2.0 * T_rot_init) / 5.0

    # --- Initial conditions matching DSMC's create_particles convention ---
    sigma_v = float(np.sqrt(kB * T_trans_init / m))
    v0 = rng.normal(0.0, sigma_v, size=(n_particles, 3)).astype(np.float32)
    er0 = rng.exponential(scale=kB * T_rot_init, size=n_particles).astype(np.float32)

    n_pairs = n_particles // 2

    def _pair_marginals(velocities, e_rots):
        perm = rng.permutation(n_particles)
        i = perm[:n_pairs]
        j = perm[n_pairs : 2 * n_pairs]
        g = velocities[i] - velocities[j]
        E_rel = 0.25 * m * np.sum(g**2, axis=1)
        E_tot = np.maximum(E_rel + e_rots[i] + e_rots[j], 1e-30)
        Erot_pair = np.maximum(e_rots[i] + e_rots[j], 1e-30)
        return E_rel / E_tot, e_rots[i] / Erot_pair

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # Equipartition reference lines (the eventual target if the kernel
    # equilibrates the system at the energy-conserving T_eq).
    eta_tr_eq = 3.0 / 7.0
    eta_rot_eq = 0.5
    axes[0, 0].axhline(
        T_eq, color="black", lw=1, ls="--", label=f"$T_{{eq}}$ ({T_eq:.0f} K)"
    )
    axes[0, 1].axhline(
        eta_tr_eq,
        color="black",
        lw=1,
        ls="--",
        label=rf"equipartition ({eta_tr_eq:.4f})",
    )

    final_summary: dict[str, dict[str, float]] = {}

    def _temps(velocities, e_rots):
        T_trans = float(np.mean(0.5 * m * np.sum(velocities**2, axis=1)) / (1.5 * kB))
        T_rot = float(np.mean(e_rots) / kB)
        return T_trans, T_rot

    for (label, model), color in zip(models.items(), palette):
        velocities = v0.copy()
        e_rots = er0.copy()

        T_trans_hist = [_temps(velocities, e_rots)[0]]
        T_rot_hist = [_temps(velocities, e_rots)[1]]
        eta_tr_0, eta_rot_0 = _pair_marginals(velocities, e_rots)
        mean_eta_tr = [float(eta_tr_0.mean())]
        mean_eta_rot = [float(eta_rot_0.mean())]

        for _ in range(n_iters):
            perm = rng.permutation(n_particles)
            i = perm[:n_pairs]
            j = perm[n_pairs : 2 * n_pairs]
            vi, eri, vj, erj = model.batch_collide(  # type: ignore[attr-defined]
                velocities[i],
                e_rots[i],
                velocities[j],
                e_rots[j],
                m=m,
                zrot=zrot,
            )
            velocities[i] = vi
            velocities[j] = vj
            e_rots[i] = eri
            e_rots[j] = erj

            T_t, T_r = _temps(velocities, e_rots)
            T_trans_hist.append(T_t)
            T_rot_hist.append(T_r)
            eta_tr_step, eta_rot_step = _pair_marginals(velocities, e_rots)
            mean_eta_tr.append(float(eta_tr_step.mean()))
            mean_eta_rot.append(float(eta_rot_step.mean()))

        iters = np.arange(len(T_trans_hist))
        axes[0, 0].plot(
            iters, T_trans_hist, color=color, ls="-", label=rf"$T_{{trans}}$ {label}"
        )
        axes[0, 0].plot(
            iters, T_rot_hist, color=color, ls=":", label=rf"$T_{{rot}}$ {label}"
        )
        axes[0, 1].plot(
            iters, mean_eta_tr, color=color, ls="-", label=rf"$\eta_{{tr}}$ {label}"
        )
        axes[0, 1].plot(
            iters, mean_eta_rot, color=color, ls=":", label=rf"$\eta_{{rot,A}}$ {label}"
        )

        # Final histograms
        final_eta_tr, final_eta_rot = _pair_marginals(velocities, e_rots)
        axes[1, 0].hist(
            final_eta_tr,
            bins=40,
            range=(0, 1),
            density=True,
            alpha=0.4,
            color=color,
            label=label,
        )
        axes[1, 1].hist(
            final_eta_rot,
            bins=40,
            range=(0, 1),
            density=True,
            alpha=0.4,
            color=color,
            label=label,
        )

        final_summary[label] = {
            "T_trans_final": T_trans_hist[-1],
            "T_rot_final": T_rot_hist[-1],
            "eta_tr_final": mean_eta_tr[-1],
            "eta_rot_A_final": mean_eta_rot[-1],
        }

    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("Temperature [K]")
    axes[0, 0].set_title(
        f"$T_{{trans}}$ (solid) and $T_{{rot}}$ (dotted) vs iteration "
        f"— initial ({T_trans_init:.0f}, {T_rot_init:.0f}) K"
    )
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].set_title(
        r"$\langle \eta_{tr} \rangle$ (solid), $\langle \eta_{rot,A} \rangle$ (dotted)"
    )
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].set_xlabel(r"$\eta_{tr}$")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].set_title(r"Final $\eta_{tr}$ marginal")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_xlabel(r"$\eta_{rot,A}$")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title(r"Final $\eta_{rot,A}$ marginal")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(
        f"Panel 7 — Iterative kernel from $T_{{trans}}$={T_trans_init:.0f} K, "
        f"$T_{{rot}}$={T_rot_init:.0f} K → $T_{{eq}}$≈{T_eq:.0f} K "
        f"({n_iters} iters, N={n_particles}, zrot={zrot:.2f})"
    )
    fig.tight_layout()

    print(
        f"\nIterative kernel from T_trans={T_trans_init:.0f} K, T_rot={T_rot_init:.0f} K"
        f" (energy-conserving T_eq ≈ {T_eq:.1f} K):"
    )
    print(
        f"  Equipartition target: <eta_tr>={eta_tr_eq:.4f}, <eta_rot_A>={eta_rot_eq:.4f}, "
        f"T_trans=T_rot={T_eq:.1f} K"
    )
    print(
        f"  {'source':<25} {'T_trans':>10} {'T_rot':>10} {'<eta_tr>':>10} {'<eta_rot_A>':>12}"
    )
    for label, vals in final_summary.items():
        print(
            f"  {label:<25} {vals['T_trans_final']:>10.2f} {vals['T_rot_final']:>10.2f} "
            f"{vals['eta_tr_final']:>10.4f} {vals['eta_rot_A_final']:>12.4f}"
        )

    return fig


def plot_iterative_drift_ntc(
    models: dict[str, object],
    *,
    species: Species | None = None,
    T_trans_init: float = 300.0,
    T_rot_init: float = 100.0,
    n_particles: int = 4000,
    n_iters: int = 200,
    randomseed: int = 0,
):
    """Panel 8: iterative kernel with NTC-biased pair acceptance.

    Identical scaffolding to panel 7, but pairs are accepted with probability
    |v_i − v_j| / v_rmax (rejection sampling) instead of uniformly. This
    over-samples high-relative-velocity pairs the same way real DSMC does —
    so the distribution of pairs the kernel sees matches the NTC-biased
    distribution at inference time.

    Together with panel 7 this isolates the hypothesis: if wf3 stabilises in
    panel 7 (uniform pairing) but diverges or stabilises further from
    equipartition in panel 8 (NTC pairing), the failure mode is the
    training/inference distribution mismatch caused by NTC selection.
    """
    sim_species: Species = species if species is not None else Species.H2()

    rng = np.random.default_rng(randomseed)
    kB = 1.380649e-23
    m = sim_species.mass
    zrot = sim_species.zrot_mdn

    T_eq = (3.0 * T_trans_init + 2.0 * T_rot_init) / 5.0

    sigma_v = float(np.sqrt(kB * T_trans_init / m))
    v0 = rng.normal(0.0, sigma_v, size=(n_particles, 3)).astype(np.float32)
    er0 = rng.exponential(scale=kB * T_rot_init, size=n_particles).astype(np.float32)

    n_pairs = n_particles // 2

    def _temps(velocities, e_rots):
        T_trans = float(np.mean(0.5 * m * np.sum(velocities**2, axis=1)) / (1.5 * kB))
        T_rot = float(np.mean(e_rots) / kB)
        return T_trans, T_rot

    def _pair_marginals(velocities, e_rots):
        """Marginal η_tr / η_rot_A across a uniform random pairing of the pool."""
        perm = rng.permutation(n_particles)
        i = perm[:n_pairs]
        j = perm[n_pairs : 2 * n_pairs]
        g = velocities[i] - velocities[j]
        E_rel = 0.25 * m * np.sum(g**2, axis=1)
        E_tot = np.maximum(E_rel + e_rots[i] + e_rots[j], 1e-30)
        Erot_pair = np.maximum(e_rots[i] + e_rots[j], 1e-30)
        return E_rel / E_tot, e_rots[i] / Erot_pair

    def _ntc_accept(velocities, vrmax: float):
        """Pick n_pairs random candidates, accept each with prob |g| / v_rmax."""
        perm = rng.permutation(n_particles)
        i = perm[:n_pairs]
        j = perm[n_pairs : 2 * n_pairs]
        g_speed = np.linalg.norm(velocities[i] - velocities[j], axis=1)
        accept = rng.random(n_pairs) < (g_speed / max(vrmax, 1e-30))
        return i[accept], j[accept]

    def _estimate_vrmax(velocities) -> float:
        """Estimate v_rmax from a sample of random pairs."""
        sample = min(2000, n_pairs)
        perm = rng.permutation(n_particles)
        si = perm[:sample]
        sj = perm[sample : 2 * sample]
        g_speed = np.linalg.norm(velocities[si] - velocities[sj], axis=1)
        return float(np.max(g_speed))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    eta_tr_eq = 3.0 / 7.0
    eta_rot_eq = 0.5
    axes[0, 0].axhline(
        T_eq, color="black", lw=1, ls="--", label=f"$T_{{eq}}$ ({T_eq:.0f} K)"
    )
    axes[0, 1].axhline(
        eta_tr_eq,
        color="black",
        lw=1,
        ls="--",
        label=rf"equipartition ({eta_tr_eq:.4f})",
    )

    final_summary: dict[str, dict[str, float]] = {}

    for (label, model), color in zip(models.items(), palette):
        velocities = v0.copy()
        e_rots = er0.copy()

        T_trans_hist, T_rot_hist, mean_eta_tr, mean_eta_rot = [], [], [], []
        accept_rates: list[float] = []

        T_t, T_r = _temps(velocities, e_rots)
        T_trans_hist.append(T_t)
        T_rot_hist.append(T_r)
        eta_tr_0, eta_rot_0 = _pair_marginals(velocities, e_rots)
        mean_eta_tr.append(float(eta_tr_0.mean()))
        mean_eta_rot.append(float(eta_rot_0.mean()))

        for _ in range(n_iters):
            vrmax = _estimate_vrmax(velocities)
            idx_i, idx_j = _ntc_accept(velocities, vrmax)
            accept_rates.append(len(idx_i) / n_pairs)

            if len(idx_i) > 0:
                vi, eri, vj, erj = model.batch_collide(  # type: ignore[attr-defined]
                    velocities[idx_i],
                    e_rots[idx_i],
                    velocities[idx_j],
                    e_rots[idx_j],
                    m=m,
                    zrot=zrot,
                )
                velocities[idx_i] = vi
                velocities[idx_j] = vj
                e_rots[idx_i] = eri
                e_rots[idx_j] = erj

            T_t, T_r = _temps(velocities, e_rots)
            T_trans_hist.append(T_t)
            T_rot_hist.append(T_r)
            eta_tr_s, eta_rot_s = _pair_marginals(velocities, e_rots)
            mean_eta_tr.append(float(eta_tr_s.mean()))
            mean_eta_rot.append(float(eta_rot_s.mean()))

        iters = np.arange(len(T_trans_hist))
        axes[0, 0].plot(
            iters, T_trans_hist, color=color, ls="-", label=rf"$T_{{trans}}$ {label}"
        )
        axes[0, 0].plot(
            iters, T_rot_hist, color=color, ls=":", label=rf"$T_{{rot}}$ {label}"
        )
        axes[0, 1].plot(
            iters, mean_eta_tr, color=color, ls="-", label=rf"$\eta_{{tr}}$ {label}"
        )
        axes[0, 1].plot(
            iters, mean_eta_rot, color=color, ls=":", label=rf"$\eta_{{rot,A}}$ {label}"
        )

        final_eta_tr, final_eta_rot = _pair_marginals(velocities, e_rots)
        axes[1, 0].hist(
            final_eta_tr,
            bins=40,
            range=(0, 1),
            density=True,
            alpha=0.4,
            color=color,
            label=label,
        )
        axes[1, 1].hist(
            final_eta_rot,
            bins=40,
            range=(0, 1),
            density=True,
            alpha=0.4,
            color=color,
            label=label,
        )

        final_summary[label] = {
            "T_trans_final": T_trans_hist[-1],
            "T_rot_final": T_rot_hist[-1],
            "eta_tr_final": mean_eta_tr[-1],
            "eta_rot_A_final": mean_eta_rot[-1],
            "mean_accept_rate": float(np.mean(accept_rates)) if accept_rates else 0.0,
        }

    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("Temperature [K]")
    axes[0, 0].set_title(
        rf"NTC-biased: $T_{{trans}}$ (solid), $T_{{rot}}$ (dotted) — "
        rf"initial ({T_trans_init:.0f}, {T_rot_init:.0f}) K"
    )
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].set_title(
        r"NTC-biased: $\langle\eta_{tr}\rangle$ (solid), $\langle\eta_{rot,A}\rangle$ (dotted)"
    )
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].set_xlabel(r"$\eta_{tr}$")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].set_title(r"Final $\eta_{tr}$ marginal (NTC-biased)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_xlabel(r"$\eta_{rot,A}$")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title(r"Final $\eta_{rot,A}$ marginal (NTC-biased)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(
        f"Panel 8 — NTC-biased iterative kernel "
        f"(initial $T_{{trans}}$={T_trans_init:.0f}, $T_{{rot}}$={T_rot_init:.0f} K, "
        f"$T_{{eq}}$≈{T_eq:.0f} K, {n_iters} iters, zrot={zrot:.2f})"
    )
    fig.tight_layout()

    print(
        f"\nNTC-biased iterative kernel from T_trans={T_trans_init:.0f}, "
        f"T_rot={T_rot_init:.0f} K (energy-conserving T_eq ≈ {T_eq:.1f} K):"
    )
    print(
        f"  Equipartition target: <eta_tr>={eta_tr_eq:.4f}, <eta_rot_A>={eta_rot_eq:.4f}, "
        f"T_trans=T_rot={T_eq:.1f} K"
    )
    print(
        f"  {'source':<25} {'T_trans':>10} {'T_rot':>10} {'<eta_tr>':>10} "
        f"{'<eta_rot_A>':>12} {'accept':>8}"
    )
    for label, vals in final_summary.items():
        print(
            f"  {label:<25} {vals['T_trans_final']:>10.2f} {vals['T_rot_final']:>10.2f} "
            f"{vals['eta_tr_final']:>10.4f} {vals['eta_rot_A_final']:>12.4f} "
            f"{vals['mean_accept_rate']:>8.3f}"
        )

    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_diagnostics(
    good_path: str | Path,
    bad_path: str | Path,
    dataset_path: str | Path,
    output_subdir: str = "diagnostics",
    good_label: str = "good (wf7)",
    bad_label: str = "bad (wf3)",
    T_trans_init: float = 300.0,
    T_rot_init: float = 100.0,
    n_iters: int = 200,
):
    """Load both models and the CTC dataset, run all four diagnostic panels."""
    print(f"Loading good model: {good_path}")
    good = load_mdn(good_path)
    print(f"Loading bad model:  {bad_path}")
    bad = load_mdn(bad_path)
    models = {good_label: good, bad_label: bad}

    print(f"Loading dataset:    {dataset_path}")
    raw = load_collision_dataset(dataset_path)
    X, y_t, _ = prepare_training_tensors(raw, wf=1.0)
    y_ctc = y_t.numpy()

    print("Running panel 1: clipping rate ...")
    fig1 = plot_clipping_rate(models, X, raw)
    fig1.savefig(
        paths.plot_path("panel1_clipping_rate.png", subdir=output_subdir), dpi=150
    )

    print("Running panel 2: mixture components ...")
    fig2 = plot_mixture_components(models, X, raw)
    fig2.savefig(
        paths.plot_path("panel2_mixture_components.png", subdir=output_subdir), dpi=150
    )

    print("Running panel 3: conditional scatter ...")
    fig3 = plot_conditional_scatter(models, X, raw, y_ctc)
    fig3.savefig(
        paths.plot_path("panel3_conditional_scatter.png", subdir=output_subdir), dpi=150
    )

    print("Running panel 4: KL divergence vs energy ...")
    fig4 = plot_kl_vs_energy(models, X, raw, y_ctc)
    fig4.savefig(
        paths.plot_path("panel4_kl_vs_energy.png", subdir=output_subdir), dpi=150
    )

    print("Running panel 5: response curve / fixed point ...")
    fig5 = plot_response_curve(models, X, raw, y_ctc)
    fig5.savefig(
        paths.plot_path("panel5_response_curve.png", subdir=output_subdir), dpi=150
    )

    print("Running panel 6: conditional spread / slope ...")
    fig6 = plot_response_spread(models, X, raw, y_ctc)
    fig6.savefig(
        paths.plot_path("panel6_response_spread.png", subdir=output_subdir), dpi=150
    )

    print(
        f"Running panel 7: iterative kernel from T_trans={T_trans_init}, T_rot={T_rot_init} ..."
    )
    fig7 = plot_iterative_drift(
        models,
        T_trans_init=T_trans_init,
        T_rot_init=T_rot_init,
        n_iters=n_iters,
    )
    fig7.savefig(
        paths.plot_path("panel7_iterative_drift.png", subdir=output_subdir), dpi=150
    )

    print(f"Running panel 8: NTC-biased iterative kernel ...")
    fig8 = plot_iterative_drift_ntc(
        models,
        T_trans_init=T_trans_init,
        T_rot_init=T_rot_init,
        n_iters=n_iters,
    )
    fig8.savefig(
        paths.plot_path("panel8_iterative_drift_ntc.png", subdir=output_subdir), dpi=150
    )

    print(f"\nFigures saved under: {paths.PLOTS_DIR / output_subdir}")
    return models


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--good",
        default="results/models/mdn/weightsensitivity/H2_200000_dataseed41/mdn_H2_wf7.pth",
    )
    p.add_argument(
        "--bad",
        default="results/models/mdn/weightsensitivity/H2_200000_dataseed41/mdn_H2_wf3.pth",
    )
    p.add_argument(
        "--dataset",
        default="data/H2H2_collisions_numba_b1_0_200000_seed41.npy",
    )
    p.add_argument("--output-subdir", default="diagnostics")
    p.add_argument("--show", action="store_true", help="Display plots interactively.")
    p.add_argument(
        "--trans-T-init",
        type=float,
        default=300.0,
        help="Initial translational temperature for panel 7's iterative test [K].",
    )
    p.add_argument(
        "--rot-T-init",
        type=float,
        default=100.0,
        help="Initial rotational temperature for panel 7 [K].",
    )
    p.add_argument(
        "--n-iters",
        type=int,
        default=200,
        help="Number of iterations of the kernel in panel 7.",
    )
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    run_diagnostics(
        args.good,
        args.bad,
        args.dataset,
        output_subdir=args.output_subdir,
        T_trans_init=args.trans_T_init,
        T_rot_init=args.rot_T_init,
        n_iters=args.n_iters,
    )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
