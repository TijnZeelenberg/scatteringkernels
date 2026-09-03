"""Kernel stationarity probe: net energy drift and output marginal at equilibrium.

Draws N collision states from the T_eq equilibrium distribution, applies each
kernel once, and asks: does the output marginal match the input? If not, the
kernel will drive the gas away from equipartition even when it is already there.

Two panels:
  1. Marginal distribution of η_tr' vs the equilibrium target Beta(3/2, 2).
     BL is stationary by construction; MDN output shifted toward lower η_tr'
     (more rotation) explains the persistent T_rot > T_trans bias.

  2. Conditional mean drift E[η_tr' − η_tr | η_tr] binned by η_tr.
     The zero-crossing predicts the kernel's fixed point. BL crosses at 3/7
     (correct equipartition). MDN crosses at η_tr* < 3/7, meaning its
     stationary distribution has excess rotational energy.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import paths
from config.plotting_config import PlottingConfig
from experiments.energy_relaxation import load_mdn

T_EQ = 2200.0
N_SAMPLES = 200_000
N_BINS = 40
SEED = 42

# Models to probe: label -> path (without .pth suffix — paths.model_path adds it)
MDN_MODELS: dict[str, str] = {
    "MDN λ=10": str(paths.model_path("mdn", "mdn_H2_Etr20k_Erot15k_Teq2200_db10")),
    "MDN λ=50": str(paths.model_path("mdn", "mdn_H2_Etr20k_Erot15k_Teq2200_db50")),
    "MDN λ=100": str(paths.model_path("mdn", "mdn_H2_Etr20k_Erot15k_Teq2200_db100")),
}

# Equilibrium η_tr distribution: Beta(n_trans/2, n_rot/2) = Beta(3/2, 2)
# for 3 translational + 4 rotational DOF in an H2-H2 binary collision.
_EQ_ALPHA = 1.5
_EQ_BETA = 2.0
_EQ_MEAN = _EQ_ALPHA / (_EQ_ALPHA + _EQ_BETA)  # = 3/7 ≈ 0.4286


def _sample_equilibrium(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw n states from the T_eq=2200K equilibrium distribution.

    E_total is returned in K (E/kB) to match how training data is built — the
    model's input_mean/std are K-scale; feeding Joules saturates the feature.
    """
    E_total = rng.gamma(3.5, T_EQ, size=n)  # Gamma(7/2, T), in K
    eta_tr = rng.beta(_EQ_ALPHA, _EQ_BETA, size=n)  # Beta(3/2, 2)
    eta_rot_A = rng.uniform(0.0, 1.0, size=n)  # Uniform — symmetric between particles
    return E_total, eta_tr, eta_rot_A


def _apply_mdn(
    model, E_total: np.ndarray, eta_tr: np.ndarray, eta_rot_A: np.ndarray
) -> np.ndarray:
    """Return η_tr' samples from the MDN for each equilibrium input state."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = torch.tensor(
        np.stack([E_total, eta_tr, eta_rot_A], axis=1), device=device, dtype=dtype
    )
    chunks = []
    chunk_size = 10_000
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), chunk_size):
            out = model.sample(inputs[i : i + chunk_size]).cpu().numpy()
            chunks.append(np.clip(out[:, 0], 0.0, 1.0))
    return np.concatenate(chunks)


def _apply_bl(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return η_tr' samples from the BL kernel (unconditional Beta(3/2, 2))."""
    return rng.beta(_EQ_ALPHA, _EQ_BETA, size=n)


def _binned_conditional_mean(
    x: np.ndarray, delta: np.ndarray, bins: np.ndarray, min_count: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `delta` in each bin of `x`. Drops bins with fewer than min_count samples."""
    idx = np.digitize(x, bins) - 1
    centers, means = [], []
    for i in range(len(bins) - 1):
        mask = idx == i
        if mask.sum() >= min_count:
            centers.append(0.5 * (bins[i] + bins[i + 1]))
            means.append(delta[mask].mean())
    return np.array(centers), np.array(means)


def main(output_path: str | None = None):
    rng = np.random.default_rng(SEED)
    pc = PlottingConfig()

    print(f"Sampling {N_SAMPLES:,} equilibrium states at T={T_EQ} K …")
    E_total, eta_tr, eta_rot_A = _sample_equilibrium(N_SAMPLES, rng)

    print("Applying BL kernel …")
    eta_tr_bl = _apply_bl(N_SAMPLES, rng)

    mdn_outputs: dict[str, np.ndarray] = {}
    for label, model_path in MDN_MODELS.items():
        print(f"Applying {label} …")
        model = load_mdn(model_path)
        mdn_outputs[label] = _apply_mdn(model, E_total, eta_tr, eta_rot_A)

    # --- Figure ---
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    x_grid = np.linspace(0.0, 1.0, 400)
    eq_pdf = stats.beta(_EQ_ALPHA, _EQ_BETA).pdf(x_grid)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mdn_colors = colors[: len(mdn_outputs)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ------------------------------------------------------------------
    # Panel 1: output marginal distributions
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(
        x_grid,
        eq_pdf,
        "k-",
        lw=2.0,
        label=r"Equilibrium target: Beta(3/2, 2)",
        zorder=6,
    )
    ax.hist(
        eta_tr,
        bins=bins,
        density=True,
        alpha=0.25,
        color="gray",
        label=r"Input $\eta_{tr}$ (equilibrium draw)",
    )
    ax.hist(
        eta_tr_bl,
        bins=bins,
        density=True,
        alpha=0.55,
        color="green",
        label=r"BL output $\eta_{tr}'$",
    )
    for (label, eta_tr_post), color in zip(mdn_outputs.items(), mdn_colors):
        ax.hist(
            eta_tr_post,
            bins=bins,
            density=True,
            alpha=0.45,
            color=color,
            label=rf"{label} output $\eta_{{tr}}'$",
        )

    ax.set_xlabel(
        r"$\eta_{tr}$", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight
    )
    ax.set_ylabel("Density", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight)
    ax.set_title(
        r"Output marginal $p(\eta_{tr}')$ at equilibrium inputs",
        fontsize=pc.label_fontsize,
    )
    ax.legend(fontsize=pc.legend_fontsize)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2: conditional mean drift E[η_tr' − η_tr | η_tr]
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.axhline(0.0, color="k", lw=1.0, ls="--", label="Zero drift (stationary)")
    ax.axvline(
        _EQ_MEAN,
        color="k",
        lw=1.0,
        ls=":",
        alpha=0.5,
        label=r"Equipartition $\eta_{tr}^* = 3/7$",
    )

    centers_bl, drift_bl = _binned_conditional_mean(eta_tr, eta_tr_bl - eta_tr, bins)
    ax.plot(centers_bl, drift_bl, "g-", lw=2.0, label="BL")

    for (label, eta_tr_post), color in zip(mdn_outputs.items(), mdn_colors):
        centers, drift = _binned_conditional_mean(eta_tr, eta_tr_post - eta_tr, bins)
        ax.plot(centers, drift, "-", color=color, lw=2.0, label=label)

    ax.set_xlabel(
        r"Input $\eta_{tr}$", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight
    )
    ax.set_ylabel(
        r"$\mathbb{E}[\eta_{tr}' - \eta_{tr} \mid \eta_{tr}]$",
        fontsize=pc.label_fontsize,
        fontweight=pc.label_fontweight,
    )
    ax.set_title(
        "Conditional mean energy drift at equilibrium inputs",
        fontsize=pc.label_fontsize,
    )
    ax.legend(fontsize=pc.legend_fontsize)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out = output_path or paths.plot_path("kernel_stationarity.png")
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
