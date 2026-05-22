"""Compare BL vs MDN DSMC collision logs to localize the kernel bias.

Loads two `.npz` archives written by `physics.collision_logger.CollisionLogger`
during the H2 relaxation experiment and overlays per-step aggregates plus
distributional probes at the final snapshot.

Eight diagnostic panels:

  Per-step traces (BL vs MDN):
    1. n_collisions per step                — sanity: collision rates should
                                              be similar.
    2. ⟨ΔE_trans⟩ per step (K/coll)         — net energy direction. BL → 0 at
                                              equilibrium; persistent MDN
                                              offset = the wrong fixed point.
    3. max |ΔE_total| per step (K)          — conservation. Both ≈ 0; if MDN
                                              spikes, the model leaks energy.
    4. fraction OOD                         — does MDN see more inputs outside
                                              the training support?

  Final-snapshot distributions (snapshot_every=100 → last snapshot ≈ step 900):
    5. η_trans_pre PDF + Beta(3/2,2)        — DSMC state. If MDN's pre-state
                                              already differs, the imbalance
                                              is upstream of the kernel call.
    6. η_trans_post PDF + Beta(3/2,2)       — kernel output marginal.
    7. ⟨η_trans_post | η_trans_pre⟩ + y=x   — the fixed-point map. The
                                              y=x crossing locates the
                                              stationary η_trans of the
                                              kernel. BL crosses at 3/7;
                                              an MDN crossing < 3/7 means
                                              its equilibrium has T_rot >
                                              T_trans (matches the
                                              equilibration bias).
    8. ⟨ΔE_trans | η_trans_pre⟩ (K)         — drift map. Sign and zero
                                              crossing should match panel 7.

Defaults read:
  results/logs/H2_energy_relaxation_BL.npz
  results/logs/H2_energy_relaxation_mdn_H2_wf1.npz
and write:
  results/plots/compare_BL_vs_mdn_H2_wf1.png

Run from the project root:
    python analysis/compare_collision_logs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import paths
from physics.species import Species

KB = 1.380649e-23
MASS_H2 = Species.H2().mass
ETA_EQ = 3.0 / 7.0  # Beta(3/2, 2) mean — equipartition fixed point


def load_log(path: str | Path) -> dict:
    return dict(np.load(path))


def _snapshot_slice(log: dict, k: int) -> slice:
    offs = log["snapshot_offsets"]
    return slice(int(offs[k]), int(offs[k + 1]))


def collision_quantities(log: dict, k: int, mass: float) -> dict:
    """Pre/post energies and η_trans for snapshot k."""
    s = _snapshot_slice(log, k)
    v_i_pre = log["snapshot_v_i_pre"][s]
    v_j_pre = log["snapshot_v_j_pre"][s]
    v_i_post = log["snapshot_v_i_post"][s]
    v_j_post = log["snapshot_v_j_post"][s]
    e_rot_i_pre = log["snapshot_e_rot_i_pre"][s]
    e_rot_j_pre = log["snapshot_e_rot_j_pre"][s]
    e_rot_i_post = log["snapshot_e_rot_i_post"][s]
    e_rot_j_post = log["snapshot_e_rot_j_post"][s]

    mu = 0.5 * mass
    v_rel_pre = v_i_pre - v_j_pre
    v_rel_post = v_i_post - v_j_post
    E_rel_pre = 0.5 * mu * np.sum(v_rel_pre * v_rel_pre, axis=1)
    E_rel_post = 0.5 * mu * np.sum(v_rel_post * v_rel_post, axis=1)

    e_rot_pair_pre = e_rot_i_pre + e_rot_j_pre
    e_rot_pair_post = e_rot_i_post + e_rot_j_post
    E_total_pre = E_rel_pre + e_rot_pair_pre
    E_total_post = E_rel_post + e_rot_pair_post

    safe_pre = np.where(E_total_pre > 0, E_total_pre, 1.0)
    safe_post = np.where(E_total_post > 0, E_total_post, 1.0)
    eta_trans_pre = E_rel_pre / safe_pre
    eta_trans_post = E_rel_post / safe_post

    return {
        "E_rel_pre": E_rel_pre,
        "E_rel_post": E_rel_post,
        "E_total_pre": E_total_pre,
        "E_total_post": E_total_post,
        "eta_trans_pre": eta_trans_pre,
        "eta_trans_post": eta_trans_post,
        "delta_E_trans": E_rel_post - E_rel_pre,
        "residual": E_total_post - E_total_pre,
    }


def conditional_mean(
    x: np.ndarray, y: np.ndarray, bins: int = 20, x_range=(0.0, 1.0), min_count: int = 5
):
    edges = np.linspace(x_range[0], x_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] >= min_count:
            means[i] = float(np.mean(y[m]))
    return centers, means, counts


def main(
    bl_path: str | Path,
    mdn_path: str | Path,
    output_path: str | Path,
    mass: float = MASS_H2,
):
    bl_path = Path(bl_path)
    mdn_path = Path(mdn_path)
    for p in (bl_path, mdn_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing log: {p}\nRun experiments/H2_energy_relaxation.py first."
            )

    bl = load_log(bl_path)
    mdn = load_log(mdn_path)

    if "snapshot_steps" not in bl or "snapshot_steps" not in mdn:
        raise RuntimeError(
            "Logs contain no snapshots. Re-run the experiment with "
            "CollisionLogger(snapshot_every=...) set."
        )

    k_bl = len(bl["snapshot_steps"]) - 1
    k_mdn = len(mdn["snapshot_steps"]) - 1
    qb = collision_quantities(bl, k_bl, mass)
    qm = collision_quantities(mdn, k_mdn, mass)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # ---- Row 1: per-step traces ----
    ax = axes[0, 0]
    ax.plot(bl["step"], bl["n_collisions"], label="BL", lw=1)
    ax.plot(mdn["step"], mdn["n_collisions"], label="MDN", lw=1)
    ax.set_title("Collisions per step")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(bl["step"], bl["delta_E_trans_mean"] / KB, label="BL", lw=1)
    ax.plot(mdn["step"], mdn["delta_E_trans_mean"] / KB, label="MDN", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("⟨ΔE_trans⟩ per collision (K)")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(bl["step"], np.abs(bl["energy_residual_max"]) / KB, label="BL", lw=1)
    ax.plot(mdn["step"], np.abs(mdn["energy_residual_max"]) / KB, label="MDN", lw=1)
    ax.set_title("max |ΔE_total| per step (K)")
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 3]
    ax.plot(bl["step"], bl["frac_ood"], label="BL", lw=1)
    ax.plot(mdn["step"], mdn["frac_ood"], label="MDN", lw=1)
    ax.set_title("Fraction OOD vs training caps")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- Row 2: distributional probes at the last snapshot ----
    beta = stats.beta(1.5, 2.0)
    eta_grid = np.linspace(1e-3, 1 - 1e-3, 200)

    ax = axes[1, 0]
    ax.hist(
        qb["eta_trans_pre"], bins=40, range=(0, 1), density=True,
        histtype="step", lw=1.4, label=f"BL  (n={len(qb['eta_trans_pre'])})",
    )
    ax.hist(
        qm["eta_trans_pre"], bins=40, range=(0, 1), density=True,
        histtype="step", lw=1.4, label=f"MDN (n={len(qm['eta_trans_pre'])})",
    )
    ax.plot(eta_grid, beta.pdf(eta_grid), "k--", lw=1, label="Beta(3/2,2)")
    ax.axvline(ETA_EQ, color="k", ls=":", lw=0.7)
    ax.set_title(f"η_trans_pre  (BL snap step {int(bl['snapshot_steps'][k_bl])}, "
                 f"MDN step {int(mdn['snapshot_steps'][k_mdn])})")
    ax.set_xlabel("η_trans_pre")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(
        qb["eta_trans_post"], bins=40, range=(0, 1), density=True,
        histtype="step", lw=1.4, label="BL",
    )
    ax.hist(
        qm["eta_trans_post"], bins=40, range=(0, 1), density=True,
        histtype="step", lw=1.4, label="MDN",
    )
    ax.plot(eta_grid, beta.pdf(eta_grid), "k--", lw=1, label="Beta(3/2,2)")
    ax.axvline(ETA_EQ, color="k", ls=":", lw=0.7)
    ax.set_title("η_trans_post (final snapshot)")
    ax.set_xlabel("η_trans_post")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    xb, yb, _ = conditional_mean(qb["eta_trans_pre"], qb["eta_trans_post"], bins=20)
    xm, ym, _ = conditional_mean(qm["eta_trans_pre"], qm["eta_trans_post"], bins=20)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="y=x")
    ax.plot(xb, yb, "o-", label="BL", ms=4)
    ax.plot(xm, ym, "o-", label="MDN", ms=4)
    ax.axvline(ETA_EQ, color="k", ls=":", lw=0.7)
    ax.axhline(ETA_EQ, color="k", ls=":", lw=0.7)
    ax.set_title("⟨η_trans_post | η_trans_pre⟩")
    ax.set_xlabel("η_trans_pre")
    ax.set_ylabel("⟨η_trans_post⟩")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 3]
    xb, yb, _ = conditional_mean(
        qb["eta_trans_pre"], qb["delta_E_trans"] / KB, bins=20
    )
    xm, ym, _ = conditional_mean(
        qm["eta_trans_pre"], qm["delta_E_trans"] / KB, bins=20
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(ETA_EQ, color="k", ls=":", lw=0.7)
    ax.plot(xb, yb, "o-", label="BL", ms=4)
    ax.plot(xm, ym, "o-", label="MDN", ms=4)
    ax.set_title("⟨ΔE_trans | η_trans_pre⟩ (K)")
    ax.set_xlabel("η_trans_pre")
    ax.set_ylabel("⟨ΔE_trans⟩ (K)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"BL vs MDN collision-log diagnostic\n"
        f"BL: {bl_path.name}    MDN: {mdn_path.name}",
        fontsize=11,
    )
    fig.tight_layout()
    paths.ensure_parent(output_path)
    fig.savefig(output_path, dpi=120)
    print(f"Saved figure to {output_path}")

    # Summary numbers
    def _last_nonzero(arr):
        m = np.isfinite(arr) & (arr != 0)
        return float(arr[m][-1]) if m.any() else float("nan")

    print()
    print(
        f"Final snapshot: BL step={int(bl['snapshot_steps'][k_bl])}, "
        f"MDN step={int(mdn['snapshot_steps'][k_mdn])}"
    )
    print(f"  BL  ⟨η_trans_pre⟩  = {qb['eta_trans_pre'].mean():.4f}")
    print(f"  MDN ⟨η_trans_pre⟩  = {qm['eta_trans_pre'].mean():.4f}")
    print(f"  BL  ⟨η_trans_post⟩ = {qb['eta_trans_post'].mean():.4f}")
    print(f"  MDN ⟨η_trans_post⟩ = {qm['eta_trans_post'].mean():.4f}    "
          f"(target {ETA_EQ:.4f})")
    print(
        f"  BL  ⟨ΔE_trans⟩ last step = {_last_nonzero(bl['delta_E_trans_mean']) / KB:+.2f} K"
    )
    print(
        f"  MDN ⟨ΔE_trans⟩ last step = {_last_nonzero(mdn['delta_E_trans_mean']) / KB:+.2f} K"
    )
    print(
        f"  BL  max |ΔE_total| over run = {np.max(bl['energy_residual_max']) / KB:.3e} K"
    )
    print(
        f"  MDN max |ΔE_total| over run = {np.max(mdn['energy_residual_max']) / KB:.3e} K"
    )

    plt.show()


if __name__ == "__main__":
    bl_path = paths.log_path("H2_energy_relaxation_BL.npz")
    mdn_path = paths.log_path("H2_energy_relaxation_mdn_H2_wf1.npz")
    output_path = paths.plot_path("compare_BL_vs_mdn_H2_wf1.png")
    main(bl_path, mdn_path, output_path)
