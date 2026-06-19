"""Recursive equilibrium of the trained MDN kernel at the DSMC operating point.

The MDN is trained on a *one-shot* map: minimise the NLL of the post-collision
fractions (eta_tr', eta_rot') given the pre-collision state, averaged over the
(uniform-box) CTC training distribution.  DSMC, however, applies the kernel
*recursively* and converges to the stationary distribution of the iterated
operator under NTC acceptance weighting -- an emergent global functional that the
one-shot loss never sees.  Nothing in training penalises the model for failing
detailed balance or for drifting at the operating point DSMC actually visits.

This script makes that gap visible.  On a fixed collision-energy shell we iterate
the MDN kernel the way DSMC does (a pair collides with probability proportional to
its relative speed g ~ sqrt(eta), the NTC acceptance weighting) starting from
several eta_0, and watch where the mean translational fraction <eta_tr> settles.
A reversible (detailed-balance) kernel would relax every start to equipartition
3/7; the trained MDN instead settles at a biased fixed point eta*_MDN != 3/7.

The shell is fixed at E_c ~ 5000 K, inside the dense CTC training region where the
one-shot fit is well constrained, so the biased recursive fixed point cannot be
blamed on a poor fit.  The fixed point is reported against the actual <eta_tr> the
full DSMC run converges to (which operates at a lower, energy-averaged shell).

Run:  python -m analysis.recursive_equilibrium
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig
from config.experiment_config import ExperimentConfig
from experiments.energy_relaxation import load_mdn
from analysis.kernel_stationarity import _apply_mdn  # chunked MDN sampling

pc = PlottingConfig()
cfg = ExperimentConfig()

EQ_FRACTION = 3.0 / 7.0  # equipartition: 3 trans DOF / (3 + 2 + 2)

# Collision-energy shell (E/kB, in K) the MDN is queried at -- the dense CTC
# training region, where the one-shot fit is well constrained.
E_C_MDN = 5000.0

N_PARTICLES = 10_000
N_COLLISIONS = 150  # mean collisions per molecule to iterate each chain to
N_STEPS = 350  # iteration steps (acceptance < 1, so > N_COLLISIONS steps needed)
ETA0_LIST = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
SEED = 0

# (label, trained best-model path, DSMC relaxation npy)
CASES = [
    (
        "H$_2$",
        f"results/h2/models/mdn/best_model_bs{cfg.batch_size}_ngauss{cfg.num_mixtures}.pth",
        "data/ml-dsmc/mdn/h2/best_model_relaxation.npy",
    ),
    (
        "O$_2$",
        f"results/o2/models/mdn/best_model_bs{cfg.batch_size}_ngauss{cfg.num_mixtures}.pth",
        "data/ml-dsmc/mdn/o2/best_model_relaxation.npy",
    ),
]


# --------------------------------------------------------------------------- #
# MDN kernel: `step(vals, idx, rng)` returns a full eta vector with the
# NTC-accepted particles `idx` updated by one collision at the fixed shell E_C_MDN.
# --------------------------------------------------------------------------- #
def make_mdn_step(model):
    def step(vals, idx, rng):
        out = vals.copy()
        n = len(idx)
        if n == 0:
            return out
        # Rotation partition is a nuisance variable on the eta_tr marginal;
        # integrate it out by drawing fresh each collision (cf. kernel_stationarity).
        eta_rot_A = rng.uniform(0.0, 1.0, size=n)
        E_total = np.full(n, E_C_MDN)
        out[idx] = np.clip(_apply_mdn(model, E_total, vals[idx], eta_rot_A), 0.0, 1.0)
        return out

    return step


# --------------------------------------------------------------------------- #
# Recursive iteration under NTC acceptance weighting
# --------------------------------------------------------------------------- #
def iterate(step, rng):
    """For each eta_0: collisions/molecule axis and <eta_tr> trajectory.
    Returns (list_of_x, list_of_traj, eta_star) where eta_star is the
    path-independent fixed point (mean over starts of last-20-step mean)."""
    xs_all, traj_all, finals = [], [], []
    for eta0 in ETA0_LIST:
        v = np.full(N_PARTICLES, float(eta0))
        ncoll = np.zeros(N_PARTICLES)
        xs, traj = [0.0], [v.mean()]
        for _ in range(N_STEPS):
            # NTC acceptance: a pair collides with probability ~ g ~ sqrt(eta).
            idx = np.where(rng.random(N_PARTICLES) < np.sqrt(np.clip(v, 0, 1)))[0]
            ncoll[idx] += 1
            v = step(v, idx, rng)
            xs.append(ncoll.mean())
            traj.append(v.mean())
        xs_all.append(xs)
        traj_all.append(traj)
        finals.append(np.mean(traj[-20:]))
    return xs_all, traj_all, float(np.mean(finals))


def dsmc_converged_eta(npy_path):
    """<eta_tr> the full DSMC relaxation run settles to (last 20% of steps).
    eta_tr = 1.5 T_trans / (1.5 T_trans + 2 T_rot)  -> 3/7 at equipartition."""
    a = np.load(npy_path)
    k = max(1, len(a) // 5)
    Tt = a["T_trans_mean"][-k:].mean()
    Tr = a["T_rot_mean"][-k:].mean()
    return 1.5 * Tt / (1.5 * Tt + 2.0 * Tr)


# --------------------------------------------------------------------------- #
# Figure: two panels (H2 | O2), recursive MDN trajectories from several eta_0
# --------------------------------------------------------------------------- #
def main(output_path: str | None = None):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(
        SEED
    )  # the MDN samples through torch; seed it for reproducibility
    fig, axes = plt.subplots(1, 2, figsize=(2 * pc.figsize[0], pc.figsize[1]))

    print(f"\nMDN queried at the collision shell E_c = {E_C_MDN:.0f} K")

    for ax, (title, model_path, dsmc_npy) in zip(axes, CASES):
        print(f"\n=== {title} ===")
        model = load_mdn(model_path)
        mdn_step = make_mdn_step(model)

        xs_all, traj_all, eta_star = iterate(mdn_step, rng)
        for j, (xs, traj) in enumerate(zip(xs_all, traj_all)):
            ax.plot(
                xs,
                traj,
                color="tab:red",
                lw=1.3,
                alpha=0.75,
                label="MDN" if j == 0 else None,
            )
        print(f"  MDN recursive fixed point eta* = {eta_star:.3f}")

        dsmc_eta = dsmc_converged_eta(dsmc_npy)
        print(
            f"  DSMC converged eta_tr          = {dsmc_eta:.3f}  (3/7 = {EQ_FRACTION:.3f})"
        )

        ax.axhline(
            EQ_FRACTION, color="black", lw=1.4, ls="--", label=r"equipartition $3/7$"
        )
        ax.set_title(title, fontsize=pc.label_fontsize)
        ax.set_xlabel("mean collisions per molecule", fontsize=pc.label_fontsize)
        ax.set_xlim(0.0, N_COLLISIONS)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=pc.legend_fontsize, loc="upper right")

    axes[0].set_ylabel(r"$\langle\eta_{trans}\rangle$", fontsize=pc.label_fontsize)
    fig.suptitle(
        "The MDN's recursive fixed point is biased away from equipartition",
        fontsize=pc.label_fontsize + 1,
    )
    fig.tight_layout()

    out = output_path or paths.plot_path("recursive_equilibrium.png")
    fig.savefig(out, dpi=300)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    main()
