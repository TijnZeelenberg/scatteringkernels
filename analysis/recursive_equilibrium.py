"""Recursive equilibrium of the trained MDN kernel vs. the CTC ground truth.

The MDN is trained on a *one-shot* map: minimise the NLL of the post-collision
fractions (eta_tr', eta_rot') given the pre-collision state, averaged over the
(uniform-box) CTC training distribution.  DSMC, however, applies the kernel
*recursively* and converges to the stationary distribution of the iterated
operator under NTC acceptance weighting -- an emergent global functional that the
one-shot loss never sees.  Nothing in training penalises the model for failing
detailed balance or for drifting at the operating point DSMC actually visits.

This script makes that gap visible.  On a fixed collision-energy shell we iterate
each kernel the way DSMC does (a pair collides with probability proportional to
its relative speed g ~ sqrt(eta), the NTC acceptance weighting) starting from
several eta_0, and watch where the mean translational fraction <eta_tr> settles:

  * CTC (ground truth) relaxes every initial state to equipartition 3/7.
  * The trained MDN relaxes to a *different* fixed point eta*_MDN != 3/7,
    even though its one-shot conditional-mean map sits almost on top of the CTC
    map.  The one-shot fit is good; the recursive fixed point is biased.

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

# Fixed collision-energy shell (E/kB, in K).  CTC chains stay on the energy
# shell; the MDN is queried at the shell centre E_C_MDN.
E_C_LO, E_C_HI = 4500.0, 5500.0
E_C_MDN = 5000.0

N_BINS = 50
N_PARTICLES = 12_000
N_COLLISIONS = 200  # mean collisions per molecule to iterate each chain to
N_STEPS = 350  # iteration steps (acceptance < 1, so > N_COLLISIONS steps needed)
ETA0_LIST = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
SEED = 0

# (label, CTC dataset, trained best-model path, DSMC relaxation npy)
CASES = [
    (
        "H$_2$",
        "data/ctc/h2/impactparam/Erelmax10000/"
        "H2_collisions_b1_6_uniform_Erelmax10000_ncoll1000000_seed42.npy",
        f"results/h2/models/mdn/best_model_bs{cfg.batch_size}_ngauss{cfg.num_mixtures}.pth",
        "data/ml-dsmc/mdn/h2/best_model_relaxation.npy",
    ),
    (
        "O$_2$",
        "data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax1_5.npy",
        f"results/o2/models/mdn/best_model_bs{cfg.batch_size}_ngauss{cfg.num_mixtures}.pth",
        "data/ml-dsmc/mdn/o2/best_model_relaxation.npy",
    ),
]

edges = np.linspace(0.0, 1.0, N_BINS + 1)
ctr = 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------------- #
# Kernels: each `step(vals, idx, rng)` returns a full eta vector with the
# NTC-accepted particles `idx` updated by one collision.
# --------------------------------------------------------------------------- #
def build_ctc_library(eta, etap):
    """Empirical K(eta'|eta) on the shell: per pre-eta bin, the pool of post-eta'.
    Sparse (edge) bins borrow neighbours so none is absorbing."""
    bi = np.clip(np.digitize(eta, edges) - 1, 0, N_BINS - 1)
    lib = [etap[bi == i] for i in range(N_BINS)]
    for i in range(N_BINS):
        if len(lib[i]) < 50:
            lo, hi = max(0, i - 2), min(N_BINS, i + 3)
            lib[i] = np.concatenate([etap[bi == j] for j in range(lo, hi)])
    return lib


def make_ctc_step(lib):
    def step(vals, idx, rng):
        out = vals.copy()
        b = np.clip(np.digitize(vals[idx], edges) - 1, 0, N_BINS - 1)
        for i in range(N_BINS):
            sel = idx[b == i]
            if len(sel):
                out[sel] = rng.choice(lib[i], size=len(sel))
        return out

    return step


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
# Figure: two panels (H2 | O2), recursive trajectories from several eta_0
# --------------------------------------------------------------------------- #
def main(output_path: str | None = None):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)  # the MDN samples through torch; seed it for reproducibility
    fig, axes = plt.subplots(1, 2, figsize=(2 * pc.figsize[0], pc.figsize[1]))

    kernel_colors = {"CTC": "tab:blue", "MDN": "tab:red"}

    for ax, (title, ctc_file, model_path, dsmc_npy) in zip(axes, CASES):
        print(f"\n=== {title} ===")
        d = np.load(ctc_file)
        Etr, Er1, Er2, Etrp, Er1p, Er2p = d.T
        Epre = Etr + Er1 + Er2
        Epost = Etrp + Er1p + Er2p
        eta = Etr / Epre
        etap = Etrp / Epost
        shell = (Epre >= E_C_LO) & (Epre < E_C_HI)
        eta_s, etap_s = eta[shell], etap[shell]

        ctc_step = make_ctc_step(build_ctc_library(eta_s, etap_s))
        model = load_mdn(model_path)
        mdn_step = make_mdn_step(model)

        steps = {"CTC": ctc_step, "MDN": mdn_step}

        for name, step in steps.items():
            xs_all, traj_all, eta_star = iterate(step, rng)
            for j, (xs, traj) in enumerate(zip(xs_all, traj_all)):
                ax.plot(
                    xs,
                    traj,
                    color=kernel_colors[name],
                    lw=1.3,
                    alpha=0.75,
                    label=name if j == 0 else None,
                )
            print(f"  {name:3s} recursive fixed point eta* = {eta_star:.3f}")

        dsmc_eta = dsmc_converged_eta(dsmc_npy)
        print(f"  DSMC converged eta_tr           = {dsmc_eta:.3f}  (3/7 = {EQ_FRACTION:.3f})")

        ax.set_title(title, fontsize=pc.label_fontsize)
        ax.set_xlabel("mean collisions per molecule", fontsize=pc.label_fontsize)
        ax.set_xlim(0.0, N_COLLISIONS)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=pc.legend_fontsize, loc="upper right")

    axes[0].set_ylabel(r"$\langle\eta_{trans}\rangle$", fontsize=pc.label_fontsize)
    fig.suptitle(
        "One-shot training fits the kernel but its recursive fixed point is biased",
        fontsize=pc.label_fontsize + 1,
    )
    fig.tight_layout()

    out = output_path or paths.plot_path("recursive_equilibrium.png")
    fig.savefig(out, dpi=300)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    main()
