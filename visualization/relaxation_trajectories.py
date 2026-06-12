"""Relaxation of the translational energy fraction under repeated CTC collisions.

We build the empirical single-collision kernel K(eta'|eta) from the CTC dataset at
fixed collision energy E_c (energy is conserved per collision, so the chain stays
on the same energy shell) and iterate it from several starting fractions eta_0.

Collisions are selected at the physical DSMC rate: a pair collides with
probability proportional to its relative speed g ∝ sqrt(Etr) ∝ sqrt(eta) at fixed
E_c (the g/g_max acceptance factor).  With this flux weighting every initial
state relaxes to the equipartition value 3/(3+2+2) = 3/7 -- including the one that
starts at the single-collision balance point (~0.6), where one *average* collision
exchanges zero NET energy.

This makes the single-shot vs. many-shot distinction visual: the per-collision
zero-net-exchange point is NOT the equilibrium; randomization over many flux-
weighted collisions drives the mean to 3/7.
"""

import numpy as np
import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig

pc = PlottingConfig()
rng = np.random.default_rng(0)

EQ_FRACTION = 3.0 / 7.0
E_C_LO, E_C_HI = 4500.0, 5500.0  # fixed-E_c shell, well inside the sampling box
N_BINS = 50
N_PARTICLES = 120_000
N_STEPS = 350
ETA0_LIST = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

CASES = [
    ("H$_2$",
     "data/ctc/h2/impactparam/Erelmax10000/"
     "H2_collisions_b1_6_uniform_Erelmax10000_ncoll1000000_seed42.npy"),
    ("O$_2$",
     "data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax1_5.npy"),
]

edges = np.linspace(0.0, 1.0, N_BINS + 1)
ctr = 0.5 * (edges[:-1] + edges[1:])


def build_kernel(eta, etap):
    """Empirical K(eta'|eta): per pre-eta bin, the pool of post-eta' values.
    Sparse (edge) bins borrow neighbours so none is absorbing."""
    bi = np.clip(np.digitize(eta, edges) - 1, 0, N_BINS - 1)
    lib = [etap[bi == i] for i in range(N_BINS)]
    for i in range(N_BINS):
        if len(lib[i]) < 50:
            lo, hi = max(0, i - 2), min(N_BINS, i + 3)
            lib[i] = np.concatenate([etap[bi == j] for j in range(lo, hi)])
    return lib


def one_shot_crossing(eta, etap):
    """Diagonal crossing of m(eta)=<eta'|eta> (the single-collision balance point)."""
    bi = np.clip(np.digitize(eta, edges) - 1, 0, N_BINS - 1)
    m = np.array([etap[bi == i].mean() if (bi == i).sum() > 50 else np.nan
                  for i in range(N_BINS)])
    g = m - ctr
    ok = ~np.isnan(g)
    x, y = ctr[ok], g[ok]
    idx = np.where(np.diff(np.signbit(y)))[0]
    cs = [x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]) for i in idx]
    return min(cs, key=lambda c: abs(c - 0.5)) if cs else np.nan


def collide(vals, idx, lib):
    """Apply one collision to the flux-selected particles `idx`."""
    out = vals.copy()
    b = np.clip(np.digitize(vals[idx], edges) - 1, 0, N_BINS - 1)
    for i in range(N_BINS):
        sel = idx[b == i]
        if len(sel):
            out[sel] = rng.choice(lib[i], size=len(sel))
    return out


fig, axes = plt.subplots(1, 2, figsize=(2 * pc.figsize[0], pc.figsize[1]))
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(ETA0_LIST)))

for ax, (title, fname) in zip(axes, CASES):
    d = np.load(fname)
    Etr, Er1, Er2, Etrp, Er1p, Er2p = d.T
    Epre = Etr + Er1 + Er2
    Epost = Etrp + Er1p + Er2p
    eta = Etr / Epre
    etap = Etrp / Epost
    m = (Epre >= E_C_LO) & (Epre < E_C_HI)
    eta, etap = eta[m], etap[m]

    lib = build_kernel(eta, etap)
    cross = one_shot_crossing(eta, etap)

    finals = []
    for c, eta0 in zip(cmap, ETA0_LIST):
        v = np.full(N_PARTICLES, eta0)
        ncoll = np.zeros(N_PARTICLES)          # collisions experienced per particle
        xs = [0.0]
        traj = [v.mean()]
        for _ in range(N_STEPS):
            idx = np.where(rng.random(N_PARTICLES) < np.sqrt(np.clip(v, 0, 1)))[0]
            ncoll[idx] += 1
            v = collide(v, idx, lib)
            xs.append(ncoll.mean())            # mean collisions per molecule
            traj.append(v.mean())
        finals.append(traj[-1])
        lw = 2.6 if abs(eta0 - 0.60) < 1e-9 else 1.6
        ax.plot(xs, traj, color=c, lw=lw, label=fr"$\eta_0={eta0:.2f}$")
    print(f"{title}: one-shot crossing={cross:.3f}; "
          f"final means span [{min(finals):.3f}, {max(finals):.3f}]; 3/7={EQ_FRACTION:.3f}")

    ax.axhline(EQ_FRACTION, color="black", lw=1.4, ls="--",
               label=r"equipartition $3/7$ (many-shot)")
    ax.axhline(cross, color="crimson", lw=1.2, ls=":",
               label=fr"single-collision balance ($\Delta\eta=0$), {cross:.2f}")
    ax.set_title(title, fontsize=pc.label_fontsize)
    ax.set_xlabel("mean collisions per molecule", fontsize=pc.label_fontsize)
    ax.set_xlim(0, max(xs))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=pc.legend_fontsize, loc="upper right")

axes[0].set_ylabel(r"$\langle\eta_{trans}\rangle$", fontsize=pc.label_fontsize)
fig.suptitle(
    "Repeated flux-weighted CTC collisions relax every initial state to "
    "equipartition (3/7)",
    fontsize=pc.label_fontsize + 1)
fig.tight_layout()

out = paths.plot_path("relaxation_trajectories.png")
fig.savefig(out, dpi=300)
print(f"Saved {out}")
