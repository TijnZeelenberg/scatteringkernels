"""Detailed-balance / stationarity check for the CTC collision operator.

A physical (microscopically reversible) collision operator must leave the
Maxwell-Boltzmann equilibrium distribution invariant: if the pre-collision
states are drawn from equilibrium, applying one collision must not change the
distribution of the translational energy fraction eta_trans = Etr / (Etr+Er1+Er2),
and its mean must sit at the equipartition value 3/(3+2+2) = 3/7.

The CTC datasets are sampled uniformly over an energy box, *not* from
equilibrium.  We therefore importance-reweight each collision by the
Maxwell-Boltzmann weight of its pre-collision state,

    w  =  sqrt(Etr) * exp(-(Etr + Er1 + Er2) / T)          [3 trans DOF, 2+2 rot DOF]

(the sqrt(Etr) is the translational density of states; the rotational DOS is
flat for 2 DOF).  We then compare the equilibrium-weighted distribution of
eta_trans *before* and *after* the collision.  Their overlap, centred on 3/7,
is the operational statement of detailed balance.

T is chosen low enough that the equilibrium support lies inside the sampling box
(so truncation is negligible) yet high enough to keep the effective sample size
large.
"""

import numpy as np
import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig

pc = PlottingConfig()

EQ_FRACTION = 3.0 / 7.0  # equipartition: 3 trans DOF / (3 + 2 + 2)
T_EQ = 1000.0  # K; box covers the MB support, ESS ~ 5% of 1e6
N_BINS = 60

CASES = [
    (
        "H$_2$",
        "data/ctc/h2/impactparam/Erelmax10000/"
        "H2_collisions_b1_6_uniform_Erelmax10000_ncoll1000000_seed42.npy",
    ),
    (
        "O$_2$",
        "data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax1_5.npy",
    ),
]


def mb_weights(Etr, Er1, Er2, T):
    """Maxwell-Boltzmann importance weights for the uniform-box pre-states."""
    logw = 0.5 * np.log(np.clip(Etr, 1e-9, None)) - (Etr + Er1 + Er2) / T
    logw -= logw.max()
    return np.exp(logw)


fig, axes = plt.subplots(1, 2, figsize=(2 * pc.figsize[0], pc.figsize[1]))
bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)

for ax, (title, fname) in zip(axes, CASES):
    d = np.load(fname)
    Etr, Er1, Er2, Etrp, Er1p, Er2p = d.T
    eta = Etr / (Etr + Er1 + Er2)
    etap = Etrp / (Etrp + Er1p + Er2p)

    w = mb_weights(Etr, Er1, Er2, T_EQ)
    ess = w.sum() ** 2 / (w**2).sum()
    eta_pre = np.average(eta, weights=w)
    eta_post = np.average(etap, weights=w)

    # Raw (uniform-box, non-equilibrium) pre distribution for contrast.
    ax.hist(
        eta,
        bins=bin_edges,
        density=True,
        color="0.6",
        alpha=0.35,
        label=f"uniform sampling (non-eq.), $\\langle\\eta\\rangle$={eta.mean():.3f}",
    )
    # Equilibrium-weighted pre and post distributions.
    ax.hist(
        eta,
        bins=bin_edges,
        weights=w,
        density=True,
        histtype="step",
        linewidth=2.0,
        color="tab:blue",
        label=f"MB equilibrium, before, $\\langle\\eta\\rangle$={eta_pre:.3f}",
    )
    ax.hist(
        etap,
        bins=bin_edges,
        weights=w,
        density=True,
        histtype="step",
        linewidth=2.0,
        linestyle="--",
        color="tab:orange",
        label=f"MB equilibrium, after, $\\langle\\eta\\rangle$={eta_post:.3f}",
    )
    ax.axvline(
        EQ_FRACTION,
        color="black",
        linewidth=1.2,
        linestyle=":",
        label=r"equipartition $3/7$",
    )

    ax.set_title(title, fontsize=pc.label_fontsize)
    ax.set_xlabel(r"$\eta_{trans} = E_{tr}/(E_{tr}+E_{r1}+E_{r2})$",
                  fontsize=pc.label_fontsize)
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=pc.legend_fontsize, loc="upper right")
    ax.grid(alpha=0.3)
    ax.text(
        0.03, 0.97,
        f"$T={T_EQ:.0f}$ K\ndrift $\\Delta\\langle\\eta\\rangle$={eta_post-eta_pre:+.3f}\n"
        f"ESS={ess/1e3:.0f}k",
        transform=ax.transAxes, va="top", ha="left", fontsize=pc.legend_fontsize,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
    )

axes[0].set_ylabel("probability density", fontsize=pc.label_fontsize)
fig.suptitle(
    "CTC collision operator preserves the Maxwell-Boltzmann equilibrium "
    "(detailed balance)",
    fontsize=pc.label_fontsize + 1,
)
fig.tight_layout()

out = paths.plot_path("detailed_balance.png")
fig.savefig(out, dpi=300)
print(f"Saved {out}")

# Also drop a copy next to the other thesis figures, matching create_plots.py.
thesis_dir = "../Master_Thesis_Tijn_Zeelenberg/figures"
import os
if os.path.isdir(thesis_dir):
    fig.savefig(f"{thesis_dir}/detailed_balance.png", dpi=300)
    print(f"Saved {thesis_dir}/detailed_balance.png")
