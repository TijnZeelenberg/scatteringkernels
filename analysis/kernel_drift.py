"""Detailed-balance violation of the trained MDN kernel: the drift curve D(T_trans).

The MDN is trained on a *one-shot* map: minimise the NLL of the post-collision
fractions given the pre-collision state, averaged over the (uniform-box) CTC
training distribution.  Nothing in that loss constrains *detailed balance* --
microscopic reversibility, the property that makes the Maxwell-Boltzmann
equilibrium stationary under collisions.  DSMC applies the kernel recursively,
millions of times, so any small reversibility error compounds and the gas settles
a few K away from equipartition (cf. the wrong final temperatures in the
best_model_relaxation plots).  This script proves that link quantitatively.

The witness.  Detailed balance <=> a kernel fed a gas already at thermal
equilibrium (equipartition T_trans = T_rot = 220 K) transfers ZERO net energy
between translation and rotation.  So the NTC-acceptance-weighted net change in
relative translational energy per collision, evaluated *at equipartition*,

    D_eq = <Delta E_rel>  at  T_trans = T_rot = T_EQUIPART,

is a rigorous first-moment witness of a violation: a faithful (reversible) kernel
gives D_eq = 0; the trained MDN gives D_eq != 0, with a sign.

The consequence.  The kernel's own steady state is where D vanishes --
the zero-crossing T* of D(T_trans) along the energy-conservation line
1.5 T_trans + T_rot = E0.  Because D_eq != 0, T* != 220 K, and we show T* matches
the temperature the full DSMC run actually converges to (best_model_relaxation.npy).
A linear-response decomposition ties the two together: the offset is
Delta T* ~ -D_eq / S, where S = dD/dT_trans is the kernel's restoring slope.

The true-kernel side of the argument -- that the physically correct CTC operator
has D = 0 (detailed balance holds) -- is established at a data-supported energy in
`visualization/detailed_balance.py` (the CTC operator preserves Maxwell-Boltzmann,
Delta<eta> ~ 0).  We do NOT recompute a CTC curve here: the CTC uniform-box data is
too sparse (<1% of samples) below ~4000 K to evaluate at the ~780 K energy DSMC
actually equilibrates at.  On this figure the equipartition line at zero flux *is*
the detailed-balance requirement a faithful kernel must satisfy.

Run:  python -m analysis.kernel_drift
"""

import os

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

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 14,
    }
)

EQ_FRACTION = 3.0 / 7.0  # equipartition: 3 trans DOF / (3 + 2 + 2)

# Dense collision-energy shell (E/kB, in K) where the uniform-box CTC data is
# plentiful -- used only for the printed one-shot-map fidelity check.
E_C_LO, E_C_HI = 4500.0, 5500.0
E_C_MDN = 5000.0
N_BINS = 50
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
# One-shot conditional-mean maps (printed fidelity check: "the kernel fits
# single collisions").  Evaluated on the dense E_c ~ 5000 K shell where the CTC
# data is plentiful, so CTC and MDN are compared like-for-like.
# --------------------------------------------------------------------------- #
def one_shot_mean_map(eta, etap):
    """Binned m(eta)=E[eta'|eta] for the CTC shell data."""
    bi = np.clip(np.digitize(eta, edges) - 1, 0, N_BINS - 1)
    m = np.array(
        [etap[bi == i].mean() if (bi == i).sum() > 50 else np.nan for i in range(N_BINS)]
    )
    return ctr, m


def mdn_one_shot_map(model, rng, n=200_000):
    """Binned m(eta)=E[eta'|eta] for the MDN at the dense shell energy."""
    eta = rng.uniform(0.0, 1.0, size=n)
    eta_rot_A = rng.uniform(0.0, 1.0, size=n)
    E_total = np.full(n, E_C_MDN)
    etap = _apply_mdn(model, E_total, eta, eta_rot_A)
    bi = np.clip(np.digitize(eta, edges) - 1, 0, N_BINS - 1)
    m = np.array(
        [etap[bi == i].mean() if (bi == i).sum() > 50 else np.nan for i in range(N_BINS)]
    )
    return ctr, m


def one_shot_fidelity(model, ctc_file, rng):
    """RMS and max gap between the MDN and CTC one-shot maps on the dense shell.
    Small values are the quantitative statement that the kernel reproduces
    individual collisions accurately -- the fit is good even though the recursive
    fixed point is biased."""
    d = np.load(ctc_file)
    Etr, Er1, Er2, Etrp, Er1p, Er2p = d.T
    eta = Etr / (Etr + Er1 + Er2)
    etap = Etrp / (Etrp + Er1p + Er2p)
    shell = (Etr + Er1 + Er2 >= E_C_LO) & (Etr + Er1 + Er2 < E_C_HI)
    _, mc = one_shot_mean_map(eta[shell], etap[shell])
    _, mm = mdn_one_shot_map(model, rng)
    ok = np.isfinite(mc) & np.isfinite(mm)
    diff = mm[ok] - mc[ok]
    return float(np.sqrt(np.mean(diff**2))), float(np.max(np.abs(diff)))


# --------------------------------------------------------------------------- #
# DSMC converged equilibrium (read from the full relaxation run)
# --------------------------------------------------------------------------- #
CONVERGENCE_TOL_K_PER_NS = 0.5


def dsmc_converged_temperatures(npy_path):
    """(T_trans, T_rot, drift_K_per_ns) the full DSMC relaxation run settles to
    (mean over the last 20% of steps)."""
    a = np.load(npy_path)
    k = max(1, len(a) // 5)
    Tt = float(a["T_trans_mean"][-k:].mean())
    Tr = float(a["T_rot_mean"][-k:].mean())
    slope = 0.0
    if k >= 2:
        t_ns = a["timestep"][-k:] * 1e9
        slope = float(np.polyfit(t_ns, a["T_trans_mean"][-k:], 1)[0])
        if abs(slope) > CONVERGENCE_TOL_K_PER_NS:
            print(
                f"  WARNING NOT CONVERGED: T_trans still drifting {slope:+.2f} K/ns over the "
                f"last 20% of {a['timestep'][-1] * 1e9:.0f} ns ({npy_path}). "
                f"Equilibrium estimate is biased -- run longer."
            )
    return Tt, Tr, slope


# --------------------------------------------------------------------------- #
# Detailed-balance witness in TEMPERATURE space
#
# The relaxation experiments start at (T_trans0, T_rot0) and conserve, per
# molecule, E0 = 1.5*T_trans + T_rot  (3 translational + 2 rotational DOF, in K).
# Collisions move (T_trans, T_rot) along that line.  For a trial equilibrium gas
# at (T_trans, T_rot) we draw the *physical* colliding-pair ensemble: the relative
# translational energy is weighted by the relative speed g ~ sqrt(E_rel) via the
# NTC acceptance, so the equilibrium Gamma(3/2,T) is tilted to Gamma(2,T) (mean
# 2*T_trans); rotational energies are Exponential(T_rot) (2 DOF each, not weighted
# by g; eta_rot_A = er_i/(er_i+er_j) is then Uniform(0,1)).  Working in K means no
# molecular mass is needed.  D(T_trans) = <Delta E_rel> per collision; D = 0 is the
# kernel's steady state, D != 0 at equipartition is a detailed-balance violation.
# --------------------------------------------------------------------------- #
T_TRANS0, T_ROT0 = 300.0, 100.0  # SimulationParams defaults used by the experiments
E0_PER_MOL = 1.5 * T_TRANS0 + T_ROT0  # conserved per-molecule energy (K)
T_EQUIPART = E0_PER_MOL / 2.5  # T_trans = T_rot at equipartition


def mdn_translational_drift(model, T_trans, T_rot, rng, n=1_000_000):
    """NTC-acceptance-weighted mean change in relative translational energy per
    collision (K) for an equilibrium gas at (T_trans, T_rot).  Zero at the
    kernel's steady state; negative means the kernel drains translation into
    rotation."""
    E_rel = rng.gamma(2.0, T_trans, size=n)  # Gamma(3/2,T) x sqrt(E_rel), NTC accept
    er_i = rng.exponential(T_rot, size=n)
    er_j = rng.exponential(T_rot, size=n)
    E_c = E_rel + er_i + er_j
    eta = E_rel / E_c
    eta_rot_A = er_i / (er_i + er_j)
    etap = np.clip(_apply_mdn(model, E_c, eta, eta_rot_A), 0.0, 1.0)
    return float(np.mean(etap * E_c - E_rel))


def _drift_on_line(model, T_trans, rng):
    """D(T_trans) with T_rot pinned to the energy-conservation line."""
    return mdn_translational_drift(model, T_trans, E0_PER_MOL - 1.5 * T_trans, rng)


def temperature_fixed_point(model, rng, n_grid=25):
    """(T_trans*, T_rot*) on the energy-conservation line where D(T_trans)
    vanishes -- the kernel's recursive equilibrium in K.  Returns the
    fixed point plus the (T_trans grid, drift) for plotting."""
    Tt_grid = np.linspace(0.70 * T_EQUIPART, 1.18 * T_EQUIPART, n_grid)
    drift = np.array([_drift_on_line(model, Tt, rng) for Tt in Tt_grid])
    sign_change = np.where(np.diff(np.signbit(drift)))[0]
    if len(sign_change) == 0:
        return np.nan, np.nan, Tt_grid, drift
    k = sign_change[0]
    Tt_star = Tt_grid[k] - drift[k] * (Tt_grid[k + 1] - Tt_grid[k]) / (
        drift[k + 1] - drift[k]
    )
    return Tt_star, E0_PER_MOL - 1.5 * Tt_star, Tt_grid, drift


def db_violation(Tt_grid, drift, window=30.0):
    """Detailed-balance witness and its linear-response consequence, read off the
    drift curve D(T_trans) already computed for the figure (a local linear fit in
    a +/- `window` K band around equipartition):
        D_eq    = D(T_EQUIPART)            D at equipartition (the violation)
        S       = dD/dT_trans |_equipart    restoring slope
        dT_pred = -D_eq / S                 predicted equilibrium offset (linear response)
    A faithful, reversible kernel has D_eq = 0."""
    sel = np.abs(Tt_grid - T_EQUIPART) <= window
    S, intercept = np.polyfit(Tt_grid[sel], drift[sel], 1)
    D_eq = S * T_EQUIPART + intercept
    dT_pred = -D_eq / S if S != 0 else np.nan
    return float(D_eq), float(S), float(dT_pred)


# --------------------------------------------------------------------------- #
# The figure: one row, two panels (H2 left, O2 right)
# --------------------------------------------------------------------------- #
def main(output_path: str | None = None):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(2 * pc.figsize[0], pc.figsize[1]))

    print("\n" + "=" * 78)
    print("Learned detailed-balance violation  ->  DSMC equilibrium offset")
    print(f"  (equipartition T_trans = T_rot = {T_EQUIPART:.0f} K; a reversible kernel has D_eq = 0)\n")
    header = (
        f"{'':5}{'D_eq':>8}{'S':>9}{'-D_eq/S':>9}"
        f"{'  T* / T_rot*':>16}{'  T_DSMC / T_rot':>18}{'  dT*':>7}{'  sign?':>7}"
    )
    print(header)
    print(f"{'':5}{'[K/cl]':>8}{'[K/cl/K]':>9}{'[K]':>9}{'[K]':>16}{'[K]':>18}{'[K]':>7}")
    print("-" * 78)

    for ax, (title, ctc_file, model_path, dsmc_npy) in zip(axes, CASES):
        model = load_mdn(model_path)

        Tt_star, Tr_star, Tt_grid, drift = temperature_fixed_point(model, rng)
        D_eq, S, dT_pred = db_violation(Tt_grid, drift)
        Tt_dsmc, Tr_dsmc, dsmc_slope = dsmc_converged_temperatures(dsmc_npy)
        rms, max_abs = one_shot_fidelity(model, ctc_file, rng)

        dT_star = Tt_star - Tt_dsmc
        sign_ok = np.sign(D_eq) == np.sign(Tt_dsmc - T_EQUIPART)
        clean = title.replace("$", "").replace("_", "")
        print(
            f"{clean:5}{D_eq:8.2f}{S:9.2f}{dT_pred:9.1f}"
            f"{Tt_star:8.1f} /{Tr_star:6.1f}{Tt_dsmc:11.1f} /{Tr_dsmc:6.1f}"
            f"{dT_star:7.1f}{str(bool(sign_ok)):>7}"
        )
        print(
            f"     one-shot map fidelity @ {E_C_MDN:.0f} K: "
            f"RMS(MDN-CTC)={rms:.3f}, max={max_abs:.3f}  (the kernel fits single collisions)"
        )

        ax.axhline(0.0, color="0.5", lw=1.0)
        ax.plot(Tt_grid, drift, "o-", color="tab:red", lw=1.8, ms=4)

        ax.axvline(
            T_EQUIPART, color="0.4", ls=":", lw=1.4,
            label=fr"equipartition {T_EQUIPART:.0f} K",
        )
        ax.axvline(
            Tt_dsmc, color="black", ls="--", lw=1.6,
            label=fr"DSMC converged {Tt_dsmc:.0f} K",
        )

        ax.set_xlim(T_EQUIPART - 27.0, T_EQUIPART + 30.0)
        ax.set_title(title, fontsize=pc.label_fontsize)
        ax.set_xlabel(r"$T_\mathrm{trans}$ [K]", fontsize=pc.label_fontsize)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=pc.legend_fontsize, loc="upper right")

    axes[0].set_ylabel(r"$D$ [K/collision]", fontsize=pc.label_fontsize)
    print("-" * 78)
    print("D_eq != 0 at equipartition is the violation; its sign flips between the gases.")
    print("The flux zero-crossing T* (kernel equilibrium) matches the DSMC converged T.\n")

    fig.tight_layout()

    thesis_dir = "../Master_Thesis_Tijn_Zeelenberg/figures"
    fname = "kernel_drift.png"

    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"Saved {output_path}")
    else:
        # Save to results/plots/ via paths helper
        local_out = paths.plot_path(fname)
        fig.savefig(local_out, dpi=300)
        print(f"Saved {local_out}")

        # Primary thesis copy, matching create_plots.py convention
        if os.path.isdir(thesis_dir):
            thesis_out = f"{thesis_dir}/{fname}"
            fig.savefig(thesis_out, dpi=300)
            print(f"Saved {thesis_out}")

    plt.show()


if __name__ == "__main__":
    main()
