"""H2 energy-relaxation comparison across E_rel input distributions.

Runs the standard H2 relaxation experiment (T_trans=3000K → T_rot=1000K →
equilibrium) for three MDNs trained on datasets with different E_rel
distributions (uniform, MB, NTC), then plots the results as three side-by-side
subplots sharing y-axis limits. BL-DSMC is shown as a dashed reference in
every subplot.

Models are expected in results/models/mdn/Etrans_distribution/{uniform,mb,ntc}.pth.
Generate them with: python training/parametersweeps/dist_sweep.py
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

import paths
from config.plotting_config import PlottingConfig
from experiments.energy_relaxation import (
    SimulationParams,
    load_mdn,
    load_sparta_reference,
    load_lammps_reference,
    print_relaxation_table,
    run_relaxation,
    run_relaxation_comparison,
)
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.species import Species

MODEL_DIR = paths.MDN_DIR / "Etrans_distribution"
DISTRIBUTIONS = ("uniform", "mb", "ntc")
DIST_LABELS = {"uniform": "Uniform", "mb": "Maxwell-Boltzmann", "ntc": "NTC-matched"}

SPARTA_PATH = "sparta/output/sparta_H2_energy_relaxationTtr3000_Trot1000.dat"
LAMMPS_PATH = "lammps/output/lammps_H2_energy_relaxation.dat"


def main(
    sparta_path: str = SPARTA_PATH,
    lammps_path: str = LAMMPS_PATH,
    output_path: str | None = None,
    nr_steps: int = 100,
    randomseed: int = 1,
    d: float = 10.1e-10,
    zrot_bl: float = 5.0,
    zrot_mdn: float = 5.0 / 3.5,
    ylim: tuple[float, float] = (1000.0, 3000.0),
):
    species = replace(
        Species.H2(),
        diameter=d,
        zrot_bl=zrot_bl,
        zrot_mdn=zrot_mdn,
    )
    params = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=3000.0,
        rot_temperature=1000.0,
        randomseed=randomseed,
        grid_cells=(5, 5, 5),
        box_size=1.0e-7,
        dt=1.0e-11,
    )

    # --- run BL once as shared reference ------------------------------------
    print("\n--- Running DSMC relaxation: BL (reference) ---")
    bl_model = borgnakke_larssen_model(randomseed=randomseed)
    bl_stats = run_relaxation(species, bl_model, params=params)

    # --- run one MDN per distribution ---------------------------------------
    mdn_results: dict[str, dict] = {}
    for dist in DISTRIBUTIONS:
        model_path = MODEL_DIR / f"{dist}.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Run training/parametersweeps/dist_sweep.py first."
            )
        print(f"\n--- Running DSMC relaxation: MDN ({dist}) ---")
        model = load_mdn(str(model_path), randomseed=randomseed)
        mdn_results[dist] = run_relaxation(species, model, params=params)

    # --- reference data -----------------------------------------------------
    sparta = load_sparta_reference(sparta_path) if sparta_path else None
    try:
        lammps = load_lammps_reference(lammps_path)
    except Exception:
        lammps = None

    # --- console summary ----------------------------------------------------
    all_results = {"BL (ML-DSMC)": bl_stats} | {
        f"MDN ({DIST_LABELS[d]})": mdn_results[d] for d in DISTRIBUTIONS
    }
    print_relaxation_table(
        all_results, sparta, rot_temperature_initial=params.rot_temperature, lammps=lammps
    )

    # --- 3-subplot figure ---------------------------------------------------
    out = output_path or str(paths.plot_path("H2_dist_sweep_relaxation.png"))
    fig = _plot(bl_stats, mdn_results, sparta=sparta, lammps=lammps, ylim=ylim)
    paths.ensure_parent(out)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {out}")
    plt.show()


def _plot(
    bl_stats: dict,
    mdn_results: dict[str, dict],
    *,
    sparta: dict | None,
    lammps: dict | None,
    ylim: tuple[float, float],
) -> plt.Figure:
    pc = PlottingConfig()
    fig, axes = plt.subplots(1, 3, figsize=(pc.figsize[0] * 1.8, pc.figsize[1]),
                             sharey=True)

    for ax, dist in zip(axes, DISTRIBUTIONS):
        stats = mdn_results[dist]
        t = stats["timestep"]
        bl_t = bl_stats["timestep"]

        # MDN traces
        ax.plot(t, stats["T_trans_mean"], color="tab:orange",
                label=r"$T_\mathrm{trans}$ MDN")
        ax.plot(t, stats["T_rot_mean"], color="tab:blue",
                label=r"$T_\mathrm{rot}$ MDN")

        # BL reference
        ax.plot(bl_t, bl_stats["T_trans_mean"], color="tab:orange",
                linestyle="--", label=r"$T_\mathrm{trans}$ BL")
        ax.plot(bl_t, bl_stats["T_rot_mean"], color="tab:blue",
                linestyle="--", label=r"$T_\mathrm{rot}$ BL")

        # SPARTA reference
        if sparta is not None:
            ax.plot(sparta["t"], sparta["T_trans"], color="tab:orange",
                    linestyle=":", label=r"$T_\mathrm{trans}$ SPARTA")
            ax.plot(sparta["t"], sparta["T_rot"], color="tab:blue",
                    linestyle=":", label=r"$T_\mathrm{rot}$ SPARTA")

        ax.set_title(DIST_LABELS[dist], fontsize=pc.label_fontsize,
                     fontweight=pc.label_fontweight)
        ax.set_xlabel("Time [s]", fontsize=pc.label_fontsize)
        ax.ticklabel_format(style="sci", scilimits=(-2, 3))
        ax.set_ylim(*ylim)
        ax.grid()
        ax.legend(fontsize=pc.legend_fontsize)

    axes[0].set_ylabel("Temperature [K]", fontsize=pc.label_fontsize,
                        fontweight=pc.label_fontweight)
    fig.suptitle("H2 energy relaxation — E$_\\mathrm{rel}$ training distribution comparison",
                 fontsize=pc.label_fontsize + 1, fontweight=pc.label_fontweight)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
