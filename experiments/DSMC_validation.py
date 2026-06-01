"""Validate the local DSMC implementation (BL model) against SPARTA's VHS/VSS data."""

import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig
from experiments.energy_relaxation import (
    SimulationParams,
    load_sparta_reference,
    run_relaxation,
)
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.species import Species


def main(
    sparta_vhs_path: str = "data/sparta_H2_energy_relaxationVHS_zinv0151.dat",
    sparta_vss_path: str = "data/sparta_H2_energy_relaxationVSS_zinv0151.dat",
    output_path: str | None = None,
    nr_steps: int = 100,
    randomseed: int = 2,
):
    species = Species.H2()
    params = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=300.0,
        rot_temperature=100.0,
        randomseed=randomseed,
        grid_cells=(10, 10, 10),
    )

    bl_stats = run_relaxation(
        species, borgnakke_larssen_model(randomseed=randomseed), params=params
    )
    sparta_vhs = load_sparta_reference(sparta_vhs_path)
    sparta_vss = load_sparta_reference(sparta_vss_path)

    print("Final mean temperatures:")
    print(
        f"BL: T_trans = {bl_stats['T_trans_mean'][-20:-1].mean():.2f} K, "
        f"T_rot = {bl_stats['T_rot_mean'][-20:-1].mean():.2f} K"
    )
    print(
        f"SPARTA VHS: T_trans = {sparta_vhs['T_trans'][-20:-1].mean():.2f} K, "
        f"T_rot = {sparta_vhs['T_rot'][-20:-1].mean():.2f} K"
    )
    print(
        f"SPARTA VSS: T_trans = {sparta_vss['T_trans'][-20:-1].mean():.2f} K, "
        f"T_rot = {sparta_vss['T_rot'][-20:-1].mean():.2f} K"
    )

    pc = PlottingConfig()
    fig, ax = plt.subplots(figsize=pc.figsize)
    ax.plot(bl_stats["timestep"], bl_stats["T_trans_mean"], label=r"$T_{trans}$ BL VHS")
    ax.plot(bl_stats["timestep"], bl_stats["T_rot_mean"], label=r"$T_{rot}$ BL VHS")
    ax.plot(
        sparta_vhs["t"],
        sparta_vhs["T_trans"],
        color="red",
        linestyle="--",
        label=r"$T_{trans}$ BL VHS (SPARTA)",
    )
    ax.plot(
        sparta_vhs["t"],
        sparta_vhs["T_rot"],
        color="blue",
        linestyle="--",
        label=r"$T_{rot}$ BL VHS (SPARTA)",
    )
    ax.plot(
        sparta_vss["t"],
        sparta_vss["T_trans"],
        color="green",
        linestyle="--",
        label=r"$T_{trans}$ BL VSS (SPARTA)",
    )
    ax.plot(
        sparta_vss["t"],
        sparta_vss["T_rot"],
        color="orange",
        linestyle="--",
        label=r"$T_{rot}$ BL VSS (SPARTA)",
    )

    ax.set_xlabel(
        "Time [s]", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight
    )
    ax.set_ylabel(
        "Temperature [K]", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight
    )
    ax.ticklabel_format(style="sci", scilimits=(-2, 3))
    ax.set_ylim(20, 450)
    ax.grid()
    ax.legend(loc="upper right", fontsize=pc.legend_fontsize, ncol=2)

    output_path = output_path or paths.plot_path("DSMC_validation.png")
    fig.savefig(paths.ensure_parent(output_path), dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
