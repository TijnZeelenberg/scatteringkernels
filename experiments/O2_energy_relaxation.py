"""O2 energy-relaxation experiment: MDN vs SPARTA reference."""

from __future__ import annotations

import matplotlib.pyplot as plt

import paths
from experiments.energy_relaxation import (
    SimulationParams,
    load_mdn,
    load_sparta_reference,
    plot_relaxation_comparison,
    print_relaxation_table,
    run_relaxation_comparison,
)
from physics.species import Species


def main(
    mdn_model_path: str = "results/models/weightsensitivity/O2_400000_uniform/mdn_O2_wf7.pth",
    sparta_path: str = "data/sparta_O2_energy_relaxation.dat",
    output_path: str | None = None,
    nr_steps: int = 2000,
    randomseed: int = 1,
):
    species = Species.O2()
    params = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=300.0,
        rot_temperature=100.0,
        randomseed=randomseed,
        grid_cells=(10, 10, 10),
    )

    models = {"MDN (ML-DSMC)": load_mdn(mdn_model_path, randomseed=randomseed)}
    results = run_relaxation_comparison(species, models, params=params)
    sparta = load_sparta_reference(sparta_path)

    print_relaxation_table(
        results, sparta, rot_temperature_initial=params.rot_temperature
    )

    output_path = output_path or paths.plot_path("O2_energy_relaxation.png")
    plot_relaxation_comparison(
        results,
        sparta,
        output_path=output_path,
        ylim=(20, 400),
        sparta_clip=nr_steps,
    )
    plt.show()


if __name__ == "__main__":
    main()
