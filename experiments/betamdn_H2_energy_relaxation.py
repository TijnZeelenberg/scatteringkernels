"""H2 energy-relaxation experiment: Beta MDN vs Borgnakke-Larssen vs SPARTA."""

from __future__ import annotations

import matplotlib.pyplot as plt

import paths
from experiments.energy_relaxation import (
    SimulationParams,
    load_beta_mdn,
    load_sparta_reference,
    plot_relaxation_comparison,
    print_relaxation_table,
    run_relaxation_comparison,
)
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.species import Species


def main(
    beta_mdn_model_path: str = "results/models/beta_mdn/H2H2v1.pth",
    sparta_path: str = "data/sparta_H2_energy_relaxationVHS_zinv0151.dat",
    output_path: str | None = None,
    nr_steps: int = 100,
    randomseed: int = 1,
):
    species = Species.H2()
    params = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=300.0,
        rot_temperature=100.0,
        randomseed=randomseed,
        grid_cells=(5, 5, 5),
    )

    models = {
        "Beta MDN (ML-DSMC)": load_beta_mdn(beta_mdn_model_path, randomseed=randomseed),
        "BL (ML-DSMC)": borgnakke_larssen_model(randomseed=randomseed),
    }
    results = run_relaxation_comparison(species, models, params=params)
    sparta = load_sparta_reference(sparta_path)

    print_relaxation_table(results, sparta, rot_temperature_initial=params.rot_temperature)

    output_path = output_path or paths.plot_path("betamdn_H2_energy_relaxation.png")
    plot_relaxation_comparison(results, sparta, output_path=output_path)
    plt.show()


if __name__ == "__main__":
    main()
