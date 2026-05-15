"""H2 viscosity via Green-Kubo, comparing an MDN model and Borgnakke-Larssen."""

from __future__ import annotations

from experiments.energy_relaxation import SimulationParams, load_mdn, run_relaxation
from experiments.viscosity import green_kubo_viscosity
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.species import Species


def main(
    mdn_model_path: str = "results/models/mdn/weightsensitivity/H2_400000_dataseed42/mdn_H2_wf4.pth",
    nr_steps: int = 200,
    randomseed: int = 1,
    equilibration_steps: int = 50,
    max_lag: int = 100,
):
    species = Species.H2()
    params = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=220.0,
        rot_temperature=220.0,
        randomseed=randomseed,
        grid_cells=(5, 5, 5),
    )
    params_bl = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=220.0,
        rot_temperature=220.0,
        randomseed=randomseed,
        grid_cells=(10, 10, 10),
    )

    mdn = load_mdn(mdn_model_path, randomseed=randomseed)
    bl = borgnakke_larssen_model(randomseed=randomseed)

    mdn_stats = run_relaxation(species, mdn, params=params)
    bl_stats = run_relaxation(species, bl, params=params_bl)

    volume = params.box_size ** 3
    mdn_visc = green_kubo_viscosity(
        mdn_stats, dt=params.dt, volume=volume,
        equilibration_steps=equilibration_steps, max_lag=max_lag,
    )
    bl_visc = green_kubo_viscosity(
        bl_stats, dt=params.dt, volume=volume,
        equilibration_steps=equilibration_steps, max_lag=max_lag,
    )

    print(f"Equilibrium temperature MDN: {mdn_visc.T_eq:.2f} K")
    print(f"Computed viscosity MDN:      {mdn_visc.viscosity:.6e} Pa·s")
    print(f"Equilibrium temperature BL:  {bl_visc.T_eq:.2f} K")
    print(f"Computed viscosity BL:       {bl_visc.viscosity:.6e} Pa·s")

    return mdn_visc, bl_visc


if __name__ == "__main__":
    main()
