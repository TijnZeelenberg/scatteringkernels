"""Generate a cached BL-DSMC energy-relaxation trace.

The H2 experiment used to re-run the Borgnakke-Larssen DSMC simulation every
time, which is wasted compute when the only thing changing between runs is
the MDN model under test. This script runs the BL reference once and writes
its output to a .dat file in the same column layout as LAMMPS/SPARTA, so
`experiments/H2_energy_relaxation.py` can load it instead of re-running it.

Output columns: `timestep simtime Ttrans Trot` (single header row).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataclasses import replace

import paths
from experiments.energy_relaxation import SimulationParams, run_relaxation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.species import Species

# Classical LJ/rigid-rotor calibration — must match H2_energy_relaxation.py.
_D_CLASSICAL_MD = 10.1e-10
_ZROT_CLASSICAL_MD = 5.0


def main():
    p = argparse.ArgumentParser(
        description="Run BL-DSMC and cache the energy-relaxation trace to a .dat file."
    )
    p.add_argument("--species", type=str, default="H2", choices=["H2", "O2"])
    p.add_argument("--output", type=str, default="data/bl_H2_energy_relaxation.dat")
    p.add_argument("--box-size", type=float, default=1.0e-7, help="cubic box edge [m]")
    p.add_argument("--dt", type=float, default=1.0e-12, help="DSMC timestep [s]")
    p.add_argument("--nr-steps", type=int, default=1000)
    p.add_argument("--N", type=int, default=20000, help="number of simulated particles")
    p.add_argument("--trans-temperature", type=float, default=3000.0)
    p.add_argument("--rot-temperature", type=float, default=1000.0)
    p.add_argument("--grid-cells", type=int, nargs=3, default=(5, 5, 5))
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    base = Species.H2() if args.species == "H2" else Species.O2()
    species = replace(
        base,
        diameter=_D_CLASSICAL_MD,
        zrot_bl=_ZROT_CLASSICAL_MD,
        zrot_mdn=_ZROT_CLASSICAL_MD,
    )
    params = SimulationParams(
        nr_steps=args.nr_steps,
        trans_temperature=args.trans_temperature,
        rot_temperature=args.rot_temperature,
        randomseed=args.seed,
        grid_cells=tuple(args.grid_cells),  # type: ignore[arg-type]
        box_size=args.box_size,
        dt=args.dt,
        N_sim=args.N,
        N_real=args.N,
    )
    model = borgnakke_larssen_model(randomseed=args.seed)
    print(
        f"Running BL-DSMC: {species.name}, box={args.box_size:.2e} m, "
        f"dt={args.dt:.2e} s, nr_steps={args.nr_steps}"
    )
    stats = run_relaxation(species, model, params=params)

    out_path = paths.ensure_parent(Path(args.output))
    step_idx = np.arange(args.nr_steps)
    table = np.column_stack(
        [
            step_idx,
            stats["timestep"],
            stats["T_trans_mean"],
            stats["T_rot_mean"],
        ]
    )
    np.savetxt(
        out_path,
        table,
        header="timestep simtime Ttrans Trot",
        comments="",
        fmt=["%d", "%.6e", "%.6e", "%.6e"],
    )
    print(f"Wrote {len(table)} rows to {out_path}")


if __name__ == "__main__":
    main()
