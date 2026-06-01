"""Run energy-relaxation simulations for all impact-parameter MDN models and save results."""

import numpy as np

import paths
from experiments.energy_relaxation import SimulationParams, load_mdn, run_relaxation
from physics.species import Species

b_facs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
nr_steps = 500
params = SimulationParams(nr_steps=nr_steps)
species = Species.O2()

out_dir = paths.ensure_dir(paths.DATA_DIR / "ml-dsmc" / "mdn" / "o2" / "impactparam")

dtype = np.dtype([("timestep", float), ("T_trans_mean", float), ("T_rot_mean", float)])

for bfac in b_facs:
    bfac_tag = str(bfac).replace(".", "_")
    model_path = f"results/o2/models/mdn/impactparam/mdn_O2_b{bfac_tag}.pth"
    model = load_mdn(model_path, randomseed=params.randomseed)
    stats = run_relaxation(species, model, params=params)

    arr = np.empty(len(stats["timestep"]), dtype=dtype)
    arr["timestep"] = stats["timestep"]
    arr["T_trans_mean"] = stats["T_trans_mean"]
    arr["T_rot_mean"] = stats["T_rot_mean"]

    out = out_dir / f"relaxation_b{bfac_tag}.npy"
    np.save(out, arr)
    print(f"Saved {out}")
