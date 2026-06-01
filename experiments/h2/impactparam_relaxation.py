"""Run energy-relaxation simulations for all impact-parameter MDN models and save results."""

import numpy as np

import paths
from experiments.energy_relaxation import SimulationParams, load_mdn, run_relaxation
from physics.species import Species

b_facs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

nr_steps = 350

params = SimulationParams(nr_steps=nr_steps)
species = Species.H2()

out_dir = paths.ensure_dir(paths.DATA_DIR / "ml-dsmc" / "mdn" / "h2" / "impactparam")

dtype = np.dtype([("timestep", float), ("T_trans_mean", float), ("T_rot_mean", float)])

for bfac in b_facs:
    bfac_tag = str(bfac).replace(".", "_")
    model_path = (
        f"results/h2/models/mdn/impactparam/Erelmax10000/mdn_H2_b{bfac_tag}.pth"
    )
    model = load_mdn(model_path, randomseed=params.randomseed)
    stats = run_relaxation(species, model, params=params)

    arr = np.empty(len(stats["timestep"]), dtype=dtype)
    arr["timestep"] = stats["timestep"]
    arr["T_trans_mean"] = stats["T_trans_mean"]
    arr["T_rot_mean"] = stats["T_rot_mean"]

    out = out_dir / f"relaxation_b{bfac_tag}.npy"
    np.save(out, arr)
    print(f"Saved {out}")
