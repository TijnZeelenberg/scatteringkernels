"""Run energy-relaxation simulations for all batch-size MDN models and save results."""

import numpy as np

import paths
from experiments.energy_relaxation import SimulationParams, load_mdn, run_relaxation
from physics.species import Species

batch_sizes = [750]

nr_steps = 800

params = SimulationParams(nr_steps=nr_steps)
species = Species.H2()

out_dir = paths.ensure_dir(paths.DATA_DIR / "ml-dsmc" / "mdn" / "h2" / "batch_size")

dtype = np.dtype([("timestep", float), ("T_trans_mean", float), ("T_rot_mean", float)])

for bs in batch_sizes:
    model_path = f"results/h2/models/mdn/batch_size/Erelmax10000/b1_6/n_epochs300/mdn_H2_bs{bs}.pth"
    print(f"Running relaxation for batch size {bs}...")
    model = load_mdn(model_path, randomseed=params.randomseed)
    stats = run_relaxation(species, model, params=params)

    arr = np.empty(len(stats["timestep"]), dtype=dtype)
    arr["timestep"] = stats["timestep"]
    arr["T_trans_mean"] = stats["T_trans_mean"]
    arr["T_rot_mean"] = stats["T_rot_mean"]

    out = out_dir / f"relaxation_bs{bs}.npy"
    np.save(out, arr)
    print(f"Saved {out}")
