# ML Scattering Kernels

Master's thesis project on replacing physics-based molecular collision models with machine-learned scattering kernels inside a DSMC (Direct Simulation Monte Carlo) simulation. The target application is rarefied gas dynamics — specifically $H_2$ and $O_2$ at low pressure.

The central question is: can a neural network learn the conditional energy redistribution of a molecular collision from trajectory data, and then act as a drop-in replacement for a phenominological model like the Borgnakke-Larssen physics model, while still producing correct macroscopic transport properties (e.g. viscosity, energy relaxation)?

## How it all fits together

```
ctc_adjusted/            ← generate training data via classical trajectory simulation
    │
    ▼
data/                    ← raw and filtered collision datasets (.npy / .csv)
    │
    ▼
training/                ← train MDN on collision data
    │
    ▼
results/models/          ← saved model weights
    │
    ▼
experiments/             ← validate model inside a full DSMC simulation
```

## Repository layout

### `ctc_adjusted/` — Classical Trajectory Code (data generation)

Simulates individual H₂-H₂ or O₂-O₂ molecular collisions using a Lennard-Jones potential and rigid-rotor dynamics. Each collision is independent and run in parallel. The output is a dataset of `(E_trans_in, E_rot_A_in, E_rot_B_in, E_trans_out, E_rot_A_out, E_rot_B_out)` tuples — the raw training data for the ML models.

- `ctc_h2_multiple_collisions.py` — main simulation script (numpy version)
- `ctc_h2_multiple_collisions_numba.py` — faster Numba-JIT version
- `lj.py`, `get_fij.py`, `get_rdot.py`, `get_vdot.py`, `get_wdot.py` — force/torque/velocity update helpers
- `get_rand_rot_mat.py`, `get_m.py` — geometry utilities
- `dscatter.py` — scattering angle sampling
- `visualize.py` — quick CTC output inspection plots

### `data/` — Collision datasets

Stores the raw CTC output files and SPARTA reference data.

- `H2H2_collisions.csv` / `H2H2_collisions_numba.npy` — H₂ collision datasets
- `O2O2_collisions.npy` / `O2O2_collisions_uniform.npy` — O₂ collision datasets
- `sparta_H2_energy_relaxation.dat` / `sparta_O2_energy_relaxation.dat` — reference DSMC results from SPARTA (used as ground truth in validation experiments)
- `filter_inelastic.py` — script to remove elastic collisions from a dataset before training (elastic collisions carry no information about rotational energy exchange)

### `physics/` — Physics models

- `dsmc.py` — Full DSMC simulation engine. Sets up a particle box with a spatial grid, selects collision pairs via Enskog-modified NTC (No-Time-Counter), and calls a pluggable collision model per pair. Also tracks stress tensor components for viscosity computation via Green-Kubo.
- `borgnakkelarssen_model.py` — The classical physics baseline. Probabilistically redistributes translational and rotational energy at each collision according to the Borgnakke-Larssen model. Implements the same `collide()` interface as the ML models so it can be swapped in/out without changing the DSMC code.

### `machinelearning/` — ML collision models

Both models implement a `collide()` method matching the physics model interface.

- `mdn.py` — **Mixture Density Network** (PyTorch). Takes `(E_total, η_trans, η_rot_A)` as input and outputs Gaussian mixture parameters over post-collision energy fractions `(η_trans_out, η_rot_A_out)`. Normalization parameters are saved alongside model weights. This is the primary ML model.
- `gmm.py` — **Gaussian Mixture Model** (sklearn). Simpler non-conditional baseline: fits a GMM directly to the output distribution and samples from it regardless of input. Useful as a sanity-check lower bound.

### `training/` — Model training

- `param_optimization.py` — Trains the MDN using `hyperopt` for Bayesian hyperparameter search. Loads a collision dataset, converts raw energies to normalized fractions, fits the MDN, and saves trained weights to `results/models/`.
- `trainer.py` — Trains the MDN using `torch` and `torch.nn`. Loads a collision dataset, converts raw energies to normalized fractions, fits the MDN, and saves trained weights to `results/models/`.

### `experiments/` — Validation

Each experiment runs a full DSMC simulation with multiple collision models and compares a macroscopic observable against a reference.

- `H2_energy_relaxation.py` — Starts a gas with mismatched translational and rotational temperatures (300 K vs 100 K) and tracks how they equilibrate over time. Compares Borgnakke-Larssen, MDN, and SPARTA reference data.
- `O2_energy_relaxation.py` — Same experiment for O₂.
- `viscosity.py` — Runs a DSMC simulation in equilibrium and extracts shear viscosity via Green-Kubo integration of the stress autocorrelation function.

### `analysis/` — Analysis utilities

- `kl_divergence.py` — Computes KL divergence between two sets of samples using Gaussian KDE on a shared grid. Used to quantitatively compare model and CTC output distributions.
- `ctc_equilibrium.py` — Checks equilibrium properties of the CTC dataset.

### `visualization/` — Plotting

- `plots.py` — Reusable plot functions (scatter plots, histograms, energy curves) that take a config object and data arrays. Keeps styling consistent across the project.

### `config/` — Configuration

- `experiment_config.py` — MDN hyperparameters (learning rate, batch size, epochs, hidden dim, number of mixture components) and GMM settings.
- `plotting_config.py` — Figure styling defaults (font sizes, colors, DPI).

### `create_plots.py` — Distribution comparison

Top-level script that loads a CTC dataset, samples from both the MDN and GMM, and produces side-by-side comparison plots of the output energy fraction distributions.

### `utils/helpers.py`

Small utilities (dataset loading, normalization helpers).

## Getting started

```bash
# Activate the virtual environment
source .venv/bin/activate

# (Optional) regenerate collision training data
cd ctc_adjusted && python ctc_h2_multiple_collisions_numba.py

# Train the MDN
python training/trainer.py

# Run energy relaxation validation
python experiments/H2_energy_relaxation.py

# Run viscosity validation
python experiments/viscosity.py

# Compare output distributions visually
python create_plots.py
```

Trained model weights and plots are saved to `results/` (gitignored).

## Physics conventions

- All collision models work in the **center-of-mass frame**.
- The ML models operate on normalized **energy fractions**: `η_trans = E_trans / E_total`, `η_rot_A = E_rot_A / (E_rot_A + E_rot_B)`. These are dimensionless, bounded in [0, 1], and independent of the absolute energy scale.
- Total energy and momentum must be conserved by every collision model.

## Open TODOs

See `todo.md` for the full list. Key items still open:

- Increase numerical precision from float32 to float64 throughout DSMC and MDN
- Support bulk viscosity measurement via compression waves
- Make DSMC accept SPARTA configuration file format directly
