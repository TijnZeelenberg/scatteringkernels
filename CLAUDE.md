# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis project on machine learning models for scattering kernels in rarefied gas dynamics. The goal is to replace physics-based collision models (Borgnakke-Larssen) with ML-trained models (MDN/BetaMDN) within DSMC simulations, while preserving physical conservation laws and matching macroscopic transport properties like viscosity.

## Environment

Python 3.14 virtual environment in `.venv/`. Activate with:
```bash
source .venv/bin/activate
```

No `requirements.txt` or `pyproject.toml` — dependencies are inferred from imports: `torch`, `numpy`, `sklearn`, `scipy`, `matplotlib`, `tqdm`, `numba`, `hyperopt`.

## Common Commands

```bash
# Generate CTC collision dataset (Numba-accelerated, ~400k collisions)
python ctc_adjusted/ctc_h2_multiple_collisions_numba.py

# Train MDN or BetaMDN on a collision dataset
python training/trainer.py
python training/betamdn_trainer.py

# Run weighting-factor sweep (train models with wf ∈ [0.25..7], then run DSMC experiments)
python training/betamdn_wfsweep.py      # trains beta MDN models
python visualization/betamdn_wfsweep.py # runs DSMC experiments with each

# Run energy relaxation validation experiment
python experiments/H2_energy_relaxation.py
python experiments/betamdn_H2_energy_relaxation.py
python experiments/O2_energy_relaxation.py

# Run viscosity validation (Green-Kubo)
python experiments/H2viscosity.py

# Hyperparameter search (hyperopt)
python training/param_optimization.py

# Generate comparison plots (CTC vs MDN vs BetaMDN distributions)
python create_plots.py
```

There is no test suite. Validation is done via DSMC experiments in `experiments/` and visual comparison in `create_plots.py`.

## Architecture

### Data Flow

1. **Data generation**: `ctc_adjusted/ctc_h2_multiple_collisions_numba.py` runs classical trajectory calculations (Lennard-Jones + rigid-rotor H2) to produce collision datasets saved as `.npy` files in `data/`. Each row is `(Etr, Erot_A, Erot_B, Etr', Erot_A', Erot_B')`.

2. **Training**: `training/trainer.py` / `training/betamdn_trainer.py` load a collision dataset, convert raw energies to normalized energy fractions, train the model, and save weights + normalization params to `results/models/`.

3. **Inference**: DSMC calls the collision model's `collide()` or `batch_collide()` method per collision pair. `MixtureDensityNetwork`, `BetaMixtureDensityNetwork`, and `borgnakke_larssen_model` all implement this interface.

4. **Validation**: Experiments in `experiments/` run DSMC with different collision models and compare energy relaxation curves against SPARTA reference data in `data/` or compute viscosity via Green-Kubo autocorrelation.

### Key Modules

**`physics/dsmc.py`** — Core DSMC simulation engine. Uses Enskog-modified NTC collision selection with Carnahan-Starling EOS pair correlation. Implements cell-based spatial partitioning and vectorized collision pair deduplication (each particle collides ≤1 time per timestep). Tracks stress tensor components (Pxy, Pxz, Pyz) for viscosity.

**`machinelearning/mdn.py`** — Gaussian Mixture Density Network (PyTorch). Takes 3D input `(E_total, η_trans, η_rot_A)` and outputs parameters for K Gaussian mixture components over 2D post-collision energy fractions. Input is z-score normalized; output is also normalized and denormalized at sample time. Methods: `forward`, `sample`, `collide`, `batch_collide`, `save_model`, `load_model`.

**`machinelearning/beta_mdn.py`** — Beta Mixture Density Network (PyTorch). Same interface as MDN but uses Beta distributions instead of Gaussians. Beta is naturally bounded to (0, 1), so output fractions need no clipping. Only input is normalized (output is not, since Beta already lives on [0, 1]). Loss functions `beta_mdn_loss` and `beta_mdn_loss_weighted` are module-level functions.

**`physics/borgnakkelarssen_model.py`** — Physics-based collision baseline. Stochastic model that enforces energy conservation; used as reference in all experiments.

**`machinelearning/gmm.py`** — Sklearn-based GMM; simpler baseline for visual distribution comparison.

**`analysis/kl_divergence.py`** — KDE-based KL divergence utility for comparing distributions.

### Physics Conventions

- Input energy fractions: `η_trans = E_trans / E_total`, `η_rot_A = E_rot_A / (E_rot_A + E_rot_B)`
- Output energy fractions: `η_trans' = E_trans' / E_total`, `η_rot_A' = E_rot_A' / (E_rot_A' + E_rot_B')`
- Total energy is conserved: `E_total_pre = E_total_post` (enforced by construction in `collide()`)
- All collision models operate in the center-of-mass frame
- `zrot` = rotational collision number (Z_rot); for H2: `zrot_bl = 1/0.151`, `zrot_mdn = zrot_bl / 3.5`. Always pass to `create_particles()`.

### Weighting Factor (`wf`)

Training samples are weighted by `E_trans^wf` to up-weight high-energy collisions (more physically relevant under NTC collision selection). The sweep `wf ∈ [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7]` is the main hyperparameter study. Models are saved with the `wf` value encoded in the filename (e.g. `beta_mdn_H2_wf4.pth`).

### Output Paths

```
results/models/mdn/weightsensitivity/H2_400000_dataseed42/   # Gaussian MDN models
results/models/beta_mdn/weightsensitivity/H2_400000/         # Beta MDN models
results/models/beta_mdn/H2H2v1.pth                          # single trained Beta MDN
results/plots/                                               # experiment figures
```

### Configuration

- `config/experiment_config.py`: MDN hyperparams — `lr=2e-4`, `batch_size=256`, `num_epochs=200`, `hidden_dim=128`, `num_mixtures=5`, `trainval_split=0.7`, `random_seed=42`
- `config/plotting_config.py`: Figure styling defaults

## Known TODOs (from `todo.md`)

- Change DSMC/MDN numerical precision from float32 to float64
- Define a standard collision model interface for easier model swapping
- Add viscosity extraction method directly to the DSMC class
- Support bulk viscosity via compression waves
- Make DSMC accept SPARTA configuration file format
