# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis research on **scattering kernel modeling for rarefied gas dynamics** using machine learning and molecular dynamics simulation. The goal is to train ML models (MDN/GMM) on classical trajectory calculation (CTC) collision data and use them as kernels in Direct Simulation Monte Carlo (DSMC) simulations, then validate against physical observables like viscosity and energy relaxation.

## Repository Structure

- **scatteringkernels/**: Main Python research package — ML models, DSMC simulation, training, analysis
- **ctc_benjamin/**: Python port of CTC code for H2-H2 collision data generation
- **data_aldo/**: Aldo Frezzotti's original Fortran CTC code for N2/O2 collision data

## Running Code

There is no build system. Scripts are run directly:

```bash
# Run DSMC simulation
python scatteringkernels/run_dsmc.py

# Train MDN model
python scatteringkernels/training/trainer.py

# Viscosity experiment (Green-Kubo method)
python scatteringkernels/experiments/viscosity.py

# Energy relaxation experiment
python scatteringkernels/experiments/energy_relaxation.py

# Generate plots
python scatteringkernels/create_plots.py
```

Compile the Fortran CTC code (requires Intel Fortran compiler `ifx`):
```bash
bash data_aldo/Compilation/build_exe.sh
```

## Architecture

### Physics Pipeline

```
CTC collision data (ctc_benjamin/ or data_aldo/)
  → CSV dataset (scatteringkernels/data/)
  → ML model training (MDN/GMM)
  → DSMC simulation using trained kernel
  → Physical validation (viscosity, energy relaxation)
```

### Key Modules

**`physics/dsmc.py`** — `DSMC_Simulation` class: box/grid initialization, particle velocity sampling (Maxwell-Boltzmann), cell-based collision pair selection, periodic boundary conditions, momentum transfer and energy history tracking. Supports both monoatomic and diatomic (rotational DOF) molecules.

**`physics/borgnakkelarssen_model.py`** — `borgnakke_larssen_model`: post-collision velocity/energy sampling. Handles center-of-mass frame collision dynamics, energy/momentum conservation, and inelastic collision probability for diatomic molecules (~1/245).

**`machinelearning/mdn_model.py`** — `MixtureDensityNetwork`: PyTorch model (input_dim=3, output_dim=2). Predicts mixture-of-Gaussians parameters (π, μ, σ) for post-collision energy fractions. Includes save/load functionality.

**`machinelearning/gmm_model.py`** — `GaussianMixtureModel`: scikit-learn wrapper, used as baseline comparison to MDN.

**`training/trainer.py`** — Loads H2H2/O2O2 datasets, converts raw energies to fractional form, trains MDN with Adam optimizer, saves to `results/models/`.

**`experiments/viscosity.py`** — Green-Kubo viscosity calculation via shear stress autocorrelation (kinetic + collisional contributions to pressure tensor).

### Data Format

Collision datasets in `scatteringkernels/data/` (CSV):
```
Etr, Erot1_in, Erot2_in, Etr_out, Erot1_out, Erot2_out
```
Model inputs are energy fractions (η = E_component / E_total), not raw energies.

### Configuration

- `config/experiment_config.py`: MDN hyperparameters (5 mixtures, 128 hidden units, lr=1e-3, batch=64), GMM settings, train/val split (0.8)
- `config/plotting_config.py`: Figure sizes, fonts, colormap settings

## Known Issues / TODOs

- DSMC currently uses float32; should be upgraded to float64 for accuracy
- Viscosity extraction via Green-Kubo is work-in-progress (current branch: `viscosity`)
- Bulk viscosity support not yet implemented
