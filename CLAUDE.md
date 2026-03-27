# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis project on machine learning models for scattering kernels in rarefied gas dynamics. The goal is to replace physics-based collision models (Borgnakke-Larssen) with ML-trained models (MDN/GMM) within DSMC simulations, while preserving physical conservation laws and matching macroscopic transport properties like viscosity.

## Environment

Python 3.14 virtual environment in `.venv/`. Activate with:
```bash
source .venv/bin/activate
```

No `requirements.txt` or `pyproject.toml` — dependencies are inferred from imports: `torch`, `numpy`, `sklearn`, `scipy`, `matplotlib`, `tqdm`.

## Common Commands

```bash
# Train MDN models on collision datasets
python training/trainer.py

# Run energy relaxation validation experiment
python experiments/energy_relaxation.py

# Run viscosity validation experiment (Green-Kubo)
python experiments/viscosity.py

# Generate comparison plots (CTC vs MDN vs GMM distributions)
python create_plots.py

# Legacy standalone DSMC runner
python run_dsmc.py
```

There is no test suite. Validation is done via simulation experiments in `experiments/` and visual comparison in `create_plots.py`.

## Architecture

### Data Flow

1. **Training**: `training/trainer.py` loads CTC collision data (`data/H2H2_collisions.csv`, `data/O2O2_collisions.csv`), normalizes features, trains the MDN, and saves model + normalization params to `results/models/`.

2. **Inference**: DSMC simulation calls the collision model's `collide()` method per collision pair. Both `borgnakke_larssen_model` (physics) and `MixtureDensityNetwork` (ML) implement this interface.

3. **Validation**: Experiments in `experiments/` run DSMC with different collision models and compare energy relaxation curves against SPARTA reference data (`data/sparta_energy_relaxation.dat`) or compute viscosity via Green-Kubo autocorrelation.

### Key Modules

**`physics/dsmc.py`** — Core DSMC simulation engine. Uses Enskog-modified NTC collision selection with Carnahan-Starling EOS pair correlation. Implements cell-based spatial partitioning and vectorized collision pair deduplication (each particle collides ≤1 time per timestep). Tracks stress tensor components (Pxy, Pxz, Pyz) for viscosity.

**`machinelearning/mdn.py`** — Mixture Density Network (PyTorch). Takes 3D input `(E_total, η_trans, η_rot_A)` and outputs parameters for 5 Gaussian mixture components over 2D post-collision energy fractions. Methods: `forward`, `sample`, `collide`, `batch_collide`. Normalization parameters are stored alongside model weights.

**`physics/borgnakkelarssen_model.py`** — Physics-based collision baseline. Deterministic model that enforces momentum and energy conservation analytically.

**`machinelearning/gmm.py`** — Sklearn-based GMM; simpler baseline for comparison against MDN.

### Physics Conventions

- Energy fractions: `η_trans = E_trans / E_total`, `η_rot_A = E_rot_A / E_total` (must satisfy `0 ≤ η ≤ 1`)
- All collision models operate in the center-of-mass frame
- Both models must conserve: total energy (`E_total_pre = E_total_post`) and momentum

### Configuration

- `config/experiment_config.py`: MDN hyperparams (lr=1e-3, batch=64, epochs=100, hidden=128, 5 mixture components), GMM settings
- `config/plotting_config.py`: Figure styling defaults

### Output Paths

Results and plots are saved to `results/` (gitignored). Model weights saved as `results/models/mdn_{molecule}.pth`.

## Known TODOs (from `todo.md`)

- Change DSMC/MDN numerical precision from float32 to float64
- Define a standard collision model interface for easier model swapping
- Add viscosity extraction method directly to the DSMC class
- Support bulk viscosity via compression waves
- Make DSMC accept SPARTA configuration file format
