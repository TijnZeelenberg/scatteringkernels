# ML Scattering Kernels

Master's thesis project on replacing physics-based molecular collision models with machine-learned scattering kernels inside a DSMC (Direct Simulation Monte Carlo) simulation. The target application is rarefied gas dynamics — specifically H₂ and O₂ at low pressure.

The central question is: can a neural network learn the conditional energy redistribution of a molecular collision from trajectory data, and act as a drop-in replacement for the Borgnakke-Larssen phenomenological model, while still producing correct macroscopic transport properties (viscosity, energy relaxation)?

## Pipeline overview

```
ctc_adjusted/                    generate collision data (classical trajectory, Numba)
    │
    ▼
data/*.npy                       (Etr, Erot_A, Erot_B, Etr', Erot_A', Erot_B') per row
    │
    ▼
scripts/run_pipeline.py train    train MDN or Beta MDN
    │
    ▼
results/models/                  saved .pth weights
    │
    ▼
scripts/run_pipeline.py relaxation / viscosity    validate inside full DSMC
    │
    ▼
results/plots/                   figures
```

## Getting started

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Generate collision training data (~5 min for 400k, ~15 min for 1M on 16 cores)
python ctc_adjusted/ctc_h2_multiple_collisions_numba.py

# 3. Train a model
python scripts/run_pipeline.py train \
    --kind mdn \
    --dataset data/H2H2_collisions_numba_b1_0_Etr20k_Erot15k_400000_seed42.npy \
    --T-eq 2200

# 4. Run an energy-relaxation experiment
python scripts/run_pipeline.py relaxation \
    --species H2 \
    --model-kind mdn \
    --model results/models/mdn/mdn_run.pth \
    --include-bl \
    --sparta data/sparta_H2_energy_relaxationVHS_zinv0151.dat \
    --output results/plots/H2_relaxation.png
```

Trained model weights and plots are saved to `results/` (gitignored). The directory is created automatically on first use.

## CLI reference — `scripts/run_pipeline.py`

All workflows are exposed as sub-commands of a single entry point. Run any command with `--help` for the full option list.

### `train` — train a single model

```
python scripts/run_pipeline.py train
    --kind {mdn,beta_mdn}          model architecture
    --dataset PATH                 .npy collision dataset
    [--output PATH]                output .pth path (default: results/models/<kind>/<kind>_run.pth)
    [--epochs N]                   max training epochs (default: 100)
    [--batch-size N]               mini-batch size (default: 128)
    [--lr FLOAT]                   Adam learning rate (default: 2e-4)
    [--T-eq FLOAT]                 equilibrium temperature [K] for NTC importance weighting
                                   (principled — overrides --wf when set)
    [--wf FLOAT]                   legacy polynomial weighting exponent (default: 1.0)
    [--patience N]                 early-stopping patience in epochs (default: 30)
    [--showplots]                  show loss curve after training
```

`--T-eq` is the recommended weighting mode. It applies the exact NTC importance ratio
`w ∝ √E_trans · exp(−E_total / T_eq)`, which matches the collision-energy distribution
DSMC feeds the kernel at equilibrium temperature `T_eq`. Use `--wf` only for legacy
sweep comparisons.

### `wf-sweep` — train across weighting factors

```
python scripts/run_pipeline.py wf-sweep
    --kind {mdn,beta_mdn}
    --dataset PATH
    --tag TAG                      label for the output directory (e.g. H2_400000)
    [--trainseed N]                training random seed; models go into trainseed<N>/ subdir
    [--epochs N]  [--batch-size N]  [--lr FLOAT]  [--patience N]
```

Trains wf ∈ {0.25, 0.5, 1, 2, 3, 4, 5, 6, 7} in sequence. Output:
`results/models/<kind>/weightsensitivity/<tag>/`.

### `wf-sweep-eval` — validate a whole sweep

```
python scripts/run_pipeline.py wf-sweep-eval
    --kind {mdn,beta_mdn}
    --tag TAG
    --sparta PATH                  SPARTA reference .dat file
    [--trainseed N]
    [--species {H2,O2}]
```

Runs an energy-relaxation DSMC for every model in the sweep and writes a 3×3
comparison figure to `results/plots/`.

### `relaxation` — energy-relaxation experiment

```
python scripts/run_pipeline.py relaxation
    [--species {H2,O2}]            (default: H2)
    [--model PATH]                 trained .pth model (omit to run BL only)
    [--model-kind {mdn,beta_mdn}]
    [--include-bl]                 also run the Borgnakke-Larssen reference
    [--sparta PATH]                SPARTA reference .dat file
    [--trans-T FLOAT]              initial translational temperature [K] (default: 300)
    [--rot-T FLOAT]                initial rotational temperature [K] (default: 100)
    [--nr-steps N]                 DSMC timesteps (default: 100)
    [--randomseed N]
    [--output PATH]                save comparison figure to this path
```

### `viscosity` — Green-Kubo shear viscosity

```
python scripts/run_pipeline.py viscosity
    --model PATH
    [--species {H2,O2}]
    [--model-kind {mdn,beta_mdn}]
    [-T FLOAT]                     equilibrium temperature [K] (default: 220)
    [--nr-steps N]                 DSMC steps (default: 200)
    [--equilibration N]            steps to discard before ACF (default: 50)
    [--max-lag N]                  ACF lag cutoff (default: 100)
    [--randomseed N]
```

Prints `T_eq` and `viscosity [Pa·s]` to stdout.

## Running on the HPC cluster (TU/e)

Pre-written Slurm scripts live in `hpc/`. Submit from the project root after
cloning the repo to the cluster and creating a virtualenv (`~/ctc_env`):

```bash
# First-time setup on the cluster
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv ~/ctc_env
source ~/ctc_env/bin/activate
pip install numpy numba tqdm pandas matplotlib torch

# Data generation (16 CPU cores, ~3 h for 1M collisions)
sbatch hpc/run_data_generation.sh

# Model training (1 GPU, ~1-2 h for 100 epochs)
# Edit the configuration block at the top of the script before submitting.
sbatch hpc/run_training.sh

# Monitor
squeue --me
```

Training automatically uses CUDA if a GPU is present; falls back to CPU otherwise.

## Repository layout

```
ctc_adjusted/           Classical Trajectory Code — data generation
    ctc_h2_multiple_collisions_numba.py   main simulation (Numba, parallel)
    ctc_h2_multiple_collisions.py         slower NumPy reference version
    lj.py  get_fij.py  get_rdot.py ...   force/torque/geometry helpers

data/                   Collision datasets and reference data
    *.npy               CTC output: (Etr, Erot_A, Erot_B, Etr', Erot_A', Erot_B')
    sparta_*.dat        SPARTA reference DSMC results (ground truth)
    bl_H2_energy_relaxation.dat   cached BL-DSMC run (regen: scripts/generate_bl_dsmc.py)

physics/
    dsmc.py             DSMC simulation engine (Enskog NTC, stress tensor tracking)
    species.py          Species dataclass (H2, O2) — mass, diameter, zrot
    borgnakkelarssen_model.py   physics collision baseline (same interface as ML models)

machinelearning/
    mdn.py              Gaussian Mixture Density Network (PyTorch)
    beta_mdn.py         Beta Mixture Density Network (PyTorch, naturally bounded output)
    gmm.py              sklearn GMM baseline

training/
    core.py             train_collision_model() — single entry point for both architectures
    data_prep.py        dataset loading, NTC importance weighting, time-reversal augmentation
    trainer.py          thin __main__ wrapper for MDN
    betamdn_trainer.py  thin __main__ wrapper for Beta MDN
    wfsweep.py          run_wf_sweep() — full wf ∈ {0.25..7} sweep

experiments/
    energy_relaxation.py   run_relaxation, run_relaxation_comparison, plot_relaxation_comparison
    viscosity.py           green_kubo_viscosity, plot_acf

visualization/
    wfsweep.py          run_wf_sweep_experiments — DSMC + figure for every model in a sweep

scripts/
    run_pipeline.py     CLI entry point (see CLI reference above)
    generate_bl_dsmc.py regenerate the cached BL-DSMC reference

hpc/
    run_data_generation.sh   Slurm: CTC data on 16 CPU cores
    run_training.sh          Slurm: MDN training on 1 GPU

analysis/
    kl_divergence.py    KDE-based KL divergence for distribution comparison
    lammps_zrot.py      fit 1/Z_rot from LAMMPS relaxation output

lammps/                 LAMMPS input files and output (classical MD reference)
sparta/                 SPARTA input files and output (DSMC reference)

config/
    experiment_config.py    MDN hyperparameters (lr, batch_size, hidden_dim, ...)
    plotting_config.py      figure styling defaults

paths.py                Central output-path helpers (auto-creates results/ on demand)
```

## Physics conventions

- All collision models operate in the **center-of-mass frame**.
- Inputs to the ML models are normalized energy fractions:
  - `η_trans = E_trans / E_total`
  - `η_rot_A = E_rot_A / (E_rot_A + E_rot_B)`
- Outputs are post-collision fractions of the same conserved `E_total`:
  - `η_trans'`, `η_rot_A'` ∈ [0, 1]
- Total energy is conserved by construction in every `collide()` implementation.
- `zrot` (rotational collision number Z_rot) controls the fraction of collisions
  that exchange rotational energy. All collision models share the same `collide()`
  and `batch_collide()` interface, making them drop-in swappable inside the DSMC.

## Open TODOs

- Increase numerical precision from float32 to float64 throughout DSMC and MDN
- Support bulk viscosity measurement via compression waves
- Make DSMC accept SPARTA configuration file format directly
- Define a formal collision model interface (Protocol/ABC) for cleaner model swapping
