# ML Scattering Kernels

Master's thesis project on replacing physics-based molecular collision models with
machine-learned scattering kernels inside a DSMC (Direct Simulation Monte Carlo)
simulation. The target application is rarefied gas dynamics — specifically H₂ and O₂
at low pressure.

The central question: can a neural network learn the conditional energy redistribution
of a molecular collision from trajectory data, act as a drop-in replacement for the
Borgnakke-Larssen phenomenological model, and still produce correct macroscopic
transport properties (viscosity, energy relaxation)?

## Pipeline

```
ctc_adjusted/ctc_h2.py           classical trajectory simulation (Numba, parallel)
    │                            → data/ctc/**/*.npy  columns:
    │                              (Etr, Er1, Er2, Etr', Er1', Er2')  [K]
    ▼
training/trainer.py              train the Gaussian MDN
training/betamdn_trainer.py      train the Beta MDN
    │                            → results/models/**/*.pth
    ▼
experiments/H2_energy_relaxation.py    validate inside a full DSMC against SPARTA/LAMMPS
experiments/H2viscosity.py             Green-Kubo shear viscosity
analysis/kernel_stationarity.py        equilibrium/detailed-balance diagnostics
    │
    ▼
visualization/                   figures for the thesis report
    │
    ▼
results/plots/                   figures
```

## Setup

```bash
uv venv
uv pip install -r requirements.txt      # or: uv sync
```

For a CUDA build of PyTorch, install it from the matching wheel index:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Run everything from the project root, so that top-level modules (`paths`, `physics`,
`machinelearning`, …) resolve:

```bash
uv run python experiments/H2_energy_relaxation.py
```

## What is and is not in the repository

Tracked: all source code, the SPARTA and LAMMPS input decks and their reference output
(`sparta/`, `lammps/`), and the Slurm job scripts (`hpc/`).

**Not tracked** (see `.gitignore`): `data/` and `results/`. Collision datasets are
hundreds of MB and trained weights are regenerable (since fixed seeds are used throughout the training loops), so a fresh clone has neither. The
experiment scripts default to dataset and model paths under those directories — expect
a `FileNotFoundError` until you generate or copy them in. Regenerate with the data
generation and training steps below.

## Entry points

There is no central CLI. Each script is run directly and is configured by editing the
constants at the top of its `__main__` block or the keyword defaults of its `main()`.
This is deliberate for a research codebase where the parameters being swept change
weekly.

### Generate collision data

```bash
uv run python ctc_adjusted/ctc_h2.py
```

Configuration lives in the settings block at the top of the file: `ncoll`, `seed`,
`dist` (`uniform` | `mb` | `ntc`), `T_eq`, `E_rel_max`, `bfac`. Writes to
`data/ctc/H2/impactparam/`. On 16 cores, roughly 3 h for 10⁶ collisions — see
`hpc/run_data_generation.sh`.

`ctc_adjusted/ctc_h2_impactparamsweep.py` is the same simulation swept over impact
parameter.

### Train a model

```bash
uv run python training/trainer.py           # Gaussian MDN
uv run python training/betamdn_trainer.py   # Beta MDN
```

Both are thin wrappers over `training.core.train_collision_model`, which takes
`kind`, `datapath`, `outputpath`, `epochs`, `batch_size`, `lr`, `wf`, `patience`.
Dataset and output paths are set in each wrapper's `__main__`. CUDA is used
automatically when available. GPU job script for use on the TU/e High Performance Cluster: `hpc/run_training.sh`.

`wf` applies polynomial importance weighting `w ∝ E_trans^wf`; leave it at `None` for
an unweighted NLL.

### Validate inside DSMC

```bash
uv run python experiments/H2_energy_relaxation.py   # H2 relaxation vs BL, SPARTA, LAMMPS
uv run python experiments/O2_energy_relaxation.py   # same for O2
uv run python experiments/H2viscosity.py            # Green-Kubo shear viscosity
uv run python experiments/DSMC_validation.py        # DSMC engine sanity checks
```

`main()` in each takes the model path, reference data paths, step count, seed, and
species parameters as keyword arguments.

### Analysis

```bash
uv run python analysis/kernel_stationarity.py     # is the kernel stationary at equilibrium?
uv run python analysis/compare_collision_logs.py  # compare per-collision DSMC logs
uv run python analysis/lammps_zrot.py             # fit 1/Z_rot from LAMMPS output
uv run python analysis/ctc_equilibrium.py         # CTC dataset equilibrium check
```

`kernel_stationarity.py` hardcodes the model paths and `T_EQ` it probes; edit the
`MDN_MODELS` dict at the top.

## Tests

```bash
uv run pytest
```

`tests/test_conservation.py` asserts that every collision model conserves pair momentum
and total redistributable energy, and pins the scalar and vectorized Borgnakke-Larssen
paths to the same redistribution law.

## Repository layout

```
physics/
    dsmc.py                     DSMC engine: NTC collision selection, cell grid,
                                stress-tensor tracking
    borgnakkelarssen_model.py   Borgnakke-Larssen baseline collision model
    species.py                  Species dataclass (H2, O2): mass, diameter, zrot
    collision_logger.py         per-collision diagnostics, including the fraction of
                                inputs outside the model's training range

machinelearning/
    mdn.py                      Gaussian Mixture Density Network (PyTorch)
    beta_mdn.py                 Beta Mixture Density Network — bounded output
    gmm.py                      sklearn GMM baseline

training/
    core.py                     train_collision_model() — one entry point, both models
    data_prep.py                dataset loading, weighting, time-reversal augmentation
    trainer.py                  runnable wrapper for the Gaussian MDN
    betamdn_trainer.py          runnable wrapper for the Beta MDN
    parametersweeps/            impact-parameter sweep

experiments/
    energy_relaxation.py        shared relaxation harness: run, tabulate, plot
    viscosity.py                Green-Kubo viscosity and autocorrelation
    H2_energy_relaxation.py     H2 experiment
    O2_energy_relaxation.py     O2 experiment
    H2viscosity.py              H2 viscosity experiment
    DSMC_validation.py          engine sanity checks

analysis/
    kernel_stationarity.py      equilibrium marginal and conditional-drift diagnostics
    ctc_equilibrium.py          CTC dataset equilibrium check
    compare_collision_logs.py   diff two DSMC collision logs
    kl_divergence.py            KDE-based KL divergence
    lammps_zrot.py              fit 1/Z_rot from LAMMPS relaxation output

ctc_adjusted/                   Classical Trajectory Code — collision data generation.
                                Adapted from a previous master's student's code; see
                                "Provenance" below.
    ctc_h2.py                   main simulation
    ctc_h2_impactparamsweep.py  impact-parameter sweep
    lj.py get_fij.py get_rdot.py get_wdot.py get_vdot.py get_m.py
                                force, torque and geometry helpers

sparta/                         SPARTA input decks and reference output (DSMC ground truth)
lammps/                         LAMMPS input decks and reference output (MD ground truth)
hpc/                            Slurm job scripts (TU/e cluster)
config/                         model hyperparameters and figure styling

visualization/                  all plotting code for the thesis report — every figure
                                in the written report is produced here
    plot.py                     energy relaxation, loss history, density scatter,
                                histogram comparisons (CTC vs MDN vs GMM)
    create_plots.py             impact-parameter sweep loss curves

tests/                          conservation invariants
paths.py                        central output-path helpers; creates results/ on demand
```

## Physics conventions

- All collision models operate in the **center-of-mass frame**.
- Model inputs are normalized energy fractions:
  - `η_trans = E_trans / E_total`
  - `η_rot_A = E_rot_A / (E_rot_A + E_rot_B)`
  where `E_total` is the redistributable energy (relative kinetic + rotational); the
  center-of-mass kinetic energy never enters the pool.
- Outputs are post-collision fractions of the same conserved `E_total`, in [0, 1].
- Total energy and pair momentum are conserved by construction in every `collide()` and
  `batch_collide()` implementation, and asserted in `tests/`.
- `zrot` (rotational collision number) sets the fraction of collisions that exchange
  rotational energy, via an inelastic probability `1/zrot`.
- CTC datasets store energies as `E / k_B` in Kelvin. The DSMC works in Joules and the
  models convert at their boundary.

## The collision-model interface

Every collision model — the Borgnakke-Larssen baseline, the Gaussian MDN, the Beta MDN —
exposes the same pair of methods:

```python
collide(velocity_i, e_rot_i, velocity_j, e_rot_j, m, zrot=1.0)
batch_collide(velocity_i, e_rot_i, velocity_j, e_rot_j, m, zrot=1.0)
```

`physics/dsmc.py` dispatches on `hasattr(collision_model, "batch_collide")` and
otherwise falls back to the scalar path, so the engine never depends on which model it
is holding and swapping models is a one-line change in an experiment script. The
interface is duck-typed rather than a formal `Protocol`; making it explicit is on the
list below.

## Provenance

`ctc_adjusted/` is a previous master's student's classical trajectory integrator,
adapted for this project (hence "adjusted"). Everything else is written for this thesis.
Parts of the codebase were written with AI assistance — principally the path handling
(`paths.py`), dataset preparation (`training/data_prep.py`, `training/core.py`) and the
logging layer (`physics/collision_logger.py`).

## Known limitations and TODOs

- Many interfaces are duck-typed rather than formalized; the collision model interface is one example.
- The DSMC cannot read SPARTA configuration files directly, so settings are maintained
  in two places.
- An abstraction has not been made for the experiment scripts; the H2 and O2 experiments are copy-paste variants of the same harness.
- Bulk viscosity (via compression waves) is not implemented.
