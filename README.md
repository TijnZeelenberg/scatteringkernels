# ML Scattering Kernels

Master's thesis project on replacing the phenomenological Borgnakke-Larssen collision
model inside a DSMC (Direct Simulation Monte Carlo) rarefied-gas simulation with a
mixture density network trained on classical trajectory data. The pipeline is three
stages: generate molecular collision data with a classical trajectory simulation, fit a
neural network to it, then run the DSMC with that network as its collision model.

## Setup

```bash
uv sync
```

For a CUDA build of PyTorch:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Run everything from the project root and **as a module** (`-m`). Running a script by its
path puts the script's own directory on `sys.path` instead of the project root, and the
top-level imports (`paths`, `physics`, `machinelearning`) then fail.

## Run it

A small CTC dataset and an MDN trained on that dataset are committed under `examples/`, so a fresh
clone can run the pipeline without generating any data first. Train a model on that
dataset and drop it straight into a DSMC simulation:

```bash
uv run python -m scripts.run_example
```

Takes about 30 seconds on an intel i7 CPU and prints:

```
=== Training an MDN on examples/ ===
Time-reversal augmentation: 5000 -> 10000 rows
Training device: cpu
Training complete. Best validation loss: 1.5071 at epoch 199
  train NLL: 3.128 -> 1.421
    val NLL: 3.148 -> 1.508

=== DSMC energy relaxation ===
  2000 particles, 50 steps, T_trans 300 K / T_rot 100 K at t=0
  Borgnakke-Larssen    T_trans  293.7 ->  226.9 K   T_rot   95.8 ->  196.1 K   energy drift 0.0e+00
  MDN                  T_trans  293.7 ->  208.2 K   T_rot   95.8 ->  224.0 K   energy drift 0.0e+00

  equipartition temperature for 3+2 DOF: 220.0 K
```

Both the Borgnakke-Larssen and MDN model relax the translational and rotational temperatures towards the
same equipartition value while conserving total energy exactly — the MDN faster, which
is what the trajectory data says should happen.

`--dsmc-only` skips training and uses the committed model instead:

```bash
uv run python -m scripts.run_example --dsmc-only
```

### About the example data

```
examples/ctc_H2_ncoll5000.npy    5000-collision CTC dataset
examples/mdn_H2_ncoll5000.pth    Gaussian MDN trained on it
```

The dataset is produced by `ctc_adjusted/ctc_h2.py` itself, using the settings that
module ships with, at 5000 collisions instead of 10⁶. Regenerate both with
`uv run python -m scripts.make_example_data` (~90 s on intel i7 cores).

These exist to make the code runnable, not to reproduce a result. 5000 collisions is
three orders of magnitude smaller than the thesis datasets.

## The full pipeline

`data/` and `results/` are untracked — collision datasets run to hundreds of MB and
trained weights are regenerable from fixed seeds. There is no central CLI: each script
is run directly and configured by editing the constants at the top of its `__main__`
block, which suits a research codebase where the parameters being swept change weekly.

```bash
# 1. generate collision data  (~3 h for 10⁶ collisions on 16 cores)
uv run python -m ctc_adjusted.ctc_h2

# 2. train a kernel
uv run python -m training.trainer           # Gaussian MDN
uv run python -m training.betamdn_trainer   # Beta MDN

# 3. validate it inside a full DSMC against SPARTA and LAMMPS references
uv run python -m experiments.H2_energy_relaxation
uv run python -m experiments.O2_energy_relaxation
uv run python -m experiments.H2viscosity            # Green-Kubo shear viscosity
uv run python -m experiments.DSMC_validation        # engine sanity checks

# diagnostics
uv run python -m analysis.kernel_stationarity       # is the kernel stationary at equilibrium?
uv run python -m analysis.compare_collision_logs
uv run python -m analysis.lammps_zrot               # fit 1/Z_rot from LAMMPS output
uv run python -m analysis.ctc_equilibrium
```

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

ctc_adjusted/                   Classical Trajectory Code — collision data generation
    ctc_h2.py                   main simulation (Numba, parallel)
    ctc_h2_impactparamsweep.py  impact-parameter sweep
    lj.py get_fij.py get_rdot.py get_wdot.py get_vdot.py get_m.py
                                force, torque and geometry helpers

visualization/                  all plotting code for the thesis report — every figure
                                in the written report is produced here

scripts/
    run_example.py              train + simulate on the committed example data
    make_example_data.py        regenerate the artifacts in examples/

examples/                       committed dataset and model, so a fresh clone runs
sparta/  lammps/                input decks and reference output (ground truth)
hpc/                            Slurm job scripts (TU/e cluster)
config/                         model hyperparameters and figure styling
tests/                          conservation invariants
paths.py                        central output-path helpers; creates results/ on demand
```

## The collision-model interface

Every collision model — the Borgnakke-Larssen baseline, the Gaussian MDN, the Beta MDN —
exposes the same pair of methods:

```python
collide(velocity_i, e_rot_i, velocity_j, e_rot_j, m, zrot=1.0)
batch_collide(velocity_i, e_rot_i, velocity_j, e_rot_j, m, zrot=1.0)
```

Swapping models is a one-line change in an experiment script.

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
  rotational energy, via an inelastic probability `1/zrot`. Each collision model has its own value of zrot.
- CTC datasets store energies as `E / k_B` in Kelvin. The DSMC works in Joules and the
  models convert at their boundary.

## Provenance

`ctc_adjusted/` is based on a CTC integrator by Benjamin Vollebregt, adapted from its original matlab implementation to Python and Numba.
Parts of the codebase were written with AI assistance: principally the path handling (`paths.py`), the plotting (`visualization/`) and the logging layer (`physics/collision_logger.py`).

## Possible improvements

- **Formalize the collision-model interface.** It is duck-typed via `hasattr`, which is
  what let the two Borgnakke-Larssen code paths silently diverge on their Beta exponents;
  a `Protocol` would have caught that at type-check time rather than in a conservation
  test written afterwards.
- **Vectorize `calculate_no_collisions`.** Finding `vrmax` is an O(N²) Python loop within
  each cell, and the one hot path in the engine that is not vectorized. Bird's method —
  carrying a running `vrmax` per cell across timesteps — is the standard fix.
- **Take configuration out of Python constants.** Each script is configured by editing its
  `__main__` block; `SimulationParams` shows what a config object looks like, but the DSMC
  engine's staged builder (`create_box` → `create_grid` → `create_particles`) predates it.
- **Abstract the experiment scripts.** The H2 and O2 experiments are copy-paste variants
  of the same harness.
- **Read SPARTA configuration directly.** Simulation settings are currently maintained in
  two places, in the SPARTA input decks and in the Python experiment scripts.
- **Enforce detailed balance structurally in the MDN kernels.** It is currently encouraged by time-reversal
  augmentation of the training set and a loss penalty, rather than guaranteed by the
  model's form.
- **Implement bulk viscosity computation** via compression waves.
