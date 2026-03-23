# Copilot Instructions for `scatteringkernels`

## Build, test, and lint commands

This repository currently uses script-based workflows (no `pytest`, lint, or build config files are present).

- Run MDN training on both datasets:
  - `python training/trainer.py`
- Run DSMC simulation entrypoint:
  - `python run_dsmc.py`
- Run energy-relaxation experiment:
  - `python experiments/energy_relaxation.py`
- Generate comparison plots (CTC vs MDN vs GMM):
  - `python create_plots.py`

Single-run equivalent (closest to “single test” in this repo):

- Run one targeted experiment script:
  - `python experiments/energy_relaxation.py`

## High-level architecture

The project has two connected tracks:

1. Physics simulation track:
   - `physics/dsmc.py` implements `DSMC_Simulation` (box/grid creation, particle initialization, timestepping, collisions, energy history collection).
   - Collision behavior is injected via a model object with `postsample(...)` (e.g. `physics/borgnakkelarssen_model.py`).
   - `run_dsmc.py` and `experiments/energy_relaxation.py` are executable entry scripts that wire DSMC + collision model + plotting.

2. Data-driven scattering-kernel track:
   - `training/trainer.py` trains `machinelearning/mdn_model.py` on CSV collision datasets and saves model artifacts to `results/models/`.
   - `create_plots.py` loads a trained MDN, fits `machinelearning/gmm_model.py`, samples both, computes KL metric (`analysis/kl_divergence.py`), and plots outputs (`visualization/plot.py`).
   - `utils/helpers.py` contains CSV-to-feature transformation used for model inputs/outputs.

Configuration is centralized in:
- `config/experiment_config.py` for model/training parameters.
- `config/plotting_config.py` for figure styling and save paths.

## Key conventions in this codebase

- **Collision-model interface contract**: DSMC expects a collision model object exposing:
  - `postsample(velocity_i, e_rot_i, velocity_j, e_rot_j, m, T) -> (new_v_i, new_e_rot_i, new_v_j, new_e_rot_j)`.
  Keep this exact signature/return shape for compatibility with `DSMC_Simulation.perform_collisions`.

- **Energy-fraction feature convention**: both training and inference use transformed variables, not raw CSV columns:
  - Inputs: `[E_total, eta_trans, eta_rot_A]`
  - Outputs: post-collision fractions `[eta_trans_post, eta_rot_A_post]`
  Preserve this transformation logic when adding datasets or models.

- **Model persistence convention for MDN**:
  - Save not only `state_dict` but also normalization tensors (`input_mean/std`, `output_mean/std`) in the same file.
  - Sampling assumes normalization state exists (`load_model(...)` or prior training before `sample(...)`).

- **Config-driven defaults**:
  - Use `ExperimentConfig` and `PlottingConfig` values instead of hardcoding hyperparameters or styling in new scripts.

- **Output locations are part of workflow**:
  - Trained models go to `results/models/`.
  - Generated figures go to `results/plots/`.
  Downstream scripts assume these paths.
