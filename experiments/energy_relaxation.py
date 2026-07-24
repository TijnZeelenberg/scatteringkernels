"""Reusable energy-relaxation experiment helpers.

The energy-relaxation experiments are all the same shape: set up a DSMC
simulation for a given species, plug in a collision model, run for some number
of steps, and compare T_trans/T_rot against a SPARTA reference trace. The
three species-specific scripts (`H2_energy_relaxation`, `O2_energy_relaxation`,
`betamdn_H2_energy_relaxation`) used to repeat this code verbatim — now they
all call `run_relaxation` here.

Public functions:

    run_relaxation(...)               — set up + run one DSMC simulation
    run_relaxation_comparison(...)    — run multiple models for direct comparison
    load_sparta_reference(path)       — read a SPARTA .dat file
    print_relaxation_table(...)       — final T's and 90% relaxation times
    plot_relaxation_comparison(...)   — single figure with all traces
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import paths
from config.experiment_config import ExperimentConfig
from config.plotting_config import PlottingConfig
from machinelearning.beta_mdn import BetaMixtureDensityNetwork
from machinelearning.mdn import MixtureDensityNetwork
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.collision_logger import CollisionLogger
from physics.dsmc import DSMC_Simulation
from physics.species import Species


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------


@dataclass
class SimulationParams:
    """Common DSMC simulation settings shared across experiments."""

    box_size: float = 1.0e-7  # m
    dt: float = 1.0e-11  # s
    nr_steps: int = 250
    trans_temperature: float = 300.0  # K
    rot_temperature: float = 100.0  # K
    N_sim: int = 20000
    N_real: int = 20000
    grid_cells: tuple[int, int, int] = (5, 5, 5)
    randomseed: int = 42


@dataclass
class RelaxationResult:
    """Bundles everything a downstream plotter or analysis needs."""

    label: str
    stats: dict
    sim: DSMC_Simulation


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_mdn(
    model_path: str | Path, randomseed: int = 1, config: ExperimentConfig | None = None
):
    config = config or ExperimentConfig()
    model = MixtureDensityNetwork(
        input_dim=3,
        output_dim=2,
        num_mixtures=config.num_mixtures,
        hidden_dim=config.hidden_dim,
        randomseed=randomseed,
    )
    model.load_model(str(model_path))
    return model


def load_beta_mdn(
    model_path: str | Path, randomseed: int = 1, config: ExperimentConfig | None = None
):
    config = config or ExperimentConfig()
    model = BetaMixtureDensityNetwork(
        input_dim=3,
        output_dim=2,
        num_mixtures=config.num_mixtures,
        hidden_dim=config.hidden_dim,
        randomseed=randomseed,
    )
    model.load_model(str(model_path))
    return model


def _attach_clamp_counter(model) -> dict:
    """Wrap ``model.sample`` to tally how often the raw eta_tr'/eta_rot' samples
    land outside [0, 1] and get clamped downstream by ``batch_collide`` (mdn.py).

    ``model.sample`` returns the raw, pre-clip values (it only clamps inputs and
    mixture weights, never the outputs), so counting outputs outside [0, 1] here
    reproduces exactly the clamp the collision step applies. Returns a counts
    dict that fills in as the simulation runs; report it with
    ``_print_clamp_rates``.
    """
    counts = {
        "tr_below": 0,
        "tr_above": 0,
        "rot_below": 0,
        "rot_above": 0,
        "total": 0,
    }
    orig_sample = model.sample

    def counting_sample(x):
        out = orig_sample(x)
        arr = out.detach().cpu().numpy()
        counts["tr_below"] += int((arr[:, 0] < 0.0).sum())
        counts["tr_above"] += int((arr[:, 0] > 1.0).sum())
        counts["rot_below"] += int((arr[:, 1] < 0.0).sum())
        counts["rot_above"] += int((arr[:, 1] > 1.0).sum())
        counts["total"] += arr.shape[0]
        return out

    model.sample = counting_sample  # instance attribute shadows the bound method
    return counts


def _print_clamp_rates(counts: dict) -> None:
    """Print the MDN output clamp rates accumulated during a relaxation run."""
    total = counts["total"]
    print("\nMDN output clamp rates (raw samples clipped to [0, 1] by batch_collide)")
    if total == 0:
        print("  no MDN samples were drawn (no inelastic collisions?)")
        return
    print(f"  inelastic MDN samples drawn      = {total:,}")
    for frac, lo, hi in [
        ("eta_tr'", "tr_below", "tr_above"),
        ("eta_rot'", "rot_below", "rot_above"),
    ]:
        n_lo, n_hi = counts[lo], counts[hi]
        print(
            f"  {frac:<9} clamp @ 0 = {n_lo / total:.3%} ({n_lo:,})"
            f"   clamp @ 1 = {n_hi / total:.3%} ({n_hi:,})"
            f"   total = {(n_lo + n_hi) / total:.3%}"
        )


# ---------------------------------------------------------------------------
# Core run + reference loading
# ---------------------------------------------------------------------------


def run_relaxation(
    species: Species,
    collision_model,
    *,
    params: SimulationParams | None = None,
    zrot: float | None = None,
    collision_logger: CollisionLogger | None = None,
) -> dict:
    """Run a single DSMC energy-relaxation simulation.

    Args:
        species: physical parameters for the gas.
        collision_model: anything implementing `.collide()` / `.batch_collide()`.
        params: simulation parameters (defaults to `SimulationParams()`).
        zrot: rotational collision number to use. Defaults to `species.zrot_mdn`
              for ML models and `species.zrot_bl` for the BL model.
        collision_logger: optional `CollisionLogger`. When provided, per-step
            aggregates and snapshot collisions are dumped on `finalize()`.

    Returns:
        The `stats` dict produced by `DSMC_Simulation.run_simulation`.
    """
    params = params or SimulationParams()
    if zrot is None:
        zrot_value: float = (
            species.zrot_bl
            if isinstance(collision_model, borgnakke_larssen_model)
            else species.zrot_mdn
        )
    else:
        zrot_value = float(zrot)

    sim = DSMC_Simulation(random_seed=params.randomseed)
    sim.create_box(box_size=params.box_size)
    sim.create_grid(*params.grid_cells)
    sim.create_particles(
        N_sim=params.N_sim,
        N_real=params.N_real,
        mass=species.mass,
        d=species.diameter,
        trans_temperature=params.trans_temperature,
        rot_temperature=params.rot_temperature,
        zrot=zrot_value,
    )
    sim.run_simulation(
        nr_steps=params.nr_steps,
        dt=params.dt,
        collision_model=collision_model,
        collision_logger=collision_logger,
    )
    return sim.get_stats()


def run_relaxation_comparison(
    species: Species,
    models: dict[str, object],
    *,
    params: SimulationParams | None = None,
    collision_logger_factory: Callable[[str], CollisionLogger | None] | None = None,
) -> dict[str, dict]:
    """Run a relaxation simulation for each (label -> collision_model).

    Args:
        species: physical parameters for the gas.
        models: mapping of label -> collision_model.
        params: simulation parameters shared across runs.
        collision_logger_factory: optional callable `label -> CollisionLogger`
            (or None to skip logging for that label). Each model gets its own
            logger so output paths don't collide.

    Returns a dict mapping the same labels to their `stats` outputs.
    """
    results: dict[str, dict] = {}
    for label, model in models.items():
        print(f"\n--- Running DSMC relaxation: {label} ---")
        logger = collision_logger_factory(label) if collision_logger_factory else None
        results[label] = run_relaxation(
            species, model, params=params, collision_logger=logger
        )
    return results


def load_sparta_reference(path: str | Path) -> dict:
    """Load a SPARTA energy-relaxation .dat file.

    Returns dict with keys: timestep, t, T_trans, T_rot.
    """
    arr = np.loadtxt(str(path), skiprows=2, ndmin=2)
    return {
        "timestep": arr[:, 0],
        "t": arr[:, 1],
        "T_trans": arr[:, 2],
        "T_rot": arr[:, 3],
    }


def load_lammps_reference(path: str | Path) -> dict:
    """Load a LAMMPS energy-relaxation .dat file.

    Same column layout as SPARTA (timestep, t, T_trans, T_rot) but with a
    single-line header.
    """
    arr = np.loadtxt(str(path), skiprows=1, ndmin=2)
    return {
        "timestep": arr[:, 0],
        "t": arr[:, 1],
        "T_trans": arr[:, 2],
        "T_rot": arr[:, 3],
    }


def load_bl_reference(path: str | Path) -> dict:
    """Load a cached BL-DSMC energy-relaxation trace.

    The file is produced by `scripts/generate_bl_dsmc.py` and shares the
    LAMMPS column layout. The returned dict uses the *DSMC* stats convention
    (keys `timestep`, `T_trans_mean`, `T_rot_mean`) so it can be dropped
    straight into the `results` dict consumed by `plot_relaxation_comparison`.
    """
    arr = np.loadtxt(str(path), skiprows=1, ndmin=2)
    return {
        "timestep": arr[:, 1],  # physical time, matches stats["timestep"] semantics
        "T_trans_mean": arr[:, 2],
        "T_rot_mean": arr[:, 3],
    }


# ---------------------------------------------------------------------------
# Reporting + plotting
# ---------------------------------------------------------------------------


def _relaxation_time_95(timesteps, trace, T_initial: float) -> float:
    """Time at which `trace` first reaches 95% of (final - initial)."""
    final = trace[-20:-1].mean()
    threshold = T_initial + 0.95 * (final - T_initial)
    indices = np.where(trace >= threshold)[0]
    if len(indices) == 0:
        return float("nan")
    return float(timesteps[indices[0]])


def print_relaxation_table(
    results: dict[str, dict],
    sparta: dict | None = None,
    lammps: dict | None = None,
    *,
    rot_temperature_initial: float,
    bl: dict | None = None,
    bl_label: str = "BL (ML-DSMC)",
):
    """Print final mean T_trans/T_rot and 90% rotational relaxation time."""
    print("\nFinal mean temperatures:")
    for label, stats in results.items():
        ft = stats["T_trans_mean"][-20:-1].mean()
        fr = stats["T_rot_mean"][-20:-1].mean()
        print(f"  {label}: T_trans = {ft:.2f} K, T_rot = {fr:.2f} K")
    if sparta is not None:
        ft = sparta["T_trans"][-20:-1].mean()
        fr = sparta["T_rot"][-20:-1].mean()
        print(f"  SPARTA: T_trans = {ft:.2f} K, T_rot = {fr:.2f} K")
    if lammps is not None:
        ft = lammps["T_trans"][-20:-1].mean()
        fr = lammps["T_rot"][-20:-1].mean()
        print(f"  LAMMPS: T_trans = {ft:.2f} K, T_rot = {fr:.2f} K")
    if bl is not None:
        ft = bl["T_trans_mean"][-20:-1].mean()
        fr = bl["T_rot_mean"][-20:-1].mean()
        print(f"  {bl_label}: T_trans = {ft:.2f} K, T_rot = {fr:.2f} K")

    print("\nRelaxation times (T_rot reaches 90% of equilibrium):")
    for label, stats in results.items():
        t95 = _relaxation_time_95(
            stats["timestep"], stats["T_rot_mean"], rot_temperature_initial
        )
        print(f"  {label}: {t95:.4e} s")
    if sparta is not None:
        t95 = _relaxation_time_95(sparta["t"], sparta["T_rot"], rot_temperature_initial)
        print(f"  SPARTA: {t95:.4e} s")
    if lammps is not None:
        t95 = _relaxation_time_95(lammps["t"], lammps["T_rot"], rot_temperature_initial)
        print(f"  LAMMPS: {t95:.4e} s")
    if bl is not None:
        t95 = _relaxation_time_95(
            bl["timestep"], bl["T_rot_mean"], rot_temperature_initial
        )
        print(f"  {bl_label}: {t95:.4e} s")


def plot_relaxation_comparison(
    results: dict[str, dict],
    sparta: dict | None,
    *,
    output_path: str | Path,
    sparta_label: str = "BL (SPARTA DSMC)",
    ylim: tuple[float, float] = (20, 450),
    sparta_clip: int | None = None,
    lammps: dict | None = None,
    lammps_label: str = "LAMMPS MD",
    lammps_clip: int | None = None,
    bl: dict | None = None,
    bl_label: str = "BL (ML-DSMC)",
    bl_clip: int | None = None,
):
    """Plot T_trans and T_rot for each model + SPARTA/LAMMPS reference traces.

    Output directory is created automatically.
    """
    pc = PlottingConfig()
    fig, ax = plt.subplots(figsize=pc.figsize)
    for label, stats in results.items():
        ax.plot(
            stats["timestep"], stats["T_trans_mean"], label=rf"$T_{{trans}}$ {label}"
        )
        ax.plot(stats["timestep"], stats["T_rot_mean"], label=rf"$T_{{rot}}$ {label}")

    if sparta is not None:
        clip = slice(None) if sparta_clip is None else slice(0, sparta_clip)
        ax.plot(
            sparta["t"][clip],
            sparta["T_trans"][clip],
            color="red",
            linestyle="--",
            label=rf"$T_{{trans}}$ {sparta_label}",
        )
        ax.plot(
            sparta["t"][clip],
            sparta["T_rot"][clip],
            color="blue",
            linestyle="--",
            label=rf"$T_{{rot}}$ {sparta_label}",
        )

    if lammps is not None:
        clip = slice(None) if lammps_clip is None else slice(0, lammps_clip)
        ax.plot(
            lammps["t"][clip],
            lammps["T_trans"][clip],
            color="red",
            linestyle=":",
            label=rf"$T_{{trans}}$ {lammps_label}",
        )
        ax.plot(
            lammps["t"][clip],
            lammps["T_rot"][clip],
            color="blue",
            linestyle=":",
            label=rf"$T_{{rot}}$ {lammps_label}",
        )

    if bl is not None:
        clip = slice(None) if bl_clip is None else slice(0, bl_clip)
        ax.plot(
            bl["timestep"][clip],
            bl["T_trans_mean"][clip],
            color="red",
            linestyle="-.",
            label=rf"$T_{{trans}}$ {bl_label}",
        )
        ax.plot(
            bl["timestep"][clip],
            bl["T_rot_mean"][clip],
            color="blue",
            linestyle="-.",
            label=rf"$T_{{rot}}$ {bl_label}",
        )

    ax.set_xlabel(
        "Time [s]", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight
    )
    ax.set_ylabel(
        "Temperature [K]", fontsize=pc.label_fontsize, fontweight=pc.label_fontweight
    )
    ax.ticklabel_format(style="sci", scilimits=(-2, 3))
    ax.set_ylim(*ylim)
    ax.grid()
    ax.legend(loc="upper right", fontsize=pc.legend_fontsize, ncol=2)

    output_path = paths.ensure_parent(output_path)
    fig.savefig(output_path, dpi=300)
    return fig, ax
