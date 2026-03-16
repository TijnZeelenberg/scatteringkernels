import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn_model import MixtureDensityNetwork
from config.experiment_config import ExperimentConfig


def _fit_shear_rate(bin_centers: np.ndarray, uy: np.ndarray, exclude: int) -> tuple[float, float]:
    """Fit u_y(x) = slope * x + intercept in the bulk; returns (slope, intercept)."""
    if exclude * 2 >= len(bin_centers):
        raise ValueError("exclude too large for number of bins")
    x = bin_centers[exclude:-exclude]
    y = uy[exclude:-exclude]
    # Remove NaNs from empty bins.
    mask = np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("Not enough populated bins to fit shear rate")
    slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
    return float(slope), float(intercept)


def _run_couette_viscosity(
    *,
    collision_model,
    rng_seed: int,
    nr_particles: int,
    molecules_per_particle: int,
    mass: float,
    temperature: float,
    box_size: float,
    nr_cells: int,
    dt: float,
    n_steps: int,
    U: float,
    equil_steps: int,
    sample_every: int,
    n_bins: int,
    exclude_bins: int,
    collision_probability: float,
    check_conservation: bool,
    progress_desc: str,
    show_progress: bool,
) -> dict:
    sim = DSMC_Simulation(random_seed=rng_seed)
    sim.initialize_domain(box_size=box_size, nr_cells=nr_cells, boundary="diffuse")
    sim.initialize_particles(
        nr_particles=nr_particles,
        molecules_per_particle=molecules_per_particle,
        mass=mass,
        temperature=temperature,
        particle_distribution="uniform",
    )

    # Couette: walls move tangentially in y with +/- U/2.
    sim.configure_diffuse_walls(
        wall_temperature=temperature,
        wall_velocity_left=np.array([0.0, -0.5 * U, 0.0], dtype=np.float32),
        wall_velocity_right=np.array([0.0, +0.5 * U, 0.0], dtype=np.float32),
    )

    bin_edges = np.linspace(0.0, box_size, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    uy_sum = np.zeros(n_bins, dtype=np.float64)
    counts_sum = np.zeros(n_bins, dtype=np.float64)

    def _progress(it, desc: str):
        if not show_progress:
            return it
        return tqdm(it, desc=desc, leave=False)

    # Run to steady state.
    for _ in _progress(range(equil_steps), f"{progress_desc}: equilibrating"):
        sim.update_positions(dt)
        pairs = sim.select_collision_pairs(collision_probability=collision_probability)
        sim.perform_collisions(collision_model, pairs, check_conservation=check_conservation)

    # Reset wall stats and start sampling.
    sim.reset_wall_stats()

    for step in _progress(range(n_steps - equil_steps), f"{progress_desc}: sampling"):
        sim.update_positions(dt)
        pairs = sim.select_collision_pairs(collision_probability=collision_probability)
        sim.perform_collisions(collision_model, pairs, check_conservation=check_conservation)

        if (step % sample_every) != 0:
            continue

        # Bin-average u_y(x)
        inds = np.clip(np.digitize(sim.positions[:, 0], bin_edges) - 1, 0, n_bins - 1)
        np.add.at(uy_sum, inds, sim.velocities[:, 1].astype(np.float64))
        np.add.at(counts_sum, inds, 1.0)

    uy = np.full(n_bins, np.nan, dtype=np.float64)
    nonzero = counts_sum > 0
    uy[nonzero] = uy_sum[nonzero] / counts_sum[nonzero]

    shear_rate, intercept = _fit_shear_rate(bin_centers, uy, exclude_bins)

    tau_left, tau_right = sim.wall_shear_stress_xy()
    # Use the average magnitude of the two wall stresses.
    tau = 0.5 * (abs(tau_left) + abs(tau_right))

    eta = float(tau / abs(shear_rate)) if shear_rate != 0.0 else float("nan")

    return {
        "eta": eta,
        "tau_left": float(tau_left),
        "tau_right": float(tau_right),
        "tau": float(tau),
        "shear_rate": float(shear_rate),
        "fit_intercept": float(intercept),
        "uy": uy,
        "bin_centers": bin_centers,
        "wall_hits_left": int(sim.wall_hits_left),
        "wall_hits_right": int(sim.wall_hits_right),
        "wall_sampling_time": float(sim.wall_sampling_time),
    }


############################
# User-editable parameters #
############################

# MDN model checkpoint
MDN_MODEL_PATH = "results/models/mdn_H2H2.pth"

# Simulation parameters
BASE_SEED = 42
N_SEEDS = 5

# Choose a target rarefied-gas state via pressure and domain length.
# With H2 at 300 K and BOX_SIZE ~ 1 mm, pressures ~ O(10 Pa) correspond to Kn ~ O(1).
TEMPERATURE = 300.0
BOX_SIZE = 1e-3
TARGET_PRESSURE_PA = 10.0

# Number of simulated particles (computational knob).
NR_PARTICLES = 20000

# Molecules represented by each simulated particle (set from TARGET_PRESSURE_PA).
# This keeps density/pressure physical while NR_PARTICLES stays manageable.
_kB = 1.380649e-23
_V = BOX_SIZE**3
_n = TARGET_PRESSURE_PA / (_kB * TEMPERATURE)
_N_real = _n * _V
MOLECULES_PER_PARTICLE = int(max(1, round(_N_real / NR_PARTICLES)))
NR_CELLS = 50
DT = 4e-7

# Couette parameters (shear in y, gradient in x)
U_WALL_DIFFERENCE = 200.0  # m/s, right wall - left wall

# Runtime/estimation controls
N_STEPS = 2000
EQUIL_STEPS = 500
SAMPLE_EVERY = 10
N_BINS = 60
EXCLUDE_BINS = 6

# Collision selection (NOTE: your pairing is a simplified placeholder, so treat results as qualitative)
COLLISION_PROBABILITY = 0.1

# Diagnostics
CHECK_CONSERVATION = False
SHOW_PROGRESS = True
SHOW_PLOT = True


def _build_mdn() -> MixtureDensityNetwork:
    config = ExperimentConfig()
    mdn = MixtureDensityNetwork(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        num_mixtures=config.num_mixtures,
        hidden_dim=config.hidden_dim,
        randomseed=config.random_seed,
    )
    mdn.load_model(MDN_MODEL_PATH)
    return mdn


def _build_bl() -> borgnakke_larssen_model:
    # RNG inside BL is used for sampling internal redistribution.
    return borgnakke_larssen_model(rng=np.random.default_rng(0))


def _molecular_mass_h2() -> float:
    # Molecular mass in kg for one H2 molecule.
    # 2.016 g/mol = 2.016e-3 kg/mol
    return float(2.016e-3 / 6.022e23)


mass = _molecular_mass_h2()
mdn = _build_mdn()
bl = _build_bl()

etas_bl: list[float] = []
etas_mdn: list[float] = []
last_res_bl: dict | None = None
last_res_mdn: dict | None = None

seeds = [BASE_SEED + i for i in range(N_SEEDS)]
for run_idx, seed in enumerate(seeds):
    # Make BL randomness depend on the run seed too.
    bl.rng = np.random.default_rng(seed)

    res_bl = _run_couette_viscosity(
        collision_model=bl,
        rng_seed=seed,
        nr_particles=NR_PARTICLES,
        molecules_per_particle=MOLECULES_PER_PARTICLE,
        mass=mass,
        temperature=TEMPERATURE,
        box_size=BOX_SIZE,
        nr_cells=NR_CELLS,
        dt=DT,
        n_steps=N_STEPS,
        U=U_WALL_DIFFERENCE,
        equil_steps=EQUIL_STEPS,
        sample_every=SAMPLE_EVERY,
        n_bins=N_BINS,
        exclude_bins=EXCLUDE_BINS,
        collision_probability=COLLISION_PROBABILITY,
        check_conservation=CHECK_CONSERVATION,
        progress_desc=f"BL (seed {seed}) [{run_idx+1}/{N_SEEDS}]",
        show_progress=SHOW_PROGRESS,
    )
    res_mdn = _run_couette_viscosity(
        collision_model=mdn,
        rng_seed=seed,
        nr_particles=NR_PARTICLES,
        molecules_per_particle=MOLECULES_PER_PARTICLE,
        mass=mass,
        temperature=TEMPERATURE,
        box_size=BOX_SIZE,
        nr_cells=NR_CELLS,
        dt=DT,
        n_steps=N_STEPS,
        U=U_WALL_DIFFERENCE,
        equil_steps=EQUIL_STEPS,
        sample_every=SAMPLE_EVERY,
        n_bins=N_BINS,
        exclude_bins=EXCLUDE_BINS,
        collision_probability=COLLISION_PROBABILITY,
        check_conservation=CHECK_CONSERVATION,
        progress_desc=f"MDN (seed {seed}) [{run_idx+1}/{N_SEEDS}]",
        show_progress=SHOW_PROGRESS,
    )

    etas_bl.append(float(res_bl["eta"]))
    etas_mdn.append(float(res_mdn["eta"]))
    last_res_bl = res_bl
    last_res_mdn = res_mdn

etas_bl_arr = np.array(etas_bl, dtype=np.float64)
etas_mdn_arr = np.array(etas_mdn, dtype=np.float64)

print("Couette viscosity estimate (wall-flux / fitted gradient)")
print(f"Seeds: {seeds}")
print("--- Borgnakke–Larsen ---")
print(f"eta mean = {etas_bl_arr.mean():.6e} Pa*s")
print(f"eta std  = {etas_bl_arr.std(ddof=1):.6e} Pa*s")
print("--- MDN kernel ---")
print(f"eta mean = {etas_mdn_arr.mean():.6e} Pa*s")
print(f"eta std  = {etas_mdn_arr.std(ddof=1):.6e} Pa*s")

if last_res_bl is not None and last_res_mdn is not None:
    print("--- Last-run details (for debugging) ---")
    print(f"BL: shear_rate={last_res_bl['shear_rate']:.6e} 1/s, tau_left={last_res_bl['tau_left']:.6e} Pa, tau_right={last_res_bl['tau_right']:.6e} Pa")
    print(f"MDN: shear_rate={last_res_mdn['shear_rate']:.6e} 1/s, tau_left={last_res_mdn['tau_left']:.6e} Pa, tau_right={last_res_mdn['tau_right']:.6e} Pa")

if SHOW_PLOT and last_res_bl is not None and last_res_mdn is not None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)

    def _plot_profile(res: dict, label: str):
        x = res["bin_centers"]
        uy = res["uy"]
        ax.plot(x, uy, marker="o", markersize=3, linewidth=1.0, label=f"{label}: binned $u_y(x)$")

        slope = res["shear_rate"]
        b = res["fit_intercept"]
        x_fit = x[EXCLUDE_BINS:-EXCLUDE_BINS]
        y_fit = slope * x_fit + b
        ax.plot(x_fit, y_fit, linewidth=2.0, label=f"{label}: linear fit")

    _plot_profile(last_res_bl, "BL")
    _plot_profile(last_res_mdn, "MDN")

    ax.axvspan(0.0, BOX_SIZE * (EXCLUDE_BINS / N_BINS), color="k", alpha=0.06)
    ax.axvspan(BOX_SIZE * (1.0 - EXCLUDE_BINS / N_BINS), BOX_SIZE, color="k", alpha=0.06)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("$u_y$ [m/s]")
    ax.set_title("Couette flow: velocity profile and bulk fit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()
