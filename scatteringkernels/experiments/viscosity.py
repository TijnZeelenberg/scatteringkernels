import numpy as np
from tqdm import tqdm
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.dsmc import DSMC_Simulation


def compute_kinetic_shear_stress(
    velocities: np.ndarray, mass: float, volume: float
) -> float:
    """Kinetic contribution: P_xy = (m/V) * sum(v_x * v_y)."""
    return (mass / volume) * np.sum(velocities[:, 0] * velocities[:, 1])


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    centered = series - np.mean(series)
    n = len(centered)

    corr = np.correlate(centered, centered, mode="full")
    corr = corr[n - 1 : n - 1 + max_lag]

    normalization = np.arange(n, n - max_lag, -1)
    return corr / normalization


def estimate_viscosity_green_kubo(
    shear_stress_series: np.ndarray,
    dt: float,
    volume: float,
    temperature: float,
    k_b: float,
    max_lag: int,
):
    c_t = autocorrelation(shear_stress_series, max_lag=max_lag)
    time_lags = np.arange(max_lag) * dt

    integral_c = np.trapezoid(c_t, x=time_lags)
    viscosity = (volume / (k_b * temperature)) * integral_c

    return viscosity, time_lags, c_t


# --- simulation parameters ---
pressure = 1.0  # Pa
box_size = 1e-5  # m
volume = box_size**3  # m^3
trans_temperature = 300.0  # K
rot_temperature = 300.0  # K

# --- run parameters ---
nr_particles = 10_000
nr_cells = 100
dt = 1e-5
nr_steps = 10_000
equilibration_steps = 1_000
max_lag = 2_000

gas_constant = 8.314
n_moles = pressure * volume / (gas_constant * trans_temperature)
mass = n_moles * 2.016e-3 / nr_particles  # effective mass per simulated particle (kg)

print("mass per simulated particle:", mass)


# --- models ---
bl_model = borgnakke_larssen_model(randomseed=42)

dsmc = DSMC_Simulation(random_seed=42)
dsmc.create_box(box_size=box_size)
dsmc.create_grid(x_cells=10, y_cells=10, z_cells=10)
dsmc.create_particles(
    nr_particles=nr_particles,
    mass=mass,
    trans_temperature=trans_temperature,
    rot_temperature=rot_temperature,
    particle_distribution="uniform",
)
dsmc.track_stats(momentum_transfer=False)


shear_stress_series = np.zeros(nr_steps)
trans_temperature_series = np.zeros(nr_steps)

print("Running DSMC for Green-Kubo viscosity...")

for step in tqdm(range(nr_steps)):
    # Move particles
    dsmc.update_positions_and_indices(dt)

    # Select collision pairs
    collision_pairs = dsmc.select_collision_pairs()

    dsmc.perform_collisions(collision_model=bl_model, collision_pairs=collision_pairs)

    # Kinetic contribution (post-collision; correct for dilute-gas DSMC)
    shear_stress_series[step] = compute_kinetic_shear_stress(
        velocities=dsmc.velocities,
        mass=mass,
        volume=volume,
    )

    # Temperature
    trans_energies = 0.5 * mass * np.sum(dsmc.velocities**2, axis=1)
    trans_temperature_series[step] = np.mean(trans_energies) / (1.5 * dsmc._kB)


# --- Production phase ---
production_shear = shear_stress_series[equilibration_steps:]
production_temperature = np.mean(trans_temperature_series[equilibration_steps:])


viscosity, time_lags, autocorr = estimate_viscosity_green_kubo(
    shear_stress_series=production_shear,
    dt=dt,
    volume=volume,
    temperature=production_temperature,
    k_b=dsmc._kB,
    max_lag=max_lag,
)


print("\n--- Green-Kubo viscosity estimate ---")
print(f"Production translational temperature: {production_temperature:.3f} K")
print(f"Mean shear stress: {np.mean(production_shear):.6e} Pa")
print(f"Std shear stress: {np.std(production_shear):.6e} Pa")
print(f"Estimated viscosity: {viscosity:.6e} Pa*s")


results = {
    "shear_stress_series": shear_stress_series,
    "autocorrelation": autocorr,
    "time_lags": time_lags,
    "estimated_viscosity": viscosity,
}
