from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn import MixtureDensityNetwork
import numpy as np
import matplotlib.pyplot as plt
from config.plotting_config import PlottingConfig
from config.experiment_config import ExperimentConfig

plotconfig = PlottingConfig()
experiment_config = ExperimentConfig()

# --- simulation parameters ---
randomseed = 1
pressure = 1  # Pa
box_size = 7.5e-6  # m
volume = box_size**3  # m^3
dt = 1e-5
nr_steps = 200
trans_temperature = 220  # K
rot_temperature = 220  # K
mass = 2.016e-3 / 6.022e23  # kg, mass of one H2 molecule

kB = 1.380649e-23  # J/K
N_sim = 20000  # number of simulated particles
N_real = 20000  # number of real molecules in the box
n_mdn = N_real / volume  # number of real molecules per simulated particle
d_H2 = 2.9e-10

# --- set up collision model ---
bl = borgnakke_larssen_model(randomseed=randomseed)
mdn = MixtureDensityNetwork(
    input_dim=3,
    output_dim=2,
    num_mixtures=5,
    hidden_dim=experiment_config.hidden_dim,
    randomseed=40,
)
mdn.load_model("results/models/mdn_H2H2.pth")

# --- set up DSMC simulation ---
mdn_dsmc = DSMC_Simulation(random_seed=randomseed)
mdn_dsmc.create_box(box_size=box_size)
mdn_dsmc.create_grid(x_cells=5, y_cells=5, z_cells=5)
mdn_dsmc.create_particles(
    N_sim=N_sim,
    N_real=N_real,
    mass=mass,
    d=d_H2,
    trans_temperature=trans_temperature,
    rot_temperature=rot_temperature,
)
bl_dsmc = DSMC_Simulation(random_seed=randomseed)
bl_dsmc.create_box(box_size=box_size)
bl_dsmc.create_grid(x_cells=10, y_cells=10, z_cells=10)
bl_dsmc.create_particles(
    N_sim=N_sim,
    N_real=N_real,
    mass=mass,
    d=d_H2,
    trans_temperature=trans_temperature,
    rot_temperature=rot_temperature,
)

# Run simulation with both models
mdn_dsmc.run_simulation(
    nr_steps=nr_steps,
    dt=dt,
    collision_model=mdn,
)
mdn_stats = mdn_dsmc.get_stats()

bl_dsmc.run_simulation(
    nr_steps=nr_steps,
    dt=dt,
    collision_model=bl,
)
bl_stats = bl_dsmc.get_stats()

# --- compute viscosity via Green-Kubo ---
equilibration_steps = 50
max_lag = 100  # max lag time for ACF (in steps)

# Discard equilibration period
Pxy_mdn = mdn_stats["Pxy"][equilibration_steps:]
Pxz_mdn = mdn_stats["Pxz"][equilibration_steps:]
Pyz_mdn = mdn_stats["Pyz"][equilibration_steps:]

Pxy_bl = bl_stats["Pxy"][equilibration_steps:]
Pxz_bl = bl_stats["Pxz"][equilibration_steps:]
Pyz_bl = bl_stats["Pyz"][equilibration_steps:]

# Subtract means (should be ~0 at equilibrium, helps numerically)
Pxy_mdn -= np.mean(Pxy_mdn)
Pxz_mdn -= np.mean(Pxz_mdn)
Pyz_mdn -= np.mean(Pyz_mdn)

Pxy_bl -= np.mean(Pxy_bl)
Pxz_bl -= np.mean(Pxz_bl)
Pyz_bl -= np.mean(Pyz_bl)

n_mdn = len(Pxy_mdn)
n_bl = len(Pxy_bl)


def autocorrelation_fft(signal, max_lag):
    """Compute normalized autocorrelation using FFT."""
    n = len(signal)
    # Zero-pad to avoid circular correlation artifacts
    padded = np.zeros(2 * n)
    padded[:n] = signal
    fft = np.fft.rfft(padded)
    acf_full = np.fft.irfft(fft * np.conj(fft))[:max_lag]
    # Normalize by number of overlapping points at each lag
    counts = np.arange(n, n - max_lag, -1)
    return acf_full / counts


def plot_acf(acf_xy, acf_xz, acf_yz, acf_avg, dt, max_lag, viscosity, T_eq):
    """Plot the stress autocorrelation functions and cumulative Green-Kubo integral."""
    lags = np.arange(max_lag) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- ACF plot ---
    ax1.plot(lags * 1e6, acf_xy / acf_avg[0], alpha=0.4, label="Pxy")
    ax1.plot(lags * 1e6, acf_xz / acf_avg[0], alpha=0.4, label="Pxz")
    ax1.plot(lags * 1e6, acf_yz / acf_avg[0], alpha=0.4, label="Pyz")
    ax1.plot(lags * 1e6, acf_avg / acf_avg[0], "k-", lw=2, label="Average")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("Normalized ACF")
    ax1.set_title(f"Stress Autocorrelation (T_eq = {T_eq:.1f} K)")
    ax1.legend()

    # --- Cumulative integral (running viscosity) ---
    prefactor = volume / (kB * T_eq)
    running_viscosity = prefactor * np.cumsum(acf_avg) * dt
    ax2.plot(lags * 1e6, running_viscosity * 1e6, "k-", lw=2)
    ax2.axhline(
        viscosity * 1e6, color="r", ls="--", lw=1, label=f"Final: {viscosity:.2e} Pa·s"
    )
    ax2.axhline(8.9, color="g", ls="--", lw=1, label="Reference H₂: 8.9e-6 Pa·s")
    ax2.set_xlabel("Lag time (μs)")
    ax2.set_ylabel("η (μPa·s)")
    ax2.set_title("Cumulative Green-Kubo Viscosity")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/acf_viscosity.png", dpi=150)
    plt.show()


acf_xy_mdn = autocorrelation_fft(Pxy_mdn, max_lag)
acf_xz_mdn = autocorrelation_fft(Pxz_mdn, max_lag)
acf_yz_mdn = autocorrelation_fft(Pyz_mdn, max_lag)

acf_xy_bl = autocorrelation_fft(Pxy_bl, max_lag)
acf_xz_bl = autocorrelation_fft(Pxz_bl, max_lag)
acf_yz_bl = autocorrelation_fft(Pyz_bl, max_lag)

# Average the three independent estimates
acf_avg_mdn = (acf_xy_mdn + acf_xz_mdn + acf_yz_mdn) / 3.0
acf_avg_bl = (acf_xy_bl + acf_xz_bl + acf_yz_bl) / 3.0

# Green-Kubo: eta = (V / kB T) * integral_0^inf <Pxy(0) Pxy(t)> dt
T_eq_mdn = float(np.mean(mdn_stats["T_trans_mean"][equilibration_steps:]))
viscosity_mdn = (volume / (kB * T_eq_mdn)) * float(np.trapezoid(acf_avg_mdn, dx=dt))
T_eq_bl = float(np.mean(bl_stats["T_trans_mean"][equilibration_steps:]))
viscosity_bl = (volume / (kB * T_eq_bl)) * float(np.trapezoid(acf_avg_bl, dx=dt))

print(f"Equilibrium temperature mdn: {T_eq_mdn:.2f} K")
print(f"Computed viscosity mdn: {viscosity_mdn:.6e} Pa·s")
print(f"Equilibrium temperature bl: {T_eq_bl:.2f} K")
print(f"Computed viscosity bl: {viscosity_bl:.6e} Pa·s")

# plot_acf(acf_xy_mdn, acf_xz_mdn, acf_yz_mdn, acf_avg_mdn, dt, max_lag, viscosity_mdn, T_eq_mdn)
