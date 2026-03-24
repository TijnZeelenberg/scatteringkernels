import numpy as np
import matplotlib.pyplot as plt
from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn import MixtureDensityNetwork
from visualization.plot import plot_energy_relaxation

# --- simulation parameters ---
pressure = 1  # Pa
box_size = 7.5e-6  # m
volume = box_size**3  # m^3
dt = 1e-9  # time step (s)
nr_steps = 7000  # number of time steps
trans_temperature = 300  # K
rot_temperature = 100  # K
gas_constant = 8.314  # J/(mol*K)
kB = 1.380649e-23  # J/K
mass = 3.34e-27  # mass of H2 molecule (kg)
d_H2 = 2.89e-10  # effective diameter of H2 (m)

nr_particles = 10000  # number of simulated particles
n_real = pressure / (kB * trans_temperature)  # number density (1/m^3)
n_sim = nr_particles / volume  # simulated number density (1/m^3)
Fn = n_real / n_sim  # scaling factor for collision frequency

# --- set up collision model ---
bl = borgnakke_larssen_model(randomseed=42)
mdn = MixtureDensityNetwork(
    input_dim=3, output_dim=2, num_mixtures=5, hidden_dim=128, randomseed=42
)
mdn.load_model("results/models/mdn_H2H2.pth")

# --- set up DSMC simulation ---
dsmc = DSMC_Simulation(random_seed=42)
dsmc.create_box(box_size=box_size)
dsmc.create_grid(x_cells=10, y_cells=10, z_cells=10)
dsmc.create_particles(
    nr_particles=nr_particles,
    Fn=Fn,
    mass=mass,
    d=d_H2,
    trans_temperature=trans_temperature,
    rot_temperature=rot_temperature,
    particle_distribution="uniform",
)

dsmc.run_simulation(
    nr_steps=nr_steps,
    dt=dt,
    collision_model=bl,
)

# --- plot energy relaxation ---
stats = dsmc.get_stats()
plot_energy_relaxation(stats)

# --- compute viscosity via Green-Kubo ---
equilibration_steps = 1000
max_lag = 200

# Discard equilibration period
Pxy = stats["Pxy"][equilibration_steps:]
Pxz = stats["Pxz"][equilibration_steps:]
Pyz = stats["Pyz"][equilibration_steps:]

# Subtract means (should be ~0 at equilibrium, helps numerically)
Pxy -= np.mean(Pxy)
Pxz -= np.mean(Pxz)
Pyz -= np.mean(Pyz)

n = len(Pxy)


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
    plt.savefig("results/plots/acf_viscosity.png", dpi=150)
    plt.show()


acf_xy = autocorrelation_fft(Pxy, max_lag)
acf_xz = autocorrelation_fft(Pxz, max_lag)
acf_yz = autocorrelation_fft(Pyz, max_lag)

# Average the three independent estimates
acf_avg = (acf_xy + acf_xz + acf_yz) / 3.0

# Green-Kubo: eta = (V / kB T) * integral_0^inf <Pxy(0) Pxy(t)> dt
T_eq = float(np.mean(stats["T_trans_mean"][equilibration_steps:]))
viscosity = (volume / (kB * T_eq)) * float(np.trapezoid(acf_avg, dx=dt))

print(f"Equilibrium temperature: {T_eq:.2f} K")
print(f"Computed viscosity: {viscosity:.6e} Pa·s")

plot_acf(acf_xy, acf_xz, acf_yz, acf_avg, dt, max_lag, viscosity, T_eq)
