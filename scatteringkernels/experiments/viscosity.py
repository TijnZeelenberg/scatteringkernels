import numpy as np
from tqdm import tqdm
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from physics.dsmc import DSMC_Simulation
from machinelearning.mdn import MixtureDensityNetwork

# --- simulation parameters ---
pressure = 1.0  # Pa
box_size = 1e-5  # m
volume = box_size**3  # m^3
trans_temperature = 300.0  # K
rot_temperature = 300.0  # K

# --- run parameters ---
nr_particles = 10_000
nr_cells = 100
equilibration_steps = 1_000
max_lag = 2_000
kB = 1.380649e-23
dt = 1e-5
gas_constant = 8.314
n_moles = pressure * volume / (gas_constant * trans_temperature)
mass = n_moles * 2.016e-3 / nr_particles  # effective mass per simulated particle (kg)


# --- models ---
bl = borgnakke_larssen_model(randomseed=42)
mdn = MixtureDensityNetwork(
    input_dim=3, output_dim=2, num_mixtures=5, hidden_dim=128, randomseed=42
)
mdn.load_model("results/models/mdn_H2H2.pth")

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

print("Running DSMC for Green-Kubo viscosity...")
dsmc.run_simulation(nr_steps=7000, dt=dt, collision_model=bl)

stats = dsmc.get_stats()

# --- compute viscosity via Green-Kubo ---

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

# For reference: H2 at 300K ~ 8.9e-6 Pa·s
