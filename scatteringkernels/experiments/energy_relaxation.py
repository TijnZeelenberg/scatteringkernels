from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn import MixtureDensityNetwork
from visualization.plot import plot_energy_relaxation

# --- simulation parameters ---
pressure = 1  # Pa
box_size = 7.5e-6  # m
volume = box_size**3  # m^3
trans_temperature = 300  # K
rot_temperature = 100  # K
gas_constant = 8.314  # J/(mol*K)
moles_per_particle = (
    pressure * volume / (gas_constant * trans_temperature)
)  # ideal gas law: n = PV/RT
mass = moles_per_particle * 2.016

# --- set up collision model ---
bl = borgnakke_larssen_model(randomseed=42)
mdn = MixtureDensityNetwork(
    input_dim=3, output_dim=2, num_mixtures=5, hidden_dim=128, randomseed=42
)
mdn.load_model("results/models/mdn_H2H2.pth")

# --- set up DSMC simulation ---
dsmc = DSMC_Simulation(random_seed=42)
dsmc.create_box(box_size=1e-5)
dsmc.create_grid(x_cells=10, y_cells=10, z_cells=10)
dsmc.create_particles(
    nr_particles=10000,
    mass=mass,
    trans_temperature=trans_temperature,
    rot_temperature=rot_temperature,
    particle_distribution="uniform",
)

dsmc.run_simulation(
    nr_steps=5000,
    dt=1e-5,
    collision_model=bl,
)

# --- plot energy relaxation ---
stats = dsmc.get_stats()
plot_energy_relaxation(stats)
