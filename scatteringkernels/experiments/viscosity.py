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

gas_constant = 8.314
n_moles = pressure * volume / (gas_constant * trans_temperature)
mass = n_moles * 2.016e-3 / nr_particles  # effective mass per simulated particle (kg)


# --- models ---
bl_model = borgnakke_larssen_model(randomseed=42)
mdn_model = MixtureDensityNetwork(
    input_dim=3, output_dim=2, num_mixtures=5, hidden_dim=128, randomseed=42
)
mdn_model.load_model("results/models/mdn_H2H2.pth")

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
dsmc.run_simulation(nr_steps=7000, dt=1e-5, collision_model=bl_model)

stats = dsmc.get_stats()
