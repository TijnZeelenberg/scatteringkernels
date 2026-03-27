from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn import MixtureDensityNetwork
from visualization.plot import plot_energy_relaxation
import numpy as np

# --- simulation parameters ---
pressure = 1  # Pa
box_size = 7.5e-6  # m
volume = box_size**3  # m^3
dt = 1e-7
nr_steps = 20000
trans_temperature = 300  # K
rot_temperature = 100  # K
mass = 2.016e-3 / 6.022e23  # kg, mass of one H2 molecule

kB = 1.380649e-23  # J/K
N_sim = 10000  # number of simulated particles
N_real = 10000  # number of real molecules in the box
n = N_real / volume  # number of real molecules per simulated particle
d_H2 = 2.9e-10

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
    N_sim=N_sim,
    N_real=N_real,
    mass=mass,
    d=d_H2,
    trans_temperature=trans_temperature,
    rot_temperature=rot_temperature,
)

dsmc.run_simulation(
    nr_steps=nr_steps,
    dt=dt,
    collision_model=bl,
)

# --- plot energy relaxation ---
stats = dsmc.get_stats()
fig, ax = plot_energy_relaxation(stats)

# add energy relaxation plot from SPARTA
DATA = np.loadtxt("../data/sparta_energy_relaxation.dat", skiprows=2)
