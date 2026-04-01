from physics.dsmc import DSMC_Simulation


from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn import MixtureDensityNetwork
from visualization.plot import plot_energy_relaxation
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
nr_steps = 100
trans_temperature = 300  # K
rot_temperature = 100  # K
mass = 2.016e-3 / 6.022e23  # kg, mass of one H2 molecule

kB = 1.380649e-23  # J/K
N_sim = 20000  # number of simulated particles
N_real = 20000  # number of real molecules in the box
n = N_real / volume  # number of real molecules per simulated particle
d_H2 = 2.9e-10

mdn = DSMC_Simulation(random_seed=randomseed)
mdn.create_box(box_size=box_size)
mdn.create_grid(x_cells=5, y_cells=5, z_cells=5)
mdn.create_particles(
    N_sim=N_sim,
    N_real=N_real,
    mass=mass,
    d=d_H2,
    trans_temperature=trans_temperature,
    rot_temperature=rot_temperature,
)
mdn.update_cell_indices()

collisions, vrmax = mdn.calculate_no_collisions(dt=dt)
collision_pairs = mdn.select_collision_pairs(dt=dt)

print(f"Total collisions: {np.sum(collisions)}")
