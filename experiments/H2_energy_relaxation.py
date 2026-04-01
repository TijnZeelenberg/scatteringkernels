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

# --- plot energy relaxation ---
fig, ax = plt.subplots(figsize=plotconfig.figsize)

ax.plot(
    mdn_stats["timestep"],
    mdn_stats["T_trans_mean"],
    label="translational temp MDN",
)
ax.plot(
    mdn_stats["timestep"],
    mdn_stats["T_rot_mean"],
    label="rotational temp MDN",
)
ax.plot(
    bl_stats["timestep"],
    bl_stats["T_trans_mean"],
    label="translational temp BL",
)
ax.plot(
    bl_stats["timestep"],
    bl_stats["T_rot_mean"],
    label="rotational temp BL",
)

ax.set_xlabel(
    "Time [s]",
    fontsize=plotconfig.label_fontsize,
    fontweight=plotconfig.label_fontweight,
)
ax.ticklabel_format(style="sci", scilimits=(0, 0))
ax.set_ylabel(
    "Energy [K]",
    fontsize=plotconfig.label_fontsize,
    fontweight=plotconfig.label_fontweight,
)
ax.set_title(
    "Energy Relaxation Over Time",
    fontsize=plotconfig.title_fontsize,
    fontweight=plotconfig.title_fontweight,
)
# Add energy relaxation plot from SPARTA
DATA = np.loadtxt("data/sparta_H2_energy_relaxation.dat", skiprows=2)

timestep_sparta = DATA[:, 0]
t_sparta = DATA[:, 1]
T_trans_sparta = DATA[:, 2]
T_rot_sparta = DATA[:, 3]

ax.plot(
    t_sparta,
    T_trans_sparta,
    label="translational temp SPARTA",
    color="red",
    linestyle="--",
)
ax.plot(
    t_sparta, T_rot_sparta, label="rotational temp SPARTA", color="blue", linestyle="--"
)

ax.legend(fontsize=plotconfig.legend_fontsize)
fig.savefig("results/plots/H2_energy_relaxation.png", dpi=300)
plt.show()
