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
randomseed = 2
pressure = 1  # Pa
box_size = 7.5e-6  # m
volume = box_size**3  # m^3
dt = 1e-5
nr_steps =100
trans_temperature = 300  # K
rot_temperature = 100  # K
mass = 2.016e-3 / 6.022e23  # kg, mass of one H2 molecule
zrot_bl = 1 / 0.151
zrot_mdn = zrot_bl/3.5

kB = 1.380649e-23  # J/K
N_sim = 20000  # number of simulated particles
N_real = 20000  # number of real molecules in the box
n = N_real / volume  # number of real molecules per simulated particle
d_H2 = 2.92e-10

# --- set up collision model ---
bl = borgnakke_larssen_model(randomseed=randomseed)
mdn = MixtureDensityNetwork(
    input_dim=3,
    output_dim=2,
    num_mixtures=experiment_config.num_mixtures,
    hidden_dim=experiment_config.hidden_dim,
    randomseed=randomseed,
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
    zrot=zrot_bl
)

# Run simulation with BL model
bl_dsmc.run_simulation(
    nr_steps=nr_steps,
    dt=dt,
    collision_model=bl,
)
bl_stats = bl_dsmc.get_stats()

# --- load SPARTA data ---
spartaVHS = np.loadtxt("data/sparta_H2_energy_relaxationVHS_zinv0151.dat", skiprows=2)
spartaVSS = np.loadtxt("data/sparta_H2_energy_relaxationVSS_zinv0151.dat", skiprows=2)

timestep_spartaVSH = spartaVHS[:, 0]
t_spartaVHS = spartaVHS[:, 1]
T_trans_spartaVHS = spartaVHS[:, 2]
T_rot_spartaVHS = spartaVHS[:, 3]

timestep_spartaVSS = spartaVSS[:, 0]
t_spartaVSS = spartaVSS[:, 1]
T_trans_spartaVSS = spartaVSS[:, 2]
T_rot_spartaVSS = spartaVSS[:, 3]


# print table of mean final temperatures from all models
print("Final mean temperatures:")
print(f"BL: T_trans = {bl_stats['T_trans_mean'][-20:-1].mean():.2f} K, T_rot = {bl_stats['T_rot_mean'][-20:-1].mean():.2f} K")
print(f"SPARTA VHS: T_trans = {T_trans_spartaVHS[-20:-1].mean():.2f} K, T_rot = {T_rot_spartaVHS[-20:-1].mean():.2f} K")
print(f"SPARTA VSS: T_trans = {T_trans_spartaVSS[-20:-1].mean():.2f} K, T_rot = {T_rot_spartaVSS[-20:-1].mean():.2f} K")


# --- plot energy relaxation ---
fig, ax = plt.subplots(figsize=plotconfig.figsize)

ax.plot(
    bl_stats["timestep"],
    bl_stats["T_trans_mean"],
    label="$T_{trans}$ BL VHS",
)
ax.plot(
    bl_stats["timestep"],
    bl_stats["T_rot_mean"],
    label="$T_{rot}$ BL VHS",
)

ax.set_xlabel(
    "Time [s]",
    fontsize=plotconfig.label_fontsize,
    fontweight=plotconfig.label_fontweight,
)
ax.ticklabel_format(style="sci", scilimits=(-2, 3))
ax.set_ylabel(
    "Temperature [K]",
    fontsize=plotconfig.label_fontsize,
    fontweight=plotconfig.label_fontweight,
)

# Add energy relaxation plots from SPARTA
ax.plot(
    t_spartaVHS,
    T_trans_spartaVHS,
    label="$T_{trans}$ BL VHS (SPARTA)",
    color="red",
    linestyle="--",
)
ax.plot(
    t_spartaVHS, T_rot_spartaVHS, label="$T_{rot}$ BL VHS (SPARTA)", color="blue", linestyle="--"
)
ax.plot(
    t_spartaVSS, T_trans_spartaVSS, label="$T_{trans}$ BL VSS (SPARTA)", color="green", linestyle="--"
)
ax.plot(
    t_spartaVSS, T_rot_spartaVSS, label="$T_{rot}$ BL VSS (SPARTA)", color="orange", linestyle="--"
)

ax.legend(loc="upper right", fontsize=plotconfig.legend_fontsize, ncol=2)
ax.set_ylim(20, 450)
ax.grid()
fig.savefig("results/plots/DSMC_validation.png", dpi=500)
plt.show()
