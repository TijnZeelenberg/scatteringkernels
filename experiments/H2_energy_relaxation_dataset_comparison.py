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
nr_steps = 200
trans_temperature = 300  # K
rot_temperature = 100  # K
mass = 2.016e-3 / 6.022e23  # kg, mass of one H2 molecule

kB = 1.380649e-23  # J/K
N_sim = 20000
N_real = 20000
n = N_real / volume
d_H2 = 2.92e-10

bmax_values = [1.0, 1.2, 1.5]
alpha_values = [0.5, 1.0, 1.5]

# --- run BL simulation once ---
bl = borgnakke_larssen_model(randomseed=randomseed)
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
bl_dsmc.run_simulation(nr_steps=nr_steps, dt=dt, collision_model=bl)
bl_stats = bl_dsmc.get_stats()

# --- load SPARTA data ---
DATA = np.loadtxt("data/sparta_H2_energy_relaxation.dat", skiprows=2)
t_sparta = DATA[:, 1]
T_trans_sparta = DATA[:, 2]
T_rot_sparta = DATA[:, 3]

# --- run MDN simulations and collect stats ---
all_stats = {}
for b in bmax_values:
    for a in alpha_values:
        model_path = f"results/models/H2H2_mdn_b{b}_alpha{a}".replace(".", "_") + ".pth"

        mdn = MixtureDensityNetwork(
            input_dim=experiment_config.input_dim,
            output_dim=experiment_config.output_dim,
            num_mixtures=experiment_config.num_mixtures,
            hidden_dim=experiment_config.hidden_dim,
            randomseed=randomseed,
        )
        mdn.load_model(model_path)

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
        mdn_dsmc.run_simulation(nr_steps=nr_steps, dt=dt, collision_model=mdn)
        all_stats[(b, a)] = mdn_dsmc.get_stats()

        mdn_stats = all_stats[(b, a)]
        print(f"b={b}, alpha={a} — Final mean temperatures:")
        print(f"  MDN: T_trans = {mdn_stats['T_trans_mean'][-20:-1].mean():.2f} K, T_rot = {mdn_stats['T_rot_mean'][-20:-1].mean():.2f} K")
        print(f"  BL:  T_trans = {bl_stats['T_trans_mean'][-20:-1].mean():.2f} K, T_rot = {bl_stats['T_rot_mean'][-20:-1].mean():.2f} K")
        print(f"  SPARTA: T_trans = {T_trans_sparta[-20:-1].mean():.2f} K, T_rot = {T_rot_sparta[-20:-1].mean():.2f} K")

# --- plot all in a single 3x3 figure ---
# rows = alpha, cols = bmax
fig, axes = plt.subplots(
    len(alpha_values),
    len(bmax_values),
    figsize=(plotconfig.figsize[0] * len(bmax_values), plotconfig.figsize[1] * len(alpha_values)),
    sharex=True,
    sharey=True,
)

for row, a in enumerate(alpha_values):
    for col, b in enumerate(bmax_values):
        ax = axes[row][col]
        mdn_stats = all_stats[(b, a)]

        ax.plot(mdn_stats["timestep"], mdn_stats["T_trans_mean"], label="T_trans MDN")
        ax.plot(mdn_stats["timestep"], mdn_stats["T_rot_mean"], label="T_rot MDN")
        ax.plot(bl_stats["timestep"], bl_stats["T_trans_mean"], label="T_trans BL")
        ax.plot(bl_stats["timestep"], bl_stats["T_rot_mean"], label="T_rot BL")
        ax.plot(t_sparta, T_trans_sparta, label="T_trans SPARTA", color="red", linestyle="--")
        ax.plot(t_sparta, T_rot_sparta, label="T_rot SPARTA", color="blue", linestyle="--")

        ax.set_title(f"b_max={b}, alpha={a}", fontsize=plotconfig.title_fontsize, fontweight=plotconfig.title_fontweight)
        ax.ticklabel_format(style="sci", scilimits=(0, 0))
        ax.grid()

        if col == 0:
            ax.set_ylabel("Temperature [K]", fontsize=plotconfig.label_fontsize, fontweight=plotconfig.label_fontweight)
        if row == len(alpha_values) - 1:
            ax.set_xlabel("Time [s]", fontsize=plotconfig.label_fontsize, fontweight=plotconfig.label_fontweight)

# single shared legend below the figure
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=plotconfig.legend_fontsize, bbox_to_anchor=(0.5, 0))
fig.suptitle("H2 Energy Relaxation — b_max vs alpha", fontsize=plotconfig.title_fontsize, fontweight=plotconfig.title_fontweight)
fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

plot_path = "results/plots/H2_energy_relaxation_dataset_comparison.png"
fig.savefig(plot_path, dpi=300)
plt.show()
print(f"Saved: {plot_path}")
