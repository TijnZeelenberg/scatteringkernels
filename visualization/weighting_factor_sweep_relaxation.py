from physics.dsmc import DSMC_Simulation
from machinelearning.mdn import MixtureDensityNetwork
import numpy as np
import matplotlib.pyplot as plt
from config.plotting_config import PlottingConfig
from config.experiment_config import ExperimentConfig
from tqdm import tqdm

plotconfig = PlottingConfig()
experiment_config = ExperimentConfig()

# --- simulation parameters (identical to H2_energy_relaxation.py) ---
randomseed = 2
box_size = 7.5e-6
volume = box_size**3
dt = 1e-5
nr_steps = 150
trans_temperature = 300
rot_temperature = 100
mass = 2.016e-3 / 6.022e23
zrot_bl = 1 / 0.151
zrot_mdn = zrot_bl / 3.5
N_sim = 20000
N_real = 20000
d_H2 = 2.92e-10

# --- load SPARTA reference data ---
spartaVHS = np.loadtxt("data/sparta_H2_energy_relaxationVHS_zinv0151.dat", skiprows=2)
t_spartaVHS = spartaVHS[:, 1]
T_trans_spartaVHS = spartaVHS[:, 2]
T_rot_spartaVHS = spartaVHS[:, 3]

weights = [0.25, 0.5, 1, 2, 4, 8]

fig, axes = plt.subplots(2, 3, figsize=(16, 12))
axes_flat = axes.flatten()

for i, wf in tqdm(
    enumerate(weights),
    desc="Running simulations with different weighting factors",
    unit="weightingfactor",
):
    ax = axes_flat[i]

    mdn = MixtureDensityNetwork(
        input_dim=3,
        output_dim=2,
        num_mixtures=experiment_config.num_mixtures,
        hidden_dim=experiment_config.hidden_dim,
        randomseed=randomseed,
    )
    mdn.load_model(
        f"results/models/weightsensitivity/mdn_H2_wf{str(wf).replace('.', '_')}.pth"
    )

    sim = DSMC_Simulation(random_seed=randomseed)
    sim.create_box(box_size=box_size)
    sim.create_grid(x_cells=5, y_cells=5, z_cells=5)
    sim.create_particles(
        N_sim=N_sim,
        N_real=N_real,
        mass=mass,
        d=d_H2,
        trans_temperature=trans_temperature,
        rot_temperature=rot_temperature,
        zrot=zrot_mdn,
    )
    sim.run_simulation(nr_steps=nr_steps, dt=dt, collision_model=mdn)
    stats = sim.get_stats()

    ax.plot(stats["timestep"], stats["T_trans_mean"], label="$T_{trans}$ MDN")
    ax.plot(stats["timestep"], stats["T_rot_mean"], label="$T_{rot}$ MDN")
    ax.plot(
        t_spartaVHS,
        T_trans_spartaVHS,
        linestyle="--",
        color="red",
        label="$T_{trans}$ SPARTA",
    )
    ax.plot(
        t_spartaVHS,
        T_rot_spartaVHS,
        linestyle="--",
        color="blue",
        label="$T_{rot}$ SPARTA",
    )

    ax.set_title(f"wf = {wf}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Temperature [K]", fontsize=9)
    ax.ticklabel_format(style="sci", scilimits=(-2, 3))
    ax.set_ylim(20, 450)
    ax.grid(True)
    ax.legend(fontsize=7)

fig.suptitle(
    f"H2 Energy Relaxation — Weighting Factor Sweep - Randomseed {randomseed}",
    fontsize=16,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(
    f"results/plots/H2_weighting_factor_sweep_relaxation_seed{randomseed}.png", dpi=300
)
plt.show()
