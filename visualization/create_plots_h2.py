import numpy as np
import matplotlib.pyplot as plt
import torch
from config.plotting_config import PlottingConfig
from config.experiment_config import ExperimentConfig
from experiments.energy_relaxation import load_mdn, run_relaxation, SimulationParams
from physics.species import Species
from dataclasses import replace
from visualization.plot import plot_density_scatter
from utils.helpers import load_dataset

plotconfig = PlottingConfig()
experimentconfig = ExperimentConfig()

## impact parameter sweep loss history ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)

bfac_sweep = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
for bfac in bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    model_dict = torch.load(
        f"results/models/mdn/impactparam/Erelmax10000/mdn_H2_b{bfac_tag}.pth"
    )
    val_loss_history = model_dict["val_loss_history"]
    ax.plot(val_loss_history, label=f"$b_{{fac}}={bfac}$")
ax.set_xlabel(
    "Epoch",
    fontsize=plotconfig.label_fontsize,
    fontweight=plotconfig.label_fontweight,
)
ax.set_ylabel(
    "Loss",
    fontsize=plotconfig.label_fontsize,
    fontweight=plotconfig.label_fontweight,
)
ax.legend(fontsize=plotconfig.legend_fontsize)
fig.tight_layout()
fig.savefig("results/plots/report/mdn_impactparam_loss_history.png", dpi=300)


# ## batch size sweep loss history ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)
batch_sizes = [1000, 2000, 5000, 10000, 12500, 15625]
for bs in batch_sizes:
    bs_tag = str(bs)
    print(f"Loading model with batch size {bs}...")
    model_dict = torch.load(
        f"results/models/mdn/batch_size/Erelmax10000/mdn_H2_b{bs_tag}.pth"
    )
    val_loss_history = model_dict["val_loss_history"]
    ax.plot(val_loss_history, label="batch size = " + bs_tag)
ax.set_xlabel(
    "Epoch",
    fontsize=plotconfig.label_fontsize,
    fontweight=plotconfig.label_fontweight,
)
ax.set_ylabel(
    "Loss",
    fontsize=plotconfig.label_fontsize,
    fontweight=plotconfig.label_fontweight,
)
ax.legend(fontsize=plotconfig.legend_fontsize)
fig.tight_layout()
fig.savefig("results/plots/report/mdn_batch_size_loss_history.png", dpi=300)


## Perform relaxation comparison between batch_size models
nr_steps = 150
dt = 1.0e-11
fig, ax = plt.subplots(figsize=plotconfig.figsize)
for bs in batch_sizes:
    bs_tag = str(bs)
    print(f"Loading model with batch size {bs}...")
    mdn = load_mdn(
        f"results/models/mdn/batch_size/Erelmax10000/mdn_H2_b{bs_tag}.pth",
        randomseed=experimentconfig.random_seed,
    )
    species = replace(
        Species.H2(),
        diameter=10.1e-10,
        zrot_bl=5.0,
        zrot_mdn=5.0 / 2.5,
    )
    params = SimulationParams(
        nr_steps=nr_steps,
        trans_temperature=300.0,
        rot_temperature=100.0,
        randomseed=42,
        grid_cells=(5, 5, 5),
        box_size=1.0e-7,
        dt=dt,
    )
    stats = run_relaxation(species=species, collision_model=mdn, params=params)
    ax.plot(stats["timestep"], stats["T_rot_mean"])
ax.hlines(
    220.0,
    xmin=0,
    xmax=nr_steps * dt,
    color="black",
    linestyle="--",
    label="Equilibrium $T_{rot}$",
)
ax.set_xlabel("Time [$s$]", fontsize=plotconfig.label_fontsize)
ax.set_ylabel("Rotational Temperature [$K$]", fontsize=plotconfig.label_fontsize)

################ H2 scatterplot of CTC and MDN predictions ##
mdn_path = "results/models/mdn/impactparam/Erelmax10000/mdn_H2_b1_5.pth"
fig, ax = plt.subplots(
    2, 2, figsize=(2 * (plotconfig.figsize[0]), 2 * plotconfig.figsize[1])
)
datafile = "data/ctc/H2/impactparam/Erelmax10000/H2_collisions_b1_5_uniform_Erelmax10000_ncoll1000000_seed42.npy"

data = load_dataset(datafile, rows=experimentconfig.num_samples)

# Sample MDN predictions
mdn = load_mdn(mdn_path, randomseed=experimentconfig.random_seed)
torch.manual_seed(experimentconfig.random_seed + 1)
mdn_samples = mdn.sample(x=data[0])

datasets = {
    "inputs": data[0][:, 1:],  # Use only the energy fractions for plotting
    "CTC": data[1],
    "MDN": mdn_samples,
}
plot_density_scatter(ax, datasets=datasets)


## impact parameter Delta_trans binned average ##
N_BINS = 80

fig, ax = plt.subplots(figsize=plotconfig.figsize)

bin_edges = np.linspace(0, 1, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

for bfac in bfac_sweep:
    tag = str(bfac).replace(".", "_")
    data = np.load(
        f"data/ctc/H2/impactparam/Erelmax10000/H2_collisions_b{tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"
    )

    E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
    eta_trans = data[:, 0] / E_c_pre
    delta_trans = data[:, 3] / E_c_post - eta_trans

    bin_idx = np.digitize(eta_trans, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, N_BINS - 1)
    bin_mean = np.array([delta_trans[bin_idx == i].mean() for i in range(N_BINS)])

    ax.plot(bin_centers, bin_mean, label=f"$b_{{fac}}={bfac}$")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel(r"$\eta_{trans}$", fontsize=plotconfig.label_fontsize)
ax.set_ylabel(r"$\Delta\eta_{trans}$", fontsize=plotconfig.label_fontsize)
ax.legend(fontsize=plotconfig.legend_fontsize)
ax.grid()
fig.tight_layout()
fig.savefig("results/plots/report/mdn_impactparam_delta_trans.png", dpi=300)


plt.show()
