import numpy as np
import matplotlib.pyplot as plt
import torch
from config.plotting_config import PlottingConfig
from config.experiment_config import ExperimentConfig
from experiments.energy_relaxation import load_mdn
from visualization.plot import plot_density_scatter
from utils.helpers import load_dataset

plotconfig = PlottingConfig()
experimentconfig = ExperimentConfig()

best_model_path = "results/h2/models/mdn/best_model_mdn_H2_bs2000_bmax1_6.pth"

## DSMC validation relaxation comparison ##
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

sparta = np.loadtxt("data/sparta/h2_energy_relaxation.dat")
lammps = np.loadtxt("data/lammps/h2_energy_relaxation.dat", skiprows=1)
bl = np.loadtxt("data/ml-dsmc/bl/h2_energy_relaxation.dat", skiprows=1)

# Convert time to nanoseconds
sparta_t = sparta[:, 1] * 1e9
lammps_t = lammps[:, 1] * 1e9
bl_t = bl[:, 1] * 1e9

sources = [
    (lammps_t, lammps[:, 2], lammps[:, 3], "MD (LAMMPS)"),
    (sparta_t, sparta[:, 2], sparta[:, 3], "SPARTA (BL)"),
    (bl_t, bl[:, 2], bl[:, 3], "ml-DSMC (BL)"),
]

for t, T_trans, T_rot, label in sources:
    axes[0].plot(t, T_trans, label=label)
    axes[1].plot(t, T_rot, label=label)

for ax, title, ylabel in zip(
    axes,
    ["Translational Temperature", "Rotational Temperature"],
    ["$T_{trans}$ [K]", "$T_{rot}$ [K]"],
):
    ax.set_xlabel(
        "Time [ns]",
        fontsize=plotconfig.label_fontsize,
        fontweight=plotconfig.label_fontweight,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=plotconfig.label_fontsize,
        fontweight=plotconfig.label_fontweight,
    )
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)
    ax.grid()

fig.tight_layout()
fig.savefig("results/h2/plots/report/dsmc_validation_relaxation.png", dpi=300)


## impact parameter Delta_trans binned average ##
bfac_sweep = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
N_BINS = 80

fig, ax = plt.subplots(figsize=plotconfig.figsize)

bin_edges = np.linspace(0, 1, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

for bfac in bfac_sweep:
    tag = str(bfac).replace(".", "_")
    data = np.load(
        f"data/ctc/h2/impactparam/Erelmax10000/H2_collisions_b{tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"
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
fig.savefig("results/h2/plots/report/mdn_impactparam_delta_trans.png", dpi=300)


## impact parameter sweep loss history ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)

for bfac in bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    model_dict = torch.load(
        f"results/h2/models/mdn/impactparam/Erelmax10000/mdn_H2_b{bfac_tag}.pth"
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
fig.savefig("results/h2/plots/report/mdn_impactparam_loss_history.png", dpi=300)


## impact parameter relaxation T_rot_mean ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)

for bfac in bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    arr = np.load(f"data/ml-dsmc/mdn/h2/impactparam/relaxation_b{bfac_tag}.npy")
    ax.plot(arr["timestep"], arr["T_rot_mean"], label=f"$b_{{fac}}={bfac}$")

ax.set_xlabel("Time [$s$]", fontsize=plotconfig.label_fontsize)
ax.set_ylabel("Rotational Temperature [$K$]", fontsize=plotconfig.label_fontsize)
ax.legend(fontsize=plotconfig.legend_fontsize)
ax.grid()
fig.tight_layout()
fig.savefig("results/h2/plots/report/mdn_impactparam_relaxation.png", dpi=300)


# num gaussians sweep loss history ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)
num_gaussians = [1, 3, 5, 8, 10, 12, 15, 18, 20]

for num in num_gaussians:
    ng_tag = str(num)
    model_dict = torch.load(
        f"results/h2/models/mdn/num_gaussians/b1_6/mdn_H2_ng{ng_tag}.pth"
    )
    val_loss_history = model_dict["val_loss_history"]
    ax.plot(val_loss_history, label="n/o Gaussians = " + ng_tag)
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
fig.savefig("results/h2/plots/report/mdn_number_gaussians_loss_history.png", dpi=300)


# ## batch size sweep loss history ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)
batch_sizes = [1000, 2000, 5000, 10000, 12500, 15625]
for bs in batch_sizes:
    bs_tag = str(bs)
    model_dict = torch.load(
        f"results/h2/models/mdn/batch_size/Erelmax10000/b1_6/mdn_H2_bs{bs_tag}.pth"
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
fig.savefig("results/h2/plots/report/mdn_batch_size_loss_history.png", dpi=300)


## batch size relaxation T_rot_mean ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)

for bs in batch_sizes:
    arr = np.load(f"data/ml-dsmc/mdn/h2/batch_size/relaxation_bs{bs}.npy")
    ax.plot(arr["timestep"], arr["T_rot_mean"], label=f"batch size = {bs}")

ax.set_xlabel("Time [$s$]", fontsize=plotconfig.label_fontsize)
ax.set_ylabel("Rotational Temperature [$K$]", fontsize=plotconfig.label_fontsize)
ax.legend(fontsize=plotconfig.legend_fontsize)
ax.grid()
fig.tight_layout()
fig.savefig("results/h2/plots/report/mdn_batchsize_relaxation.png", dpi=300)


################ H2 scatterplot of CTC and MDN predictions ##
mdn_path = best_model_path
fig, ax = plt.subplots(
    2, 2, figsize=(2 * (plotconfig.figsize[0]), 2 * plotconfig.figsize[1])
)
datafile = "data/ctc/h2/impactparam/Erelmax10000/H2_collisions_b1_6_uniform_Erelmax10000_ncoll1000000_seed42.npy"

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
fig.tight_layout()

for row in ax:
    for a in row:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)


plt.show()
