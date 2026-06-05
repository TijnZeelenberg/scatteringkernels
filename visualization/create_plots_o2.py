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

best_model_path = "results/o2/models/mdn/batch_size/mdn_O2_bs1000.pth"
plotpath = "../Master_Thesis_Tijn_Zeelenberg/figures/o2/"

## impact parameter sweep loss history ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)

bfac_sweep = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
for bfac in bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    model_dict = torch.load(f"results/o2/models/mdn/impactparam/mdn_O2_b{bfac_tag}.pth")
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
fig.savefig(f"{plotpath}mdn_impactparam_loss_history.png", dpi=300)

## impact parameter relaxation T_rot_mean ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)

for bfac in bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    arr = np.load(f"data/ml-dsmc/mdn/o2/impactparam/relaxation_b{bfac_tag}.npy")
    ax.plot(arr["timestep"], arr["T_rot_mean"], label=f"$b_{{fac}}={bfac}$")

ax.set_xlabel("Time [$s$]", fontsize=plotconfig.label_fontsize)
ax.set_ylabel("Rotational Temperature [$K$]", fontsize=plotconfig.label_fontsize)
ax.legend(fontsize=plotconfig.legend_fontsize)
ax.grid()
fig.tight_layout()
fig.savefig(f"{plotpath}mdn_impactparam_relaxation.png", dpi=300)


## impact parameter Delta_trans binned average ##
N_BINS = 80

fig, ax = plt.subplots(figsize=plotconfig.figsize)

bin_edges = np.linspace(0, 1, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

for bfac in bfac_sweep:
    tag = str(bfac).replace(".", "_")
    data = np.load(
        f"data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax{tag}.npy"
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
fig.savefig(f"{plotpath}mdn_impactparam_delta_trans.png", dpi=300)


## num Gaussians sweep loss history ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)
num_gaussians = [1, 3, 5, 8, 10, 12, 15, 18, 20]

for num in num_gaussians:
    ng_tag = str(num)
    model_dict = torch.load(
        f"results/o2/models/mdn/num_gaussians/mdn_O2_ng{ng_tag}.pth"
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
fig.savefig(f"{plotpath}mdn_number_gaussians_loss_history.png", dpi=300)


# ## batch size sweep loss history ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)
batch_sizes = [1000, 2000, 5000, 10000, 12500, 15625]
for bs in batch_sizes:
    bs_tag = str(bs)
    model_dict = torch.load(f"results/o2/models/mdn/batch_size/mdn_O2_bs{bs_tag}.pth")
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
fig.savefig(f"{plotpath}mdn_batch_size_loss_history.png", dpi=300)


## batch size relaxation T_rot_mean ##
fig, ax = plt.subplots(figsize=plotconfig.figsize)

for bs in batch_sizes:
    arr = np.load(f"data/ml-dsmc/mdn/o2/batch_size/relaxation_bs{bs}.npy")
    ax.plot(arr["timestep"], arr["T_rot_mean"], label=f"batch size = {bs}")

ax.set_xlabel("Time [$s$]", fontsize=plotconfig.label_fontsize)
ax.set_ylabel("Rotational Temperature [$K$]", fontsize=plotconfig.label_fontsize)
ax.legend(fontsize=plotconfig.legend_fontsize)
ax.grid()
fig.tight_layout()
fig.savefig(f"{plotpath}mdn_batchsize_relaxation.png", dpi=300)


################ O2 scatterplot of CTC and MDN predictions ##
# FIX: get the dataset in the right shape such that it can be passed to load_dataset().

mdn_path = best_model_path
fig, ax = plt.subplots(
    2, 2, figsize=(2 * (plotconfig.figsize[0]), 2 * plotconfig.figsize[1])
)
datafile = "data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax1_5.npy"

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
fig.savefig(f"{plotpath}mdn_ctc_scatter.png", dpi=300)

for row in ax:
    for a in row:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
