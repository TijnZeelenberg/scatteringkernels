import numpy as np
import matplotlib.pyplot as plt
import torch
from config.plotting_config import PlottingConfig
from config.experiment_config import ExperimentConfig
from experiments.energy_relaxation import load_mdn
from utils.helpers import load_dataset
from visualization.plot import plot_density_scatter
from analysis.kl_divergence import kl_divergence
from analysis.relaxation_rrmse import time_to_95

plotconfig = PlottingConfig()
experimentconfig = ExperimentConfig()

plotpath = "../Master_Thesis_Tijn_Zeelenberg/figures"

h2_bfac_sweep = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
o2_bfac_sweep = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
num_gaussians = [1, 3, 5, 8, 10, 12, 15, 18, 20]
batch_sizes = [500, 1000, 2000, 5000, 10000, 12500, 15625]
N_BINS = 100

h2_best_model_path = "results/h2/models/mdn/best_model.pth"
o2_best_model_path = "results/o2/models/mdn/best_model.pth"

# Load DSMC/MD relaxation data
sparta_h2 = np.loadtxt("data/sparta/h2_energy_relaxation.dat")
lammps_h2 = np.loadtxt("data/lammps/h2_energy_relaxation.dat", skiprows=1)
bl_h2 = np.loadtxt("data/ml-dsmc/bl/h2_energy_relaxation.dat", skiprows=1)

sparta_o2 = np.loadtxt("data/sparta/o2_energy_relaxation.dat")
lammps_o2 = np.loadtxt("data/lammps/o2_energy_relaxation.dat", skiprows=1)
bl_o2 = np.loadtxt("data/ml-dsmc/bl/o2_energy_relaxation.dat", skiprows=1)


## 0. Energy-conservation error of the training datasets ##
print(f"\n{'Dataset':<16} {'median |dE/E|':>14} {'99th pct |dE/E|':>16}")
print("-" * 48)
for label, datafile in [
    (
        "H2 bfac1_6",
        "data/ctc/h2/impactparam/Erelmax10000/H2_collisions_b1_6_uniform_Erelmax10000_ncoll1000000_seed42.npy",
    ),
    (
        "O2 bmax1_5",
        "data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax1_5.npy",
    ),
]:
    data = np.load(datafile)
    E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
    rel_err = np.abs(E_c_post - E_c_pre) / E_c_pre
    print(
        f"{label:<16} {np.median(rel_err):>14.3e} {np.percentile(rel_err, 99):>16.3e}"
    )


## 1. DSMC validation relaxation comparison 1x2 ##
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

h2_sources = [
    (lammps_h2[:, 1] * 1e9, lammps_h2[:, 3], "MD (LAMMPS)"),
    (sparta_h2[:, 1] * 1e9, sparta_h2[:, 3], "BL (SPARTA)"),
    (bl_h2[:, 1] * 1e9, bl_h2[:, 3], "ml-DSMC (BL)"),
]
o2_sources = [
    (lammps_o2[:, 1] * 1e9, lammps_o2[:, 3], "MD (LAMMPS)"),
    (sparta_o2[:, 1] * 1e9, sparta_o2[:, 3], "BL (SPARTA)"),
    (bl_o2[:, 1] * 1e9, bl_o2[:, 3], "ml-DSMC (BL)"),
]

for t, T_rot, label in h2_sources:
    axes[0].plot(t, T_rot, label=label)
for t, T_rot, label in o2_sources:
    axes[1].plot(t, T_rot, label=label)

for ax, title in zip(axes, ["H$_2$ $T_{rot}$", "O$_2$ $T_{rot}$"]):
    ax.set_xlabel(
        "Time [ns]",
        fontsize=plotconfig.label_fontsize,
        fontweight=plotconfig.label_fontweight,
    )
    ax.set_ylabel(
        "$T_{rot}$ [K]",
        fontsize=plotconfig.label_fontsize,
        fontweight=plotconfig.label_fontweight,
    )
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)
    ax.grid()
axes[0].set_xlim(0, 2.0)
axes[1].set_xlim(0, 4.0)
axes[0].set_ylim(100, 240)
axes[1].set_ylim(100, 240)

fig.tight_layout()
fig.savefig(f"{plotpath}/dsmc_validation_relaxation.png", dpi=300)


## 2. Impact parameter Delta_trans binned average 1x2 ##
bin_edges = np.linspace(0, 1, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])


fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

for bfac in h2_bfac_sweep:
    tag = str(bfac).replace(".", "_")
    data = np.load(
        f"data/ctc/h2/impactparam/Erelmax10000/H2_collisions_b{tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"
    )
    E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
    eta_trans = data[:, 0] / E_c_pre
    delta_trans = data[:, 3] / E_c_post - eta_trans
    bin_idx = np.clip(np.digitize(eta_trans, bin_edges) - 1, 0, N_BINS - 1)
    bin_mean = np.array([delta_trans[bin_idx == i].mean() for i in range(N_BINS)])
    axes[0].plot(bin_centers, bin_mean, label=f"$b_{{fac}}={bfac}$")

for bfac in o2_bfac_sweep:
    tag = str(bfac).replace(".", "_")
    data = np.load(
        f"data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax{tag}.npy"
    )
    E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
    eta_trans = data[:, 0] / E_c_pre
    delta_trans = data[:, 3] / E_c_post - eta_trans
    bin_idx = np.clip(np.digitize(eta_trans, bin_edges) - 1, 0, N_BINS - 1)
    bin_mean = np.array([delta_trans[bin_idx == i].mean() for i in range(N_BINS)])
    axes[1].plot(bin_centers, bin_mean, label=f"$b_{{max}}={bfac}$")

axes[0].set_ylabel(r"$\Delta\eta_{trans}$", fontsize=plotconfig.label_fontsize)
for ax, title in zip(axes, ["H$_2$", "O$_2$"]):
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$\eta_{trans}$", fontsize=plotconfig.label_fontsize)
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)
    ax.grid()

axes[0].set_ylim(-0.175, 0.052)
axes[1].set_ylim(-0.175, 0.052)

fig.tight_layout()
fig.savefig(f"{plotpath}/impactparam_delta_trans.png", dpi=300)


## 3. Near-elastic fraction (|Δη_tr| < 0.01) calibration 1x2 ##
THRESHOLD = 0.01
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

h2_frac_near_elastic = []
for bfac in h2_bfac_sweep:
    tag = str(bfac).replace(".", "_")
    data = np.load(
        f"data/ctc/h2/impactparam/Erelmax10000/H2_collisions_b{tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"
    )
    E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
    delta_eta_tr = data[:, 3] / E_c_post - data[:, 0] / E_c_pre
    h2_frac_near_elastic.append((np.abs(delta_eta_tr) < THRESHOLD).mean())
axes[0].plot(h2_bfac_sweep, h2_frac_near_elastic, marker="o")

o2_frac_near_elastic = []
for bfac in o2_bfac_sweep:
    tag = str(bfac).replace(".", "_")
    data = np.load(
        f"data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax{tag}.npy"
    )
    E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
    delta_eta_tr = data[:, 3] / E_c_post - data[:, 0] / E_c_pre
    o2_frac_near_elastic.append((np.abs(delta_eta_tr) < THRESHOLD).mean())
axes[1].plot(o2_bfac_sweep, o2_frac_near_elastic, marker="o")

axes[0].set_ylabel(
    r"fraction with $|\Delta \eta_{tr}| < 0.01$",
    fontsize=plotconfig.label_fontsize,
)
for ax, title in zip(axes, ["H$_2$", "O$_2$"]):
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.grid()

axes[0].set_xlabel(
    r"$b_{fac}$ ($b_{max} / \sigma$)", fontsize=plotconfig.label_fontsize
)
axes[1].set_xlabel(r"$b_{max}$", fontsize=plotconfig.label_fontsize)
axes[0].set_ylim(0.0, 1.0)
axes[1].set_ylim(0.0, 1.0)

fig.tight_layout()
fig.savefig(f"{plotpath}/impactparam_calibration.png", dpi=300)


## 4. Impact parameter sweep loss history 1x2 ##
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

for bfac in h2_bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    model_dict = torch.load(
        f"results/h2/models/mdn/impactparam/Erelmax10000/n_epochs300/mdn_H2_b{bfac_tag}.pth"
    )
    axes[0].plot(model_dict["val_loss_history"], label=f"$b_{{fac}}={bfac}$")

for bfac in o2_bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    model_dict = torch.load(f"results/o2/models/mdn/impactparam/mdn_O2_b{bfac_tag}.pth")
    axes[1].plot(model_dict["val_loss_history"], label=f"$b_{{max}}={bfac}$")

for ax, title in zip(axes, ["H$_2$", "O$_2$"]):
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
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)

fig.tight_layout()
fig.savefig(f"{plotpath}/impactparam_loss_history.png", dpi=300)


## 5. Impact parameter sweep relaxation T_rot mean 1x2 ##
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

for bfac in h2_bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    arr = np.load(f"data/ml-dsmc/mdn/h2/impactparam/relaxation_b{bfac_tag}.npy")
    axes[0].plot(arr["timestep"] * 1e9, arr["T_rot_mean"], label=f"$b_{{fac}}={bfac}$")

for bfac in o2_bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    arr = np.load(f"data/ml-dsmc/mdn/o2/impactparam/relaxation_b{bfac_tag}.npy")
    axes[1].plot(arr["timestep"] * 1e9, arr["T_rot_mean"], label=f"$b_{{max}}={bfac}$")

for ax, title in zip(axes, ["H$_2$", "O$_2$"]):
    ax.set_xlabel("Time [ns]", fontsize=plotconfig.label_fontsize)
    ax.set_ylabel("$T_{rot}$ [K]", fontsize=plotconfig.label_fontsize)
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)
    ax.grid()

axes[0].set_ylim(100, 300)
axes[1].set_ylim(100, 300)

fig.tight_layout()
fig.savefig(f"{plotpath}/impactparam_relaxation.png", dpi=300)


## 6. Num Gaussians sweep loss history 1x2 ##
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

for num in num_gaussians:
    ng_tag = str(num)
    model_dict = torch.load(
        f"results/h2/models/mdn/num_gaussians/b1_6/n_epochs300/mdn_H2_ng{ng_tag}.pth"
    )
    axes[0].plot(model_dict["val_loss_history"], label=f"n/o Gaussians = {ng_tag}")

for num in num_gaussians:
    ng_tag = str(num)
    model_dict = torch.load(
        f"results/o2/models/mdn/num_gaussians/mdn_O2_ng{ng_tag}.pth"
    )
    axes[1].plot(model_dict["val_loss_history"], label=f"n/o Gaussians = {ng_tag}")

for ax, title in zip(axes, ["H$_2$", "O$_2$"]):
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
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)

axes[0].set_ylim(-4, 3.2)
axes[1].set_ylim(-4, 3.2)

fig.tight_layout()
fig.savefig(f"{plotpath}/num_gaussians_loss_history.png", dpi=300)


## 6b. Epochs to reach NLL < -3.2 for the Gaussian sweep ##
NLL_TARGET = -3.2


def epochs_to_target(val_loss_history, target):
    below = np.where(np.asarray(val_loss_history) < target)[0]
    return int(below[0]) + 1 if below.size else None


print(f"\nEpochs to reach NLL < {NLL_TARGET} (Gaussian sweep)")
print(f"{'n Gaussians':<12} {'H2':>8} {'O2':>8}")
print("-" * 30)
for num in num_gaussians:
    ng_tag = str(num)
    h2_dict = torch.load(
        f"results/h2/models/mdn/num_gaussians/b1_6/n_epochs300/mdn_H2_ng{ng_tag}.pth"
    )
    o2_dict = torch.load(f"results/o2/models/mdn/num_gaussians/mdn_O2_ng{ng_tag}.pth")
    h2_ep = epochs_to_target(h2_dict["val_loss_history"], NLL_TARGET)
    o2_ep = epochs_to_target(o2_dict["val_loss_history"], NLL_TARGET)
    print(
        f"{num:<12} {('—' if h2_ep is None else h2_ep):>8} "
        f"{('—' if o2_ep is None else o2_ep):>8}"
    )


## 7. Batch size sweep loss history 1x2 ##
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

for bs in batch_sizes:
    model_dict = torch.load(
        f"results/h2/models/mdn/batch_size/Erelmax10000/b1_6/n_epochs300/mdn_H2_bs{bs}.pth"
    )
    axes[0].plot(model_dict["val_loss_history"], label=f"batch size = {bs}")

for bs in batch_sizes:
    model_dict = torch.load(f"results/o2/models/mdn/batch_size/mdn_O2_bs{bs}.pth")
    axes[1].plot(model_dict["val_loss_history"], label=f"batch size = {bs}")

for ax, title in zip(axes, ["H$_2$", "O$_2$"]):
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
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)

    ax.set_ylim(-4, 3.2)
axes[0].set_xlim(0, 200)
axes[1].set_xlim(0, 300)

fig.tight_layout()
fig.savefig(f"{plotpath}/batch_size_loss_history.png", dpi=300)


## 8. Batch size relaxation T_rot mean 1x2 ##
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

for bs in batch_sizes:
    arr = np.load(f"data/ml-dsmc/mdn/h2/batch_size/relaxation_bs{bs}.npy")
    axes[0].plot(arr["timestep"] * 1e9, arr["T_rot_mean"], label=f"batch size = {bs}")

for bs in batch_sizes:
    arr = np.load(f"data/ml-dsmc/mdn/o2/batch_size/relaxation_bs{bs}.npy")
    axes[1].plot(arr["timestep"] * 1e9, arr["T_rot_mean"], label=f"batch size = {bs}")

for ax, title in zip(axes, ["H$_2$", "O$_2$"]):
    ax.set_xlabel("Time [ns]", fontsize=plotconfig.label_fontsize)
    ax.set_ylabel("$T_{rot}$ [K]", fontsize=plotconfig.label_fontsize)
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)
    ax.grid()

axes[0].set_ylim(100, 240)
axes[1].set_ylim(100, 240)

fig.tight_layout()
fig.savefig(f"{plotpath}/batch_size_relaxation.png", dpi=300)


## 9. H2 scatterplot of CTC and MDN predictions 2x2 ##
fig, ax = plt.subplots(
    2, 2, figsize=(2 * plotconfig.figsize[0], 2 * plotconfig.figsize[1])
)
datafile = "data/ctc/h2/impactparam/Erelmax10000/H2_collisions_b1_6_uniform_Erelmax10000_ncoll1000000_seed42.npy"
data = load_dataset(datafile, rows=experimentconfig.num_samples)

mdn = load_mdn(h2_best_model_path, randomseed=experimentconfig.random_seed)
torch.manual_seed(experimentconfig.random_seed + 1)
mdn_samples = mdn.sample(x=data[0])

datasets = {
    "inputs": data[0][:, 1:],
    "CTC": data[1],
    "MDN": mdn_samples,
}
# plot_density_scatter(ax, datasets=datasets)
for row in ax:
    for a in row:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
for i, name in enumerate([r"eta'_tr", r"eta'_rot"]):
    kl = kl_divergence(datasets["CTC"][:, i], datasets["MDN"][:, i])
    print(f"h2 d_kl(ctc || mdn) [{name}] = {kl:.4f}")
fig.tight_layout()
# fig.savefig(f"{plotpath}/h2_mdn_ctc_scatter.png", dpi=300)


# 10. o2 scatterplot of ctc and mdn predictions 2x2 ##
fig, ax = plt.subplots(
    2, 2, figsize=(2 * plotconfig.figsize[0], 2 * plotconfig.figsize[1])
)
datafile = "data/ctc/o2/impactparam/Erelmax10000/O2_collisions_uniform_bmax1_5.npy"
data = load_dataset(datafile, rows=experimentconfig.num_samples)

mdn = load_mdn(o2_best_model_path, randomseed=experimentconfig.random_seed)
torch.manual_seed(experimentconfig.random_seed + 1)
mdn_samples = mdn.sample(x=data[0])

datasets = {
    "inputs": data[0][:, 1:],
    "CTC": data[1],
    "MDN": mdn_samples,
}
# plot_density_scatter(ax, datasets=datasets)
for row in ax:
    for a in row:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
for i, name in enumerate([r"eta'_tr", r"eta'_rot"]):
    kl = kl_divergence(datasets["CTC"][:, i], datasets["MDN"][:, i])
    print(f"o2 d_kl(ctc || mdn) [{name}] = {kl:.4f}")
fig.tight_layout()
# fig.savefig(f"{plotpath}/o2_mdn_ctc_scatter.png", dpi=300)


## 11. Best model relaxation comparison with MD 1x2 (H2 left, O2 right) ##
fig, axes = plt.subplots(
    1, 2, figsize=(2 * plotconfig.figsize[0], plotconfig.figsize[1])
)

best_model_h2 = np.load("data/ml-dsmc/mdn/h2/best_model_relaxation.npy")
best_model_o2 = np.load("data/ml-dsmc/mdn/o2/best_model_relaxation.npy")

h2_t_lammps = lammps_h2[:, 1] * 1e9
h2_t_mdn = best_model_h2["timestep"] * 1e9
o2_t_lammps = lammps_o2[:, 1] * 1e9
o2_t_mdn = best_model_o2["timestep"] * 1e9

# Use prop_cycle colors so T_trans and T_rot share a color per source, distinguished by linestyle
prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
lammps_color = prop_cycle[0]
mdn_color = prop_cycle[1]

for ax, t_lammps, t_mdn, lammps_data, mdn_data, title in zip(
    axes,
    [h2_t_lammps, o2_t_lammps],
    [h2_t_mdn, o2_t_mdn],
    [lammps_h2, lammps_o2],
    [best_model_h2, best_model_o2],
    ["H$_2$", "O$_2$"],
):
    ax.plot(
        t_lammps,
        lammps_data[:, 2],
        color=lammps_color,
        linestyle="-",
        label="$T_{trans}$ MD (LAMMPS)",
    )
    ax.plot(
        t_lammps,
        lammps_data[:, 3],
        color=lammps_color,
        linestyle="--",
        label="$T_{rot}$ MD (LAMMPS)",
    )
    ax.plot(
        t_mdn,
        mdn_data["T_trans_mean"],
        color=mdn_color,
        linestyle="-",
        label="$T_{trans}$ MDN (ml-DSMC)",
    )
    ax.plot(
        t_mdn,
        mdn_data["T_rot_mean"],
        color=mdn_color,
        linestyle="--",
        label="$T_{rot}$ MDN (ml-DSMC)",
    )
    ax.set_xlim(0, 5)
    ax.set_xlabel(
        "Time [ns]",
        fontsize=plotconfig.label_fontsize,
        fontweight=plotconfig.label_fontweight,
    )
    ax.set_ylabel(
        "Temperature [K]",
        fontsize=plotconfig.label_fontsize,
        fontweight=plotconfig.label_fontweight,
    )
    ax.set_title(title, fontsize=plotconfig.label_fontsize)
    ax.legend(fontsize=plotconfig.legend_fontsize)
    ax.grid()

axes[0].set_ylim(100, 310)
axes[1].set_ylim(100, 310)

fig.tight_layout()
fig.savefig(f"{plotpath}/best_model_relaxation.png", dpi=300)

# 95% rotational relaxation times for the best-model curves
print(f"{'Species':<8} {'Method':<16} {'t_95 [ns]':>10}")
print("-" * 36)
for species, t_lammps, t_mdn, lammps_data, mdn_data in [
    ("H2", h2_t_lammps, h2_t_mdn, lammps_h2, best_model_h2),
    ("O2", o2_t_lammps, o2_t_mdn, lammps_o2, best_model_o2),
]:
    print(
        f"{species:<8} {'MD (LAMMPS)':<16} {time_to_95(t_lammps, lammps_data[:, 3]):>10.3f}"
    )
    print(
        f"{species:<8} {'MDN (ml-DSMC)':<16} {time_to_95(t_mdn, mdn_data['T_rot_mean']):>10.3f}"
    )
