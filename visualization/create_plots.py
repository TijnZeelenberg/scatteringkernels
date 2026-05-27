import numpy as np
import matplotlib.pyplot as plt
import torch
from config.plotting_config import PlottingConfig
from visualization.plot import plot_density_scatter
from utils.helpers import load_dataset

config = PlottingConfig()

## impact parameter sweep loss history ##
fig, ax = plt.subplots(figsize=config.figsize)

bfac_sweep = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
for bfac in bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    model_dict = torch.load(f"results/models/mdn/impactparam/mdn_H2_b{bfac_tag}.pth")
    val_loss_history = model_dict["val_loss_history"]
    ax.plot(val_loss_history, label=f"$b_{{fac}}={bfac}$")
ax.set_xlabel(
    "Epoch",
    fontsize=config.label_fontsize,
    fontweight=config.label_fontweight,
)
ax.set_ylabel(
    "Loss",
    fontsize=config.label_fontsize,
    fontweight=config.label_fontweight,
)
ax.legend(fontsize=config.legend_fontsize)
fig.tight_layout()
fig.savefig("results/plots/mdn_impactparam_loss_history.png", dpi=300)
plt.show()


# ## batch size sweep loss history ##
# fig, ax = plt.subplots(figsize=config.figsize)
# batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# for bs in batch_sizes:
#     bs_tag = str(bs)
#     model_dict = torch.load("results/models/mdn/batch_size/mdn_H2_b{bs_tag}.pth")
#     val_loss_history = model_dict["val_loss_history"]
#     ax.plot(val_loss_history, label="Validation Loss")
# ax.set_xlabel(
#     "Epoch",
#     fontsize=config.label_fontsize,
#     fontweight=config.label_fontweight,
# )
# ax.set_ylabel(
#     "Loss",
#     fontsize=config.label_fontsize,
#     fontweight=config.label_fontweight,
# )
# ax.legend(fontsize=config.legend_fontsize)
# fig.tight_layout()
# fig.savefig("results/plots/mdn_batch_size_loss_history.png", dpi=300)
#
#
# ## H2 scatterplot of CTC and MDN predictions ##
# fig, ax = plt.subplots(figsizse=(2 * (config.figsize[0]), config.figsize[1]))
# ctc_data = load_dataset(
#     "data/ctc/H2/impactparam/H2_collisions_b1_0_uniform_Erelmax10000_ncoll1000000_seed42.npy"
# )
# ctc_out = ctc_data[:,1]
#
# datasets = {
#     "inputs":
#     "CTC": ctc_data,
#     "MDN": "results/datasets/mdn/H2/impactparam/H2_collisions_b1_5_uniform_Erelmax10000_ncoll1000000_seed42.npy",
# }
