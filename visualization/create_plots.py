import matplotlib.pyplot as plt
import torch

from config.plotting_config import PlottingConfig

config = PlottingConfig()

## impact parameter sweep loss history ##
fig, ax = plt.subplots(figsize=config.figsize)

bfac_sweep = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
for bfac in bfac_sweep:
    bfac_tag = str(bfac).replace(".", "_")
    model_dict = torch.load("results/models/mdn/impactparam/mdn_H2_b{bfac_tag}.pth")
    train_loss_history = model_dict["train_loss_history"]
    val_loss_history = model_dict["val_loss_history"]
    ax.plot(train_loss_history, label="Training Loss")
    ax.plot(val_loss_history, label="Validation Loss")
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
fig.show()
