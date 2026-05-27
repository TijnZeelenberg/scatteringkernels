"""Plot binned-average Δη_trans vs η_trans for all six impact-parameter CTC datasets."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig

bfac_sweep = [
    1.0,
    1.05
    1.1,
    1.15,
    1.2,
    1.25,
    1.3,
    1.35,
    1.4,
    1.45,
    1.5,
    1.55,
    1.6,
    1.65,
    1.7,
    1.75,
    1.8,
]  # impact-parameter sweep values
DATA_DIR = paths.DATA_DIR / "ctc/H2/impactparam"
FNAME_TEMPLATE = "H2_collisions_b{tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"
N_BINS = 80

pc = PlottingConfig()
fig, ax = plt.subplots(figsize=pc.figsize)

bin_edges = np.linspace(0, 1, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

for bfac in bfac_sweep:
    tag = str(bfac).replace(".", "_")
    data = np.load(DATA_DIR / FNAME_TEMPLATE.format(tag=tag))

    E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
    eta_trans = data[:, 0] / E_c_pre
    delta_trans = data[:, 3] / E_c_post - eta_trans

    bin_idx = np.digitize(eta_trans, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, N_BINS - 1)
    bin_mean = np.array([delta_trans[bin_idx == i].mean() for i in range(N_BINS)])

    ax.plot(bin_centers, bin_mean, label=f"$b_{{fac}}={bfac}$")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel(r"$\eta_{trans}$", fontsize=pc.label_fontsize)
ax.set_ylabel(r"$\Delta\eta_{trans}$", fontsize=pc.label_fontsize)
ax.legend(fontsize=pc.legend_fontsize)
ax.grid()

fig.tight_layout()
out = paths.plot_path("mdn_impactparam_delta_trans.png")
fig.savefig(out, dpi=300)
print(f"Saved to {out}")
plt.show()
