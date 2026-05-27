"""Plot binned-average Δη_trans vs η_trans for the H2 b1_5 CTC dataset."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig

DATA_PATH = (
    paths.DATA_DIR
    / "ctc/H2/H2_collisions_b1_2_uniform_Erelmax10000_ncoll1000000_seed42.npy"
)
N_BINS = 80

pc = PlottingConfig()
fig, ax = plt.subplots(figsize=pc.figsize)

bin_edges = np.linspace(0, 1, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

data = np.load(DATA_PATH)

E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
eta_trans = data[:, 0] / E_c_pre
delta_trans = data[:, 3] / E_c_post - eta_trans

bin_idx = np.digitize(eta_trans, bin_edges) - 1
bin_idx = np.clip(bin_idx, 0, N_BINS - 1)
bin_mean = np.array([delta_trans[bin_idx == i].mean() for i in range(N_BINS)])

ax.plot(bin_centers, bin_mean)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel(r"$\eta_{trans}$", fontsize=pc.label_fontsize)
ax.set_ylabel(r"$\Delta\eta_{trans}$", fontsize=pc.label_fontsize)
ax.grid()

fig.tight_layout()
out = paths.plot_path("h2_delta_trans.png")
fig.savefig(out, dpi=300)
print(f"Saved to {out}")
plt.show()
