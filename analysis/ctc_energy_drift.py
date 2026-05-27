"""Plot cumulative energy drift across collisions for each impact-parameter dataset.

A perfectly energy-conserving kernel has E_total_post = E_total_pre for every
collision, so the cumulative sum of (E_post - E_pre) stays at zero. Drift here
reveals any systematic energy creation or destruction in the CTC model.
"""

import numpy as np
import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig

BFAC_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
DATA_DIR = paths.DATA_DIR / "ctc/H2/impactparam"
FNAME_TEMPLATE = "H2_collisions_b{tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"

pc = PlottingConfig()
fig, ax = plt.subplots(figsize=pc.figsize)

for bfac in BFAC_VALUES:
    tag = str(bfac).replace(".", "_")
    data = np.load(DATA_DIR / FNAME_TEMPLATE.format(tag=tag))

    E_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_post = data[:, 3] + data[:, 4] + data[:, 5]
    cumulative_drift = np.cumsum(E_post - E_pre) / np.cumsum(E_pre)

    ax.plot(cumulative_drift, label=f"$b_{{fac}}={bfac}$")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Collision index", fontsize=pc.label_fontsize)
ax.set_ylabel("Running average relative energy drift", fontsize=pc.label_fontsize)
ax.ticklabel_format(style="sci", scilimits=(0, 0))
ax.legend(fontsize=pc.legend_fontsize)
ax.grid()

fig.tight_layout()
out = paths.plot_path("ctc_impactparam_energy_drift.png")
fig.savefig(out, dpi=300)
print(f"Saved to {out}")
plt.show()
