"""Plot bfac vs fraction of near-elastic collisions (|Δη_tr| < 1%) for H2.

Δη_tr is the change in the translational fraction of the total collisional
energy; collisions with |Δη_tr| < 0.01 exchange less than 1% of their energy
between translational and rotational modes.
"""

import numpy as np
import matplotlib.pyplot as plt

import paths
from config.plotting_config import PlottingConfig

b_facs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
DATA_DIR = paths.DATA_DIR / "ctc/h2/impactparam/Erelmax10000"
FNAME_TEMPLATE = "H2_collisions_b{tag}_uniform_Erelmax10000_ncoll1000000_seed42.npy"
THRESHOLD = 0.1

frac_near_elastic = []
for bfac in b_facs:
    tag = str(bfac).replace(".", "_")
    data = np.load(DATA_DIR / FNAME_TEMPLATE.format(tag=tag))
    E_c_pre = data[:, 0] + data[:, 1] + data[:, 2]
    E_c_post = data[:, 3] + data[:, 4] + data[:, 5]
    delta_eta_tr = data[:, 3] / E_c_post - data[:, 0] / E_c_pre
    frac_near_elastic.append((np.abs(delta_eta_tr) < THRESHOLD).mean())

pc = PlottingConfig()
fig, ax = plt.subplots(figsize=pc.figsize)
ax.plot(b_facs, frac_near_elastic, marker="o")
ax.set_xlabel(r"$b_{fac}$ ($b_{max} / \sigma$)", fontsize=pc.label_fontsize)
ax.set_ylabel(r"fraction with $|\Delta \eta_{tr}| < 0.01$", fontsize=pc.label_fontsize)
ax.grid()

fig.tight_layout()
out = paths.plot_path("impactparam_calibration_h2.png")
fig.savefig(out, dpi=300)
print(f"Saved to {out}")
plt.show()
