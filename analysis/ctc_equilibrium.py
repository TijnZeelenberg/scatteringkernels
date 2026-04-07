"""
Demonstrates that the O2-O2 CTC collision model equilibrates at a different
translational energy fraction than the classical equipartition prediction.

Classical DOF argument: eta_tr_eq = 3 / (3 + 2) = 0.60
CTC model crossover (mean delta = 0): eta_tr ~ 0.68-0.70

This is consistent with quantum mechanical suppression of rotational DOF
in O2 at low temperatures (rotational constant B ~ 1.44 cm^-1).
"""
import numpy as np
import matplotlib.pyplot as plt
from config.plotting_config import PlottingConfig

DATASET = "data/filtered/O2O2_collisions.csv"

data = np.loadtxt(DATASET, delimiter=",", skiprows=1)
Etot = data[:, 0] + data[:, 1] + data[:, 2]
eta_tr_in = data[:, 0] / Etot
eta_tr_out = data[:, 3] / (data[:, 3] + data[:, 4] + data[:, 5])
delta = eta_tr_out - eta_tr_in

# --- Bin statistics ---
bin_edges = np.linspace(0.0, 1.0, 26)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_mean = np.full(len(bin_centers), np.nan)
bin_sem = np.full(len(bin_centers), np.nan)

for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    mask = (eta_tr_in >= lo) & (eta_tr_in < hi)
    if mask.sum() > 10:
        bin_mean[i] = np.mean(delta[mask])
        bin_sem[i] = np.std(delta[mask]) / np.sqrt(mask.sum())

# --- Find crossover by linear interpolation between the two bins straddling zero ---
valid = ~np.isnan(bin_mean)
sign_changes = np.where(np.diff(np.sign(bin_mean[valid])))[0]
if len(sign_changes) > 0:
    i = sign_changes[-1]
    xv = bin_centers[valid]
    yv = bin_mean[valid]
    crossover = xv[i] - yv[i] * (xv[i + 1] - xv[i]) / (yv[i + 1] - yv[i])
else:
    crossover = np.nan

cfg = PlottingConfig()
fig, ax = plt.subplots(figsize=cfg.figsize)

ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax.fill_between(bin_centers, bin_mean - bin_sem, bin_mean + bin_sem,
                alpha=0.25, color="steelblue")
ax.plot(bin_centers, bin_mean, color="steelblue", linewidth=2,
        label=r"Mean $\Delta\eta_{tr}$ (CTC data)")

# Classical equilibrium
ax.axvline(0.60, color="red", linestyle="--", linewidth=1.5,
           label=r"Classical equipartition $\eta_{tr}^{eq}=0.60$")

# CTC crossover
if not np.isnan(crossover):
    ax.axvline(crossover, color="darkorange", linestyle="--", linewidth=1.5,
               label=rf"CTC crossover $\eta_{{tr}}\approx{crossover:.2f}$")

ax.set_xlabel(r"Pre-collision $\eta_{tr,in} = E_{tr} / E_{tot}$",
              fontsize=cfg.label_fontsize, fontweight=cfg.label_fontweight)
ax.set_ylabel(r"Mean $\Delta\eta_{tr} = \eta_{tr,out} - \eta_{tr,in}$",
              fontsize=cfg.label_fontsize, fontweight=cfg.label_fontweight)
ax.set_title("CTC vs Classical Rotational Equilibrium (O$_2$-O$_2$)",
             fontsize=cfg.title_fontsize, fontweight=cfg.title_fontweight)
ax.legend(fontsize=cfg.legend_fontsize)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig("results/plots/ctc_equilibrium_O2.png", dpi=300)
plt.show()

print(f"CTC crossover at eta_tr = {crossover:.3f}")
print(f"Classical prediction:    eta_tr = 0.600")
print(f"Difference: {crossover - 0.60:+.3f}")
