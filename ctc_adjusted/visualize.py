import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config.plotting_config import PlottingConfig
from scipy.stats import gaussian_kde

config = PlottingConfig()

def compute_density_per_point(x, y):
    data = [x, y]
    kde = gaussian_kde(data)
    return kde(data)

data = np.load("ctc_adjusted/collision_data.npy")
df = pd.DataFrame(data, columns=["Etr", "Er1", "Er2", "Etrp", "Er1p", "Er2p"])

scatterfig, scatterax = plt.subplots(1, 3, figsize=(18, 6))
Etr_density = compute_density_per_point(df["Etr"], df["Etrp"])
scatterax[0].scatter(df["Etr"], df["Etrp"], c=Etr_density, cmap="viridis", s=config.scatter_point_size, alpha=config.scatter_alpha)
scatterax[0].set_xlabel("Etr")
scatterax[0].set_ylabel("Etr'")

Er1_density = compute_density_per_point(df["Er1"], df["Er1p"])
scatterax[1].scatter(df["Er1"], df["Er1p"], c=Er1_density, cmap="viridis", s=config.scatter_point_size, alpha=config.scatter_alpha)
scatterax[1].set_xlabel("Er1")
scatterax[1].set_ylabel("Er1'")

Er2_density = compute_density_per_point(df["Er2"], df["Er2p"])
scatterax[2].scatter(df["Er2"], df["Er2p"], c=Er2_density, cmap="viridis", s=config.scatter_point_size, alpha=config.scatter_alpha)
scatterax[2].set_xlabel("Er2")
scatterax[2].set_ylabel("Er2'")

# plot histograms
histfig, histax = plt.subplots(1, 3, figsize=(18, 6))
histax[0].hist(df["Etr"], bins=50, alpha=0.5, label="Etr")
histax[0].hist(df["Etrp"], bins=50, alpha=0.5, label="Etr'")
histax[0].set_xlabel("Energy")
histax[0].set_ylabel("Count")
histax[0].legend()

histax[1].hist(df["Er1"], bins=50, alpha=0.5, label="Er1")
histax[1].hist(df["Er1p"], bins=50, alpha=0.5, label="Er1'")
histax[1].set_xlabel("Energy")
histax[1].set_ylabel("Count")
histax[1].legend()

histax[2].hist(df["Er2"], bins=50, alpha=0.5, label="Er2")
histax[2].hist(df["Er2p"], bins=50, alpha=0.5, label="Er2'")
histax[2].set_xlabel("Energy")
histax[2].set_ylabel("Count")
histax[2].legend()
scatterfig.tight_layout()
histfig.tight_layout()
plt.show()