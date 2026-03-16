from networkx import config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn_model import MixtureDensityNetwork
from config.experiment_config import ExperimentConfig

config = ExperimentConfig()
# --- simulation parameters ---
nr_particles = 1000
molecules_per_particle = 10
mass = (molecules_per_particle * 2.016 / 6.022e23)
dt = 4e-7
box_size = 1e-3
n_steps = 2000
n_bins = 100
bin_edges = np.linspace(0, box_size, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dx = bin_edges[1] - bin_edges[0]

# --- run simulation and record x-density ---
Dsmc = DSMC_Simulation(random_seed=42)
Dsmc.initialize_domain(box_size=box_size, nr_cells=50, boundary="specular")
Dsmc.initialize_particles(
    molecules_per_particle=molecules_per_particle, nr_particles=nr_particles, mass=mass, temperature=300.0,
    particle_distribution="central"
)

E_total = 0.5 * mass * np.sum(np.sum(Dsmc.velocities**2, axis=1)) + np.sum(Dsmc.rotational_energies)
print("initial total energy:", E_total)

collision_model = borgnakke_larssen_model(rng=Dsmc.rng)

density_xt_bl = np.empty((n_steps, n_bins), dtype=float)

# Simulation loop for DSMC with Borgnakke-Larssen collision model
for t in range(n_steps):
    Dsmc.update_positions(dt=dt)
    pairs = Dsmc.select_collision_pairs()
    Dsmc.perform_collisions(collision_model, pairs)

    counts, _ = np.histogram(Dsmc.positions[:, 0], bins=bin_edges)
    density_xt_bl[t] = counts / (nr_particles)  # normalized 1D density

E_total = 0.5 * mass * np.sum(np.sum(Dsmc.velocities**2, axis=1)) + np.sum(Dsmc.rotational_energies)
print("final total Borgnakke-Larssen energy:", E_total)

# --- Re-initialize DSMC state for MDN run ---
Dsmc = DSMC_Simulation(random_seed=42)
Dsmc.initialize_domain(box_size=box_size, nr_cells=50, boundary="specular")
Dsmc.initialize_particles(
    molecules_per_particle=molecules_per_particle, nr_particles=nr_particles, mass=mass, temperature=300.0,
    particle_distribution="central"
)

# --- Simulate with mdn model ---
mdn = MixtureDensityNetwork(input_dim=config.input_dim, output_dim=config.output_dim, num_mixtures=config.num_mixtures, hidden_dim=config.hidden_dim, randomseed=config.random_seed) 
mdn.load_model("results/models/mdn_H2H2.pth")
collision_model_mdn = mdn


density_xt_mdn = np.empty((n_steps, n_bins), dtype=float)

# Simulation loop
for t in range(n_steps):
    Dsmc.update_positions(dt=dt)
    pairs = Dsmc.select_collision_pairs()
    Dsmc.perform_collisions(collision_model_mdn, pairs)

    counts, _ = np.histogram(Dsmc.positions[:, 0], bins=bin_edges)
    density_xt_mdn[t] = counts / (nr_particles)  # normalized 1D density

E_total = 0.5 * mass * np.sum(np.sum(Dsmc.velocities**2, axis=1)) + np.sum(Dsmc.rotational_energies)
print("final total MDN energy:", E_total)

# --- plot: space-time density heatmaps side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)

im_bl = axes[0].imshow(
    density_xt_bl,
    aspect="auto",
    origin="lower",
    norm="log",
    cmap="viridis",
)
axes[0].set_title("Borgnakke-Larsen DSMC", fontweight="bold")
axes[0].set_xlabel("x [meters]", fontweight="bold")
axes[0].set_ylabel("time [seconds]", fontweight="bold")
axes[0].set_xticks(ticks=np.linspace(0, n_bins, 5), labels=[f"{x:.2e}" for x in np.linspace(0, box_size, 5)])
axes[0].set_yticks(ticks=np.linspace(0, n_steps, 5), labels=[f"{t*dt:.2e}" for t in np.linspace(0, n_steps, 5)])

im_mdn = axes[1].imshow(
    density_xt_mdn,
    aspect="auto",
    origin="lower",
    norm="log",
    cmap="viridis",
)
axes[1].set_title("MDN DSMC", fontweight="bold")
axes[1].set_xlabel("x [meters]", fontweight="bold")
axes[1].set_xticks(ticks=np.linspace(0, n_bins, 5), labels=[f"{x:.2e}" for x in np.linspace(0, box_size, 5)])
axes[1].set_yticks(ticks=np.linspace(0, n_steps, 5), labels=[f"{t*dt:.2e}" for t in np.linspace(0, n_steps, 5)])

fig.colorbar(im_mdn, ax=axes, label="normalized particle density", shrink=0.9, pad=0.02)
plt.show()
