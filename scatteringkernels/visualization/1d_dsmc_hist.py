import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model

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
Dsmc = DSMC_Simulation(seed=42)
Dsmc.initialize_domain(box_size=box_size, nr_cells=50, boundary="specular")
Dsmc.initialize_particles(
    molecules_per_particle=molecules_per_particle, nr_particles=nr_particles, mass=mass, temperature=300.0,
    particle_distribution="central"
)

E_total = 0.5 * mass * np.sum(np.sum(Dsmc.velocities**2, axis=1)) + np.sum(Dsmc.rotational_energies)
print("initial total energy:", E_total)

collision_model = borgnakke_larssen_model(rng=Dsmc.rng)

density_xt = np.empty((n_steps, n_bins), dtype=float)

# Simulation loop
for t in range(n_steps):
    Dsmc.update_positions(dt=dt)
    pairs = Dsmc.select_collision_pairs()
    Dsmc.perform_collisions(collision_model, pairs)

    counts, _ = np.histogram(Dsmc.positions[:, 0], bins=bin_edges)
    density_xt[t] = counts / (nr_particles)  # normalized 1D density

E_total = 0.5 * mass * np.sum(np.sum(Dsmc.velocities**2, axis=1)) + np.sum(Dsmc.rotational_energies)
print("final total energy:", E_total)

# --- plot 1: space-time density heatmap ---
plt.figure()
plt.imshow(
    density_xt,
    aspect="auto",
    origin="lower",
    norm="log",
    cmap="viridis",
 )
plt.xlabel("x [meters]", fontweight="bold")
plt.ylabel("time [seconds]", fontweight="bold")
plt.xticks(ticks=np.linspace(0, n_bins, 5), labels=[f"{x:.2e}" for x in np.linspace(0, box_size, 5)])
plt.yticks(ticks=np.linspace(0, n_steps, 5), labels=[f"{t*dt:.2e}" for t in np.linspace(0, n_steps, 5)])
plt.colorbar(label="normalized particle density")
plt.tight_layout()
plt.show()
