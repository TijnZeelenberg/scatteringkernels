import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model

# --- simulation parameters ---
nr_particles = 1000
dt = 0.0004
box_size = 1e-11
n_steps = 2000
n_bins = 100
bin_edges = np.linspace(0, box_size, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dx = bin_edges[1] - bin_edges[0]

# --- run simulation and record x-density ---
Dsmc = DSMC_Simulation(seed=42)
Dsmc.initialize_domain(box_size=box_size, nr_cells=50, boundary="specular")
Dsmc.initialize_particles(
    molecules_per_particle=10, nr_particles=nr_particles, mass=1.0, temperature=300.0,
    particle_distribution="central"
)
print(
    "total energy before simulation:",
    0.5 * np.sum(np.linalg.norm(Dsmc.velocities, axis=1) ** 2),
)
collision_model = borgnakke_larssen_model(rng=Dsmc.rng)


density_xt = np.empty((n_steps, n_bins), dtype=float)

# Simulation loop
for t in range(n_steps):
    Dsmc.update_positions(dt=dt)
    pairs = Dsmc.select_collision_pairs()
    Dsmc.perform_collisions(collision_model, pairs)

    counts, _ = np.histogram(Dsmc.positions[:, 0], bins=bin_edges)
    density_xt[t] = counts / (nr_particles)  # normalized 1D density

print(
    "total energy after simulation:",
    0.5 * np.sum(np.linalg.norm(Dsmc.velocities, axis=1) ** 2),
)

# --- plot 1: space-time density heatmap ---
plt.figure()
plt.imshow(
    density_xt,
    aspect="auto",
    origin="lower",
    norm="log",
    cmap="viridis",
 )
plt.xlabel("x [meters]")
plt.ylabel("time [seconds]")
plt.xticks(ticks=np.linspace(0, n_bins, 5), labels=[f"{x:.2e}" for x in np.linspace(0, box_size, 5)])
plt.yticks(ticks=np.linspace(0, n_steps, 5), labels=[f"{t*dt:.2f}" for t in np.linspace(0, n_steps, 5)])
plt.colorbar(label="normalized particle density")
plt.tight_layout()
plt.show()
