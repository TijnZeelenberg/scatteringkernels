import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model

nr_particles = 1000
# --- run simulation and record x-density ---
Dsmc = DSMC_Simulation(seed=41)
Dsmc.initialize_domain(box_size=1e-11, nr_cells=50, boundary="specular")
Dsmc.initialize_particles(
    molecules_per_particle=10, nr_particles=nr_particles, mass=1.0, temperature=300.0,
    particle_distribution="central"
)
print(
    "total energy before simulation:",
    0.5 * np.sum(np.linalg.norm(Dsmc.velocities, axis=1) ** 2),
)
collision_model = borgnakke_larssen_model(rng=Dsmc.rng)

n_steps = 2000
n_bins = 100
bin_edges = np.linspace(0, Dsmc.box_size, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dx = bin_edges[1] - bin_edges[0]

density_xt = np.empty((n_steps, n_bins), dtype=float)

for t in range(n_steps):
    Dsmc.update_positions(dt=0.0004)
    pairs = Dsmc.select_collision_pairs()
    Dsmc.perform_collisions(collision_model, pairs)

    counts, _ = np.histogram(Dsmc.positions[:, 0], bins=bin_edges)
    density_xt[t] = counts / (nr_particles)  # normalized 1D density over time

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
    norm="log"
 )
plt.xlabel("x")
plt.ylabel("time step")
plt.colorbar(label="n(x,t)")
plt.tight_layout()
plt.show()
