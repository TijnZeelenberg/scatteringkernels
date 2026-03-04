import numpy as np
from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model

Dsmc = DSMC_Simulation(random_seed=41)
rng = Dsmc.rng
molecules_per_particle = 100
nr_particles = 100
mass = (molecules_per_particle * 2.016 / 6.022e23)
dt = 1e-3
nr_steps = 10000

Dsmc.initialize_domain(box_size=1e-11, nr_cells=10, boundary="specular")

Dsmc.initialize_particles(
    molecules_per_particle=molecules_per_particle, nr_particles=nr_particles, mass=mass, temperature=300.0, particle_distribution="left_biased"
)

collision_model = borgnakke_larssen_model(rng=rng)


print(
    "total energy before simulation:",
    0.5 * mass * np.sum(np.linalg.norm(Dsmc.velocities, axis=1) ** 2),
)
print("Average location of particles before simulation:", np.mean(Dsmc.positions, axis=0))

Dsmc.run_simulation(collision_model, nr_steps=nr_steps, dt=dt)

print(
    "total energy after simulation:",
    0.5 * mass * np.sum(np.linalg.norm(Dsmc.velocities, axis=1) ** 2),
)
print("Average location of particles after simulation:", np.mean(Dsmc.positions, axis=0))
