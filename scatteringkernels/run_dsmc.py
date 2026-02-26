import numpy as np
from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model

Dsmc = DSMC_Simulation(seed=42)
rng = Dsmc.rng

Dsmc.initialize_domain(box_size=10.0, nr_cells=10, boundary="specular")

Dsmc.initialize_particles(nr_molecules=1000, nr_particles=100, mass=1.0, temperature=300.0)

collision_pairs = Dsmc.select_collision_pairs()

collision_model = borgnakke_larssen_model(rng=rng)


print("total energy before collision:", 0.5 * np.sum(np.linalg.norm(Dsmc.velocities, axis=1)**2))
print("velocities before collision:")
print(Dsmc.velocities)

Dsmc.perform_collisions(collision_model=collision_model, collision_pairs=collision_pairs)
print("velocities after collision:")
print(Dsmc.velocities)
print("total energy after collision:", 0.5 * np.sum(np.linalg.norm(Dsmc.velocities, axis=1)**2))