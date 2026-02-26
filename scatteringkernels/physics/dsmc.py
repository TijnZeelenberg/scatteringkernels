import numpy as np
from typing import Literal

BoundaryCondition = Literal["specular", "periodic", "absorbing"]
dimensions = 2

class DSMC_Simulation:
    """Direct Simulation Monte Carlo (DSMC) implementation.
    """

    def __init__(
        self,
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)
        self._kB = 1.380649e-23

    def initialize_domain(self, box_size:float, nr_cells:int, boundary:BoundaryCondition="specular"):
        self.box_size = box_size
        self.nr_cells = nr_cells
        self.cell_size = box_size / nr_cells
        self.boundary = boundary

    def initialize_particles(self, nr_molecules:int, nr_particles, temperature=300.0):
        if not hasattr(self, "box_size"):
            raise ValueError("Domain must be initialized before initializing particles.")
        if nr_particles < nr_molecules:
            raise ValueError("Number of particles must be greater than or equal to number of molecules.")

        self.nr_molecules = nr_molecules
        # TODO: add support for temperature gradients and non-equilibrium distributions
        self.temperature = temperature

        # Initialize particle data structures
        # TODO: add support for different particle distributions
        self.positions = self.rng.uniform(0, self.box_size, size=(nr_particles, dimensions)).astype(np.float32)
        self.velocities = self.rng.normal(0, np.sqrt(self._kB * temperature), size=(nr_particles,dimensions)).astype(np.float32)
        self.cell_indices = np.zeros(nr_particles,dtype=np.int32)

    def select_collision_pairs(self):
        # Placeholder for collision pair selection logic
        for i in range(self.nr_cells):
            particle_idx = np.asarray(self.cell_indices == i)
            print("particle_idx", particle_idx)


    def perform_collisions(self, collision_ids):
        # Placeholder for collision handling logic
        pass

    def update_positions(self):
        # Placeholder for position update logic
        pass

    def run_simulation(self, nr_steps):
        # Placeholder for the main simulation loop
        pass

