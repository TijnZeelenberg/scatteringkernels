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

    def initialize_particles(self, nr_molecules:int, nr_particles,  mass:float=1.0, temperature:float=300.0):
        if not hasattr(self, "box_size"):
            raise ValueError("Domain must be initialized before initializing particles.")
        if nr_particles > nr_molecules:
            raise ValueError("Number of particles must be smaller than or equal to number of molecules.")

        self.nr_molecules = nr_molecules
        self.mass = mass
        self.temperature = temperature # TODO: add support for temperature gradients and non-equilibrium distributions

        # Initialize particle data structures
        # TODO: add support for different particle distributions
        self.positions = self.rng.uniform(0, self.box_size, size=(nr_particles, dimensions)).astype(np.float32)
        self.velocities = self.rng.normal(0, np.sqrt(self._kB * temperature), size=(nr_particles,dimensions)).astype(np.float32)
        self.cell_indices = self.rng.integers(0, self.nr_cells, size=(nr_particles,))

    def select_collision_pairs(self, collision_probability=0.5):
        """
        Select collision pairs with given collision probability.
        
        Args:
            collision_probability: Probability of collision for each pair of particles in the same cell.
        
        Returns:
            pairs (list of arrays): List of arrays containing the indices of the selected collision pairs for each cell.
        """

        # create arrays to hold the particles in each cell
        cell_particles = [np.where(self.cell_indices == i)[0] for i in range(self.nr_cells)]

        # Choose random pairs of particles
        collision_pairs = [self.rng.choice(particles, size=(int(collision_probability * len(particles) // 2), 2), replace=False) for particles in cell_particles if len(particles) > 1]

        return collision_pairs

    def perform_collisions(self, collision_model, collision_pairs:list[np.ndarray]):
        """Perform collisions for the selected pairs of particles using the given collision model.
        
        Args:
            collision_model: Function that takes 
            collision_pairs: List of arrays containing the indices of the selected collision pairs for each cell.
        """

        for pairs in collision_pairs:
            for i, j in pairs:
                # Get the velocities of the two particles
                v_i = self.velocities[i]
                v_j = self.velocities[j]

                # Perform collision using the provided collision model
                new_v_i, new_v_j = collision_model.postsample(v_i, v_j, m=self.mass, T=self.temperature)

                # Update the velocities of the particles
                self.velocities[i] = new_v_i
                self.velocities[j] = new_v_j

    def update_positions(self, dt):
        """Update particle positions based on their velocities and the time step.
        
        Args:
            dt: Time step for the position update.
        """

        self.positions += self.velocities * dt

        # Handle boundary conditions
        if self.boundary == "specular":
            self.positions = np.mod(self.positions, self.box_size)
        elif self.boundary == "periodic":
            self.positions = np.mod(self.positions, self.box_size)
        elif self.boundary == "absorbing":
            # Remove particles that go out of bounds
            mask = np.all((self.positions >= 0) & (self.positions < self.box_size), axis=1)
            self.positions = self.positions[mask]
            self.velocities = self.velocities[mask]
            self.cell_indices = self.cell_indices[mask]
        else:
            raise ValueError(f"Unknown boundary condition: {self.boundary}")

    def run_simulation(self, nr_steps, dt):
        # Placeholder for the main simulation loop
        pass

