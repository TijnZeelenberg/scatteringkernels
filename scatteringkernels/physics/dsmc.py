import numpy as np
from typing import Literal
from time import time

BoundaryCondition = Literal["specular", "absorbing"]
ParticleDistribution = Literal["uniform","central", "gaussian", "left_biased_gaussian", "left_wall"]
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
        self.domain_init = False
        self.particle_init = False

    def initialize_domain(self, box_size:float, nr_cells:int, boundary:BoundaryCondition="specular"):
        self.box_size = box_size
        self.nr_cells = nr_cells
        self.cell_size = box_size / nr_cells
        self.boundary = boundary
        self.domain_init = True

    def initialize_particles(
        self,
        nr_particles:int,
        molecules_per_particle:int,
        mass:float,
        temperature:float=300.0,
        particle_distribution:ParticleDistribution | str = "uniform",
    ):
        if not self.domain_init:
            raise ValueError("Domain must be initialized before initializing particles.")
        if molecules_per_particle <= 0:
            raise ValueError("particles_per_molecule must be a positive integer.")

        self.nr_particles = nr_particles
        self.particles_per_molecule = molecules_per_particle
        self.mass = mass
        self.temperature = temperature # TODO: add support for temperature gradients and non-equilibrium distributions

        
        #TODO: refactor particle distribution into seperate method that takes particle positions as input
        # Initialize particle data structures
        if particle_distribution == "uniform":
            self.positions = self.rng.uniform(0.0, self.box_size, size=(nr_particles, dimensions)).astype(np.float128)
        elif particle_distribution == "central":
            self.positions = np.empty((nr_particles, dimensions), dtype=np.float128)
            self.positions[:,0] = np.full(nr_particles, self.box_size/2, dtype=np.float128)
            self.positions[:,1] = self.rng.uniform(0.0, self.box_size, size=nr_particles).astype(np.float128)
        elif particle_distribution == "gaussian":
            self.positions = self.rng.normal(self.box_size/2, self.box_size/32, size=(nr_particles, dimensions)).astype(np.float128)
        elif particle_distribution == "left_biased_gaussian":
            self.positions = np.empty((nr_particles, dimensions), dtype=np.float128)
            self.positions[:, 0] = self.rng.uniform(0.0, 0.25 * self.box_size, size=nr_particles)
            self.positions[:, 1] = self.rng.uniform(0.0, self.box_size, size=nr_particles)
        elif particle_distribution == "left_wall":
            self.positions = np.empty((nr_particles, dimensions), dtype=np.float128)
            self.positions[:, 0] = np.zeros(nr_particles, dtype=np.float128)
            self.positions[:, 1] = self.rng.uniform(0.0, self.box_size, size=nr_particles).astype(np.float128)
        else:
            raise ValueError(
                f"Unknown particle_distribution: {particle_distribution}. "
                "Use 'uniform', 'central', 'gaussian', 'left_biased_gaussian', or 'left_wall'."
            )

        self.velocities = self.rng.normal(0, np.sqrt(self._kB * temperature), size=(nr_particles,dimensions)).astype(np.float128)
        self.rotational_energies = np.zeros(nr_particles, dtype=np.float128) # TODO: add support for rotational and vibrational energy modes
        self.cell_indices = self.rng.integers(0, self.nr_cells, size=(nr_particles,))
        self.particle_init = True

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
            collision_model: Function that takes two velocity vectors and returns new velocity vectors.
            collision_pairs: List of arrays containing the indices of the selected collision pairs for each cell.
        """

        for pairs in collision_pairs:
            for i, j in pairs:
                # Get the velocities of the two particles
                v_i = self.velocities[i]
                v_j = self.velocities[j]

                # Perform collision using the provided collision model
                #TODO: define standard interface for collision models
                new_v_i, new_v_j = collision_model.postsample(v_i, v_j, m=self.mass, T=self.temperature)

                # Update the velocities of the particles
                self.velocities[i] = new_v_i
                self.velocities[j] = new_v_j

    def update_positions(self, dt):
        """Update particle positions based on their velocities and the time step.
        
        Args:
            dt: Time step for the position update.
        """
        if not self.domain_init or not self.particle_init:
            raise ValueError("Domain and particles must be initialized before updating positions.")

        self.positions += self.velocities * dt

        # Handle boundary conditions
        if self.boundary == "specular":
            # Reflect particles that go out of bounds
            for d in range(dimensions):
                out_of_bounds_low = self.positions[:, d] < 0
                out_of_bounds_high = self.positions[:, d] >= self.box_size

                self.velocities[out_of_bounds_low, d] *= -1
                self.velocities[out_of_bounds_high, d] *= -1

                self.positions[out_of_bounds_low, d] = -self.positions[out_of_bounds_low, d]
                self.positions[out_of_bounds_high, d] = 2 * self.box_size - self.positions[out_of_bounds_high, d]
        elif self.boundary == "absorbing":
            # Remove particles that go out of bounds
            mask = np.all((self.positions >= 0) & (self.positions < self.box_size), axis=1)
            self.positions = self.positions[mask]
            self.velocities = self.velocities[mask]
            self.cell_indices = self.cell_indices[mask]
        else:
            raise ValueError(f"Unknown boundary condition: {self.boundary}")

    def run_simulation(self, collision_model, nr_steps:int, dt:float):
        """Run the DSMC simulation for a given number of steps and time step."""

        start_time = time()
        for step in range(nr_steps):
            self.update_positions(dt)
            collision_pairs = self.select_collision_pairs()
            self.perform_collisions(collision_model, collision_pairs)
        end_time = time()
        print(f"Simulation took {end_time - start_time:.2f} seconds.")
