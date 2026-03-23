import numpy as np
from typing import Literal
from time import time
from tqdm import tqdm

ParticleDistribution = Literal[
    "uniform", "central", "gaussian", "left_biased_gaussian", "left_wall"
]
dimensions = 3


class DSMC_Simulation:
    """Direct Simulation Monte Carlo (DSMC) implementation."""

    def __init__(self, random_seed=None):
        self.rng = np.random.default_rng(random_seed)
        self._kB = 1.380649e-23
        self.positions = None
        self.velocities = None
        self.box_size = None
        self.nr_cells = None
        self.particle_cell_idx = None
        self._track_momentum_transfer = False

    def create_box(self, box_size: float):
        self.box_size = box_size

    def create_grid(self, x_cells: int, y_cells: int, z_cells: int):
        """Initialize the grid for cell-based collision selection.
        Args:
            x: Number of cells along the x-axis.
            y: Number of cells along the y-axis.
            z: Number of cells along the z-axis.
        """
        if self.box_size is None:
            raise ValueError(
                "Simulation domain must be initialized before creating the grid."
            )
        self.cell_sizes = (
            self.box_size / x_cells,
            self.box_size / y_cells,
            self.box_size / z_cells,
        )
        self.nr_cells = x_cells * y_cells * z_cells

    def set_particle_positions(
        self, nr_particles: int, distribution_type: ParticleDistribution
    ):
        self.nr_particles = nr_particles
        self.particle_cell_idx = np.zeros(self.nr_particles, dtype=int)

        if self.box_size is None:
            raise ValueError(
                "Simulation domain must be initialized before setting particle positions."
            )
        if distribution_type == "uniform":
            self.positions = self.rng.uniform(
                0.0, self.box_size, size=(nr_particles, dimensions)
            ).astype(np.float32)
            return
        if distribution_type == "gaussian":
            self.positions = self.rng.normal(
                self.box_size / 2, self.box_size / 32, size=(nr_particles, dimensions)
            ).astype(np.float32)
            return

        self.positions = self.rng.uniform(
            0.0, self.box_size, size=(nr_particles, dimensions)
        ).astype(np.float32)
        if distribution_type == "central":
            self.positions[:, 0] = self.box_size / 2
            return
        if distribution_type == "left_biased_gaussian":
            self.positions[:, 0] = self.rng.uniform(
                0.0, 0.25 * self.box_size, size=nr_particles
            )
            return
        if distribution_type == "left_wall":
            self.positions[:, 0] = 0.0
            return

    def create_particles(
        self,
        nr_particles: int,
        mass: float,
        particle_distribution: ParticleDistribution,
        trans_temperature: float,
        rot_temperature: float,
    ):
        self.nr_particles = nr_particles
        self.mass = mass
        self.temperature = trans_temperature

        self.set_particle_positions(
            nr_particles=nr_particles, distribution_type=particle_distribution
        )
        self.particle_cell_idx = np.zeros((nr_particles, dimensions), dtype=int)
        self.update_cell_indices()

        self.velocities = self.rng.normal(
            0,
            np.sqrt(self._kB * trans_temperature / self.mass),
            size=(nr_particles, dimensions),
        ).astype(np.float32)
        print("Velocities set")

        self.rotational_energies = self.rng.exponential(
            scale=self._kB * rot_temperature, size=nr_particles
        ).astype(np.float32)

    def track_stats(self, momentum_transfer=False):
        self._track_momentum_transfer = momentum_transfer

    def update_cell_indices(self):
        """Update the cell indices for each particle based on their current positions."""

        if self.positions is None:
            raise ValueError(
                "Particle positions must be initialized before updating cell indices."
            )
        if self.particle_cell_idx is None:
            raise ValueError(
                "Particles must be initialized before updating cell indices."
            )

        for particle in range(self.nr_particles):
            x_idx = np.floor(self.positions[particle, 0] / self.cell_sizes[0]).astype(
                int
            )
            y_idx = np.floor(self.positions[particle, 1] / self.cell_sizes[1]).astype(
                int
            )
            z_idx = np.floor(self.positions[particle, 2] / self.cell_sizes[2]).astype(
                int
            )

            # use flat cell indices
            self.particle_cell_idx[particle] = (
                x_idx + y_idx * x_idx + z_idx * x_idx * y_idx
            )

    def select_collision_pairs(self, collision_probability=0.5):
        # TODO: implement proper No-Time-Counter method for selecting collision pairs based on relative velocities and collision cross-sections, instead of using a fixed collision probability.
        """
        Select collision pairs with given collision probability.

        Args:
            collision_probability: Probability of collision for each pair of particles in the same cell.

        Returns:
            pairs (list of arrays): List of arrays containing the indices of the selected collision pairs for each cell.
        """
        if self.nr_cells is None:
            raise ValueError(
                "Grid must be initialized before selecting collision pairs."
            )

        cell_particles = [
            np.where(self.particle_cell_idx == cell)[0] for cell in range(self.nr_cells)
        ]

        # Choose random pairs of particles
        collision_pairs = [
            self.rng.choice(
                particles,
                size=(int(collision_probability * len(particles) // 2), 2),
                replace=False,
            )
            for particles in cell_particles
            if len(particles) > 1
        ]

        return collision_pairs

    def perform_collisions(self, collision_model, collision_pairs: list[np.ndarray]):
        """Perform collisions for the selected pairs of particles using the given collision model.

        Args:
            collision_model: Function that takes two velocity vectors and returns new velocity vectors.
            collision_pairs: List of arrays containing the indices of the selected collision pairs for each cell.
        """
        if self.velocities is None or self.rotational_energies is None:
            raise ValueError(
                "Particle velocities and rotational energies must be initialized before performing collisions."
            )

        if self.box_size is None:
            raise ValueError(
                "Simulation domain must be initialized before performing collisions."
            )

        momentum_transfer = 0.0
        for pair in collision_pairs:
            for i, j in pair:
                # Get the velocities and rotational energies of the two particles
                v_i = self.velocities[i].copy()
                v_j = self.velocities[j].copy()
                e_rot_i = self.rotational_energies[i].copy()
                e_rot_j = self.rotational_energies[j].copy()

                # Perform collision using the provided collision model
                new_v_i, new_e_rot_i, new_v_j, new_e_rot_j = collision_model.collide(
                    v_i, e_rot_i, v_j, e_rot_j, m=self.mass
                )

                # Update the velocities and rotational energies of the particles
                self.velocities[i] = new_v_i
                self.velocities[j] = new_v_j
                self.rotational_energies[i] = new_e_rot_i
                self.rotational_energies[j] = new_e_rot_j

                if self._track_momentum_transfer:
                    # Collisional change in v_x*v_y: Δ(v_xi*v_yi) + Δ(v_xj*v_yj)
                    momentum_transfer += self.mass * (
                        (new_v_i[0] * new_v_i[1] - v_i[0] * v_i[1])
                        + (new_v_j[0] * new_v_j[1] - v_j[0] * v_j[1])
                    )

        if self._track_momentum_transfer:
            return momentum_transfer

        return

    def update_positions_and_indices(self, dt):
        """Update particle positions based on their velocities and the time step.

        Args:
            dt: Time step for the position update.
        """
        if self.positions is None or self.velocities is None:
            raise ValueError(
                "Particle positions and velocities must be initialized before updating positions."
            )

        self.positions += self.velocities * dt

        # Handle periodic boundary conditions
        self.positions = np.mod(self.positions, self.box_size)

        self.update_cell_indices()

    def run_simulation(self, collision_model, nr_steps: int, dt: float):
        """Run the DSMC simulation for a given number of steps and time step."""

        if self.positions is None or self.velocities is None:
            raise ValueError(
                "Particle positions and velocities must be initialized before running the simulation."
            )
        if self.box_size is None:
            raise ValueError(
                "Simulation domain must be initialized before running the simulation."
            )

        energy_history = {
            "timestep": np.zeros(nr_steps),
            "T_trans_mean": np.zeros(nr_steps),
            "T_rot_mean": np.zeros(nr_steps),
            "T_trans_std": np.zeros(nr_steps),
            "T_rot_std": np.zeros(nr_steps),
            "total_energy": np.zeros(nr_steps),
        }

        start_time = time()
        for step in tqdm(
            range(nr_steps), desc="Running DSMC Simulation", unit="timestep"
        ):
            self.update_positions_and_indices(dt)
            collision_pairs = self.select_collision_pairs()
            self.perform_collisions(collision_model, collision_pairs)

            # Store energy statistics
            energy_history["timestep"][step] = step * dt

            # Convert translational kinetic energy to temperature
            # For 3 DOF: E_trans = (3/2) k_B T → T = (2/3) E_trans / k_B
            trans_energies = 0.5 * self.mass * np.sum(self.velocities**2, axis=1)
            energy_history["T_trans_mean"][step] = np.mean(trans_energies) / (
                1.5 * self._kB
            )
            energy_history["T_trans_std"][step] = np.std(trans_energies) / (
                1.5 * self._kB
            )

            # Convert rotational energy to temperature
            # For 2 DOF: E_rot = k_B T → T = E_rot / k_B
            energy_history["T_rot_mean"][step] = (
                np.mean(self.rotational_energies) / self._kB
            )
            energy_history["T_rot_std"][step] = (
                np.std(self.rotational_energies) / self._kB
            )

            # Total energy in Joules (or convert to effective temperature)
            energy_history["total_energy"][step] = np.sum(
                trans_energies + self.rotational_energies
            )

        self.energy_history = energy_history
        end_time = time()
        print(f"Simulation took {end_time - start_time:.2f} seconds.")

    def get_energy_history(self):
        """Return the energy history of the simulation."""
        if not hasattr(self, "energy_history"):
            raise ValueError("Simulation must be run before getting energy history.")
        return self.energy_history
