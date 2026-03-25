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
        self.Xref = None

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
        self.nx = x_cells
        self.ny = y_cells
        self.nz = z_cells

    def set_particle_positions(
        self, nr_particles: int, distribution_type: ParticleDistribution
    ):
        self.nr_particles = nr_particles
        self.Xref = np.zeros(self.nr_particles, dtype=int)

        if self.box_size is None:
            raise ValueError(
                "Simulation domain must be initialized before setting particle positions."
            )
        if distribution_type == "uniform":
            self.positions = self.rng.uniform(
                0.0, self.box_size, size=(nr_particles, dimensions)
            ).astype(np.float32)
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
        self.Xref = np.zeros((nr_particles, dimensions), dtype=int)
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

    def update_cell_indices(self):
        """Update the cell indices for each particle based on their current positions."""

        if self.positions is None:
            raise ValueError(
                "Particle positions must be initialized before updating cell indices."
            )
        if self.Xref is None:
            raise ValueError(
                "Particles must be initialized before updating cell indices."
            )

        x_idx = np.floor(self.positions[:, 0] / self.cell_sizes[0]).astype(int)
        y_idx = np.floor(self.positions[:, 1] / self.cell_sizes[1]).astype(int)
        z_idx = np.floor(self.positions[:, 2] / self.cell_sizes[2]).astype(int)
        self.Xref = x_idx + y_idx * self.nx + z_idx * self.nx * self.ny

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

        if self.Xref is None:
            raise ValueError(
                "Particles must be initialized before updating cell indices."
            )
        # Sort particles by cell index
        sorted_indices = np.argsort(self.Xref)
        sorted_cells = self.Xref[sorted_indices]

        # Find where each cell starts/ends in the sorted array
        changes = np.where(np.diff(sorted_cells) != 0)[0] + 1
        splits = np.split(sorted_indices, changes)

        # Build collision pairs from each cell
        collision_pairs = []
        for particles in splits:
            n = len(particles)
            if n < 2:
                continue
            nr_pairs = int(collision_probability * n // 2)
            if nr_pairs < 1:
                continue
            # Shuffle and take consecutive pairs
            shuffled = self.rng.permutation(particles)
            pairs = shuffled[: 2 * nr_pairs].reshape(nr_pairs, 2)
            collision_pairs.append(pairs)

        return collision_pairs

    def perform_collisions(self, collision_model, collision_pairs: list[np.ndarray]):
        """Perform collisions for the selected pairs of particles using the given collision model."""
        if self.velocities is None or self.rotational_energies is None:
            raise ValueError(
                "Particle velocities and rotational energies must be initialized."
            )
        if self.box_size is None:
            raise ValueError(
                "Simulation domain must be initialized before performing collisions."
            )

        Pxy_col = 0.0
        Pxz_col = 0.0
        Pyz_col = 0.0

        if not collision_pairs:
            return Pxy_col, Pxz_col, Pyz_col

        all_pairs = np.concatenate(collision_pairs, axis=0)
        idx_i = all_pairs[:, 0]
        idx_j = all_pairs[:, 1]

        v_i = self.velocities[idx_i].copy()
        v_j = self.velocities[idx_j].copy()
        e_rot_i = self.rotational_energies[idx_i].copy()
        e_rot_j = self.rotational_energies[idx_j].copy()

        if hasattr(collision_model, "batch_collide"):
            new_v_i, new_e_rot_i, new_v_j, new_e_rot_j = collision_model.batch_collide(
                v_i, e_rot_i, v_j, e_rot_j, m=self.mass
            )

            self.velocities[idx_i] = new_v_i
            self.velocities[idx_j] = new_v_j
            self.rotational_energies[idx_i] = new_e_rot_i
            self.rotational_energies[idx_j] = new_e_rot_j

            Pxy_col = self.mass * np.sum(
                (new_v_i[:, 0] * new_v_i[:, 1] - v_i[:, 0] * v_i[:, 1])
                + (new_v_j[:, 0] * new_v_j[:, 1] - v_j[:, 0] * v_j[:, 1])
            )
            Pxz_col = self.mass * np.sum(
                (new_v_i[:, 0] * new_v_i[:, 2] - v_i[:, 0] * v_i[:, 2])
                + (new_v_j[:, 0] * new_v_j[:, 2] - v_j[:, 0] * v_j[:, 2])
            )
            Pyz_col = self.mass * np.sum(
                (new_v_i[:, 1] * new_v_i[:, 2] - v_i[:, 1] * v_i[:, 2])
                + (new_v_j[:, 1] * new_v_j[:, 2] - v_j[:, 1] * v_j[:, 2])
            )
        else:
            for k in range(len(all_pairs)):
                i, j = idx_i[k], idx_j[k]
                vi_old = self.velocities[i].copy()
                vj_old = self.velocities[j].copy()

                new_vi, new_eri, new_vj, new_erj = collision_model.collide(
                    vi_old,
                    self.rotational_energies[i],
                    vj_old,
                    self.rotational_energies[j],
                    m=self.mass,
                )

                self.velocities[i] = new_vi
                self.velocities[j] = new_vj
                self.rotational_energies[i] = new_eri
                self.rotational_energies[j] = new_erj

                Pxy_col += self.mass * (
                    (new_vi[0] * new_vi[1] - vi_old[0] * vi_old[1])
                    + (new_vj[0] * new_vj[1] - vj_old[0] * vj_old[1])
                )
                Pxz_col += self.mass * (
                    (new_vi[0] * new_vi[2] - vi_old[0] * vi_old[2])
                    + (new_vj[0] * new_vj[2] - vj_old[0] * vj_old[2])
                )
                Pyz_col += self.mass * (
                    (new_vi[1] * new_vi[2] - vi_old[1] * vi_old[2])
                    + (new_vj[1] * new_vj[2] - vj_old[1] * vj_old[2])
                )

        return Pxy_col, Pxz_col, Pyz_col

    def update_positions(self, dt):
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

        stats = {
            "timestep": np.zeros(nr_steps),
            "T_trans_mean": np.zeros(nr_steps),
            "T_rot_mean": np.zeros(nr_steps),
            "T_trans_std": np.zeros(nr_steps),
            "T_rot_std": np.zeros(nr_steps),
            "total_energy": np.zeros(nr_steps),
            "Pxy": np.zeros(nr_steps),
            "Pxz": np.zeros(nr_steps),
            "Pyz": np.zeros(nr_steps),
        }

        start_time = time()
        for step in tqdm(
            range(nr_steps), desc="Running DSMC Simulation", unit="timestep"
        ):
            self.update_positions(dt)
            self.update_cell_indices()
            collision_pairs = self.select_collision_pairs()
            Pxy_col, Pxz_col, Pyz_col = self.perform_collisions(
                collision_model, collision_pairs
            )
            volume = self.box_size**3

            Pxy_kin = (
                self.mass
                * np.sum(self.velocities[:, 0] * self.velocities[:, 1])
                / volume
            )
            Pxz_kin = (
                self.mass
                * np.sum(self.velocities[:, 0] * self.velocities[:, 2])
                / volume
            )
            Pyz_kin = (
                self.mass
                * np.sum(self.velocities[:, 1] * self.velocities[:, 2])
                / volume
            )

            stats["Pxy"][step] = Pxy_kin + (Pxy_col / (volume))
            stats["Pxz"][step] = Pxz_kin + (Pxz_col / (volume))
            stats["Pyz"][step] = Pyz_kin + (Pyz_col / (volume))

            # Store energy statistics
            stats["timestep"][step] = step * dt

            # Convert translational kinetic energy to temperature
            # For 3 DOF: E_trans = (3/2) k_B T → T = (2/3) E_trans / k_B
            trans_energies = 0.5 * self.mass * np.sum(self.velocities**2, axis=1)
            stats["T_trans_mean"][step] = np.mean(trans_energies) / (1.5 * self._kB)
            stats["T_trans_std"][step] = np.std(trans_energies) / (1.5 * self._kB)

            # Convert rotational energy to temperature
            # For 2 DOF: E_rot = k_B T → T = E_rot / k_B
            stats["T_rot_mean"][step] = np.mean(self.rotational_energies) / self._kB
            stats["T_rot_std"][step] = np.std(self.rotational_energies) / self._kB

            # Total energy in Joules (or convert to effective temperature)
            stats["total_energy"][step] = np.sum(
                trans_energies + self.rotational_energies
            )

        self.stats = stats
        end_time = time()
        print(f"Simulation took {end_time - start_time:.2f} seconds.")

    def get_stats(self):
        """Return the energy history of the simulation."""
        if not hasattr(self, "stats"):
            raise ValueError("Simulation must be run before getting energy history.")
        return self.stats
