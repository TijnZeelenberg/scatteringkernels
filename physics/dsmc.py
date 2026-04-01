# DSMC implementation by Tijn Zeelenberg (2026)
from time import time
from typing import Literal
from tqdm import tqdm

import numpy as np


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
        self.volume = box_size**3

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
        self.cell_volume = self.cell_sizes[0] * self.cell_sizes[1] * self.cell_sizes[2]
        self.nr_cells = x_cells * y_cells * z_cells
        self.nx = x_cells
        self.ny = y_cells
        self.nz = z_cells
        self.cell_counts = np.zeros(self.nr_cells, dtype=int)

    def create_particles(
        self,
        N_sim: int,
        N_real: int,
        mass: float,
        d: float,
        trans_temperature: float,
        rot_temperature: float,
    ):
        if self.box_size is None:
            raise ValueError(
                "Simulation domain must be initialized before creating particles."
            )
        self.N_sim = N_sim
        self.N_real = N_real
        self.number_density = self.cell_counts / self.volume
        self.mass = mass
        self.diameter = d
        self.temperature = trans_temperature

        ## Initialize particle properties
        # Keep track of cell indices for each particle
        self.Xref = np.zeros(self.N_sim, dtype=int)

        # Distribute particles uniformly in the box
        self.positions = self.rng.uniform(
            0.0, self.box_size, size=(self.N_sim, 3)
        ).astype(np.float32)

        self.update_cell_indices()

        # Distribute velocities according to Maxwell-Boltzmann distribution
        self.velocities = self.rng.normal(
            0,
            np.sqrt(self._kB * trans_temperature / self.mass),
            size=(N_sim, dimensions),
        ).astype(np.float32)

        # Distribute rotational energies according to a Boltzmann distribution
        self.rotational_energies = self.rng.exponential(
            scale=self._kB * rot_temperature, size=N_sim
        ).astype(np.float32)

    def update_cell_indices(self):
        """Update the cell indices for each particle based on their current positions."""

        if self.nr_cells is None:
            raise ValueError("Grid must be initialized before updating cell indices.")
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

        self.cell_counts = np.bincount(self.Xref, minlength=self.nr_cells)

    def accept_collision(self, collisionpair: list, vrmax: float):
        "Decide wether a collision is accepted."
        if self.velocities is None:
            raise ValueError(
                "Particle velocities must be initialized before accepting collisions."
            )

        dv = self.velocities[collisionpair[0]] - self.velocities[collisionpair[1]]
        magdv = np.sqrt(np.sum(dv * dv))
        return self.rng.random() < (magdv / vrmax)

    def calculate_no_collisions(self, dt):
        if self.nr_cells is None:
            raise ValueError(
                "Particles must be initialized and cell indices updated before calculating collisions."
            )
        if self.velocities is None:
            raise ValueError(
                "Particle velocities must be initialized before calculating collisions."
            )

        vrmax = np.zeros(self.nr_cells, dtype=np.float32)

        # Calculate the maximum relative velocity in the cell
        for cell in range(self.nr_cells):
            cell_particles = np.where(self.Xref == cell)[0]
            if len(cell_particles) < 2:
                continue
            for particle in cell_particles:
                v_particle = self.velocities[particle]
                dv = self.velocities[cell_particles] - v_particle
                magdv = np.sqrt(np.sum(dv**2, axis=1))
                max_magdv = np.max(magdv)
                if max_magdv > vrmax[cell]:
                    vrmax[cell] = max_magdv
        # Calculate the number of collision pairs to sample in each cell
        collisions = np.zeros(self.nr_cells, dtype=int)
        collisions = np.round(
            self.cell_counts**2
            * np.pi
            * self.diameter**2
            * vrmax
            * (self.N_real / self.N_sim)
            * (dt / self.cell_volume)
        ).astype(int)

        return collisions, vrmax

    def select_collision_pairs(self, dt):
        if self.nr_cells is None:
            raise ValueError(
                "Particles must be initialized and cell indices updated before selecting collision pairs."
            )
        # Compute number of attempted collisions
        collisions, vrmax = self.calculate_no_collisions(dt=dt)

        collision_pairs = []
        for cell in range(self.nr_cells):
            cell_particles = np.where(self.Xref == cell)[0]
            if len(cell_particles) < 2:
                continue
            num_collisions = int(collisions[cell])
            if num_collisions <= 0:
                pairs = []
                collision_pairs.append(pairs)
                continue

            # Cap to max possible unique pairs within a cell
            num_collisions = min(num_collisions, len(cell_particles) // 2)

            # Randomly select pairs of particles for collision
            selected = self.rng.choice(
                cell_particles, size=2 * num_collisions, replace=False
            )
            candidate_pairs = selected.reshape(num_collisions, 2)

            pairs = [
                [int(i), int(j)]
                for i, j in candidate_pairs
                if self.accept_collision([i, j], vrmax[cell])
            ]
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
        if self.Xref is None:
            raise ValueError(
                "Particles positions must be initialized before running the simulation."
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
        total_collisions = 0
        for step in tqdm(range(nr_steps), desc="Running DSMC Simulation", unit="step"):
            self.update_positions(dt)
            self.update_cell_indices()
            collision_pairs = self.select_collision_pairs(dt=dt)
            pairs_as_arrays = [
                np.array(cell_pairs) if cell_pairs else np.empty((0, 2), dtype=int)
                for cell_pairs in collision_pairs
            ]
            total_collisions += sum(len(p) for p in collision_pairs)

            Pxy_col, Pxz_col, Pyz_col = self.perform_collisions(
                collision_model, pairs_as_arrays
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
        print(f"Total collisions: {total_collisions}")

    def get_stats(self):
        """Return the energy history of the simulation."""
        if not hasattr(self, "stats"):
            raise ValueError("Simulation must be run before getting energy history.")
        return self.stats
