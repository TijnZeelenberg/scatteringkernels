# DSMC implementation by Tijn Zeelenberg (2026)
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
        self.number_density = N_real / self.volume
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
        print("Velocities set")

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

    def select_collision_pairs(self, dt: float, cr_max: float):
        """Select collision pairs using the Enskog-modified NTC method.

        Implements the collision pair selection following Frezzotti (1997),
        "A particle scheme for the numerical solution of the Enskog equation",
        Physics of Fluids 9, 1329-1335.

        The standard DSMC No-Time-Counter (NTC) method is augmented with the
        Enskog pair correlation function Y(eta) derived from the Carnahan-Starling
        equation of state (Eq. 30 of Frezzotti 1997):

            Y(eta) = (1/2) * (2 - eta) / (1 - eta)^3

        where eta = pi * d^3 * n / 6 is the reduced density (packing fraction).

        The factor Y enhances the collision frequency in dense regions,
        accounting for the finite volume of molecules. In the dilute limit
        (eta -> 0), Y -> 1, recovering standard DSMC.

        The number of candidate collisions per cell follows from Eq. (19):
            Nc_cell = 0.5 * N * (N-1) * sigma * Y * cr_max * Fn * dt / V_cell

        Each candidate pair is accepted with probability cr / cr_max,
        consistent with the majorant frequency scheme (Eq. 23-24).

        Args:
            dt: Time step for the collision selection.
            cr_max: Current estimate of the maximum relative speed across all
                    cells. Must be tracked and updated across timesteps.

        Returns:
            collision_pairs: List of length nr_cells. Each element is a list of
                            [i, j] pairs (global particle indices) accepted
                            for collision in that cell.
            cr_max_updated:  Updated maximum relative speed, incorporating the
                            largest relative speed observed during this selection.
        """
        if self.volume is None or self.nr_cells is None:
            raise ValueError(
                "Simulation domain and grid must be initialized before selecting collision pairs."
            )
        if self.Xref is None or self.cell_counts is None:
            raise ValueError(
                "Particles must be initialized and cell indices updated before selecting collision pairs."
            )
        if self.velocities is None:
            raise ValueError(
                "Particle velocities must be initialized before selecting collision pairs."
            )
        d = self.diameter
        sigma = np.pi * d**2  # Hard-sphere total cross section
        Fn = self.N_real / self.N_sim  # Statistical weight per simulation particle
        cell_vol = (
            self.cell_sizes[0] * self.cell_sizes[1] * self.cell_sizes[2]
        )  # Volume of a single cell
        # cell_vol = self.volume / self.nr_cells  # Volume of a single cell (uniform grid)
        N = self.cell_counts  # Number of particles per cell (nr_cells,)

        # ----------------------------------------------------------------
        # Per-cell number density and Enskog pair correlation factor Y(eta)
        # Carnahan-Starling form, Eq. (30) of Frezzotti (1997)
        # ----------------------------------------------------------------
        n_cell = N * Fn / cell_vol  # Physical number density per cell
        eta = np.pi * d**3 * n_cell / 6.0  # Packing fraction per cell
        eta = np.clip(eta, 0.0, 0.98)  # Guard against singularity at eta = 1
        Y = 0.5 * (2.0 - eta) / (1.0 - eta) ** 3  # Pair correlation at contact

        # ----------------------------------------------------------------
        # Candidate collision count per cell (Enskog-modified NTC)
        #
        # Standard NTC (Bird):  M = 0.5 * N*(N-1) * sigma * cr_max * Fn * dt / V_cell
        # Enskog modification:  multiply by Y(eta) to enhance collision rate
        #                       in dense regions, following Eq. (5)-(6) and (19).
        #
        # This is a majorant (upper bound) for the true collision count;
        # the acceptance step below filters to the correct rate.
        # ----------------------------------------------------------------
        M_float = 0.5 * N * (N - 1) * sigma * Y * cr_max * Fn * dt / cell_vol
        M_float = np.where(N < 2, 0.0, M_float)  # Need at least 2 particles

        # Stochastic rounding: floor + probabilistic carry of the fractional part
        M_int = np.floor(M_float).astype(int)
        fractional = M_float - M_int
        M_int += (self.rng.random(self.nr_cells) < fractional).astype(int)

        total_candidates = int(M_int.sum())

        # Early exit if no candidates anywhere
        if total_candidates == 0:
            return [[] for _ in range(self.nr_cells)], cr_max

        # ----------------------------------------------------------------
        # Build a cell-sorted index map so that particles in the same cell
        # occupy a contiguous block. This enables O(1) random pair selection
        # within any cell.
        #
        # sorted_order[cell_start[c] : cell_start[c+1]] gives the global
        # indices of particles residing in cell c.
        # ----------------------------------------------------------------
        sorted_order = np.argsort(self.Xref, kind="mergesort")
        cell_start = np.zeros(self.nr_cells + 1, dtype=int)
        np.cumsum(N, out=cell_start[1:])

        # ----------------------------------------------------------------
        # Vectorised candidate-pair generation
        #
        # For every candidate collision, we need:
        #   1. The cell it belongs to
        #   2. Two distinct random particles from that cell
        #   3. An accept/reject decision based on cr / cr_max
        #
        # All of this is done in bulk using NumPy operations.
        # ----------------------------------------------------------------

        # Expand cell IDs: one entry per candidate collision
        cell_ids = np.repeat(np.arange(self.nr_cells), M_int)

        # Number of particles in the cell of each candidate
        n_in_cell = N[cell_ids]

        # Offset into sorted_order for the cell of each candidate
        off = cell_start[cell_ids]

        # Draw two distinct random local indices within each cell
        r1 = self.rng.integers(0, n_in_cell)  # First particle (local index)
        r2 = self.rng.integers(
            0, n_in_cell - 1
        )  # Second particle (local index, one fewer choice)
        r2 = np.where(r2 >= r1, r2 + 1, r2)  # Shift to avoid picking the same particle

        # Map local cell indices to global particle indices
        idx_i = sorted_order[off + r1]
        idx_j = sorted_order[off + r2]

        # ----------------------------------------------------------------
        # Accept/reject step (majorant frequency scheme)
        #
        # The candidate count was computed with cr_max (an upper bound on
        # relative speed). Each candidate pair is accepted as a real
        # collision with probability cr / cr_max, where cr is the actual
        # relative speed of the pair. This ensures the correct collision
        # rate on average.  See Eq. (23)-(24) of Frezzotti (1997) and the
        # null-collision technique of Koura (Ref. 15).
        # ----------------------------------------------------------------
        dv = self.velocities[idx_i] - self.velocities[idx_j]
        cr = np.sqrt(np.sum(dv * dv, axis=1))

        # Update cr_max with the largest relative speed seen this step.
        # This keeps the majorant tight and minimises false collisions.
        if len(cr) > 0:
            cr_max_updated = max(cr_max, float(cr.max()))
        else:
            cr_max_updated = cr_max

        # Accept with probability cr / cr_max
        accept_mask = self.rng.random(total_candidates) < (cr / cr_max_updated)

        # ----------------------------------------------------------------
        # Vectorized deduplication: ensure each particle collides at most
        # once per timestep.
        #
        # In the dilute (Boltzmann) limit this constraint is almost never
        # triggered because the expected number of collisions per particle
        # per timestep is << 1. In the dense (Enskog) regime, however,
        # the pair correlation factor Y(eta) significantly boosts the
        # collision rate, making it much more likely that a particle is
        # drawn into multiple candidate pairs. Allowing double-collisions
        # would violate energy/momentum conservation and overcount the
        # collision integral.
        #
        # The PANTERA Fortran reference code enforces the same rule via
        # the HAS_REACTED flag in VAHEDI_COLLIS.
        #
        # Strategy — iterative "first-pair" matching (fully vectorized):
        #
        #   For every particle, find the earliest candidate pair it appears
        #   in (via np.minimum.at). A candidate is accepted only if it is
        #   the earliest for BOTH of its particles — this guarantees no
        #   particle is claimed twice within a round.
        #
        #   Accepted particles are marked as used and their candidates are
        #   removed. Among the remaining candidates, some particles that
        #   lost out to a conflict in round 1 may now be free. The process
        #   repeats until no new pairs can be formed.
        #
        #   Each round is O(K) in NumPy (K = remaining candidates).
        #   Convergence typically takes 2–3 rounds because conflicts are
        #   sparse when pairs are drawn at random.
        # ----------------------------------------------------------------
        accepted_cells = cell_ids[accept_mask]
        accepted_i = idx_i[accept_mask]
        accepted_j = idx_j[accept_mask]
        n_accepted = len(accepted_i)

        if n_accepted == 0:
            return [[] for _ in range(self.nr_cells)], cr_max_updated

        keep = np.zeros(n_accepted, dtype=bool)
        used = np.zeros(self.N_sim, dtype=bool)
        alive = np.ones(n_accepted, dtype=bool)  # candidates still eligible

        SENTINEL = n_accepted  # larger than any valid candidate index

        while True:
            # Indices (into accepted_*) of candidates still in play
            cand_idx = np.where(alive)[0]
            if len(cand_idx) == 0:
                break

            ci = accepted_i[cand_idx]
            cj = accepted_j[cand_idx]

            # For each particle, find the earliest alive candidate it
            # belongs to. np.minimum.at scatters the minimum candidate
            # index into each particle's slot.
            first_pair = np.full(self.N_sim, SENTINEL, dtype=np.intp)
            np.minimum.at(first_pair, ci, cand_idx)
            np.minimum.at(first_pair, cj, cand_idx)

            # A candidate k is accepted if it is the first for BOTH its
            # particles — meaning neither particle has a conflicting,
            # earlier candidate.
            is_first_for_i = first_pair[ci] == cand_idx
            is_first_for_j = first_pair[cj] == cand_idx
            round_accept = is_first_for_i & is_first_for_j

            if not round_accept.any():
                break  # Only unresolvable conflicts remain

            # Commit winners
            winners = cand_idx[round_accept]
            keep[winners] = True
            used[accepted_i[winners]] = True
            used[accepted_j[winners]] = True

            # Remove winners from the alive set
            alive[winners] = False

            # Remove any remaining candidate that touches a now-used particle
            still = np.where(alive)[0]
            if len(still) == 0:
                break
            disqualified = used[accepted_i[still]] | used[accepted_j[still]]
            alive[still[disqualified]] = False

        # ----------------------------------------------------------------
        # Group kept pairs by cell (vectorized via argsort + bincount)
        # ----------------------------------------------------------------
        kept_cells = accepted_cells[keep]
        kept_i = accepted_i[keep]
        kept_j = accepted_j[keep]

        collision_pairs = [[] for _ in range(self.nr_cells)]

        if len(kept_cells) > 0:
            # Sort by cell so we can split into per-cell blocks
            order = np.argsort(kept_cells, kind="mergesort")
            sorted_cells = kept_cells[order]
            sorted_pairs = np.column_stack((kept_i[order], kept_j[order]))

            counts = np.bincount(sorted_cells, minlength=self.nr_cells)
            offsets = np.zeros(self.nr_cells + 1, dtype=int)
            np.cumsum(counts, out=offsets[1:])

            # Split the sorted array into per-cell sub-arrays
            splits = np.split(sorted_pairs, offsets[1:-1])
            collision_pairs = [s.tolist() for s in splits]

        return collision_pairs, cr_max_updated

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

        cr_max = np.sqrt(
            16 * self._kB * self.temperature / (np.pi * self.mass)
        )  # ~mean relative speed as initial estimate

        start_time = time()

        for step in range(nr_steps):
            self.update_positions(dt)
            self.update_cell_indices()
            collision_pairs, cr_max = self.select_collision_pairs(dt=dt, cr_max=cr_max)
            pairs_as_arrays = [
                np.array(cell_pairs) if cell_pairs else np.empty((0, 2), dtype=int)
                for cell_pairs in collision_pairs
            ]
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

            # Print some stats every 100 steps:
            if step % 100 == 0:
                print(
                    f"Step {step}/{nr_steps} | nr_collisions={sum(len(p) for p in collision_pairs)} | T_trans = {stats['T_trans_mean'][step]:.2f} | T_rot = {stats['T_rot_mean'][step]:.2f}"
                )

        self.stats = stats
        end_time = time()
        print(f"Simulation took {end_time - start_time:.2f} seconds.")

    def get_stats(self):
        """Return the energy history of the simulation."""
        if not hasattr(self, "stats"):
            raise ValueError("Simulation must be run before getting energy history.")
        return self.stats
