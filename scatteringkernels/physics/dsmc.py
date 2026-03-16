import numpy as np
from typing import Literal
from time import time

BoundaryCondition = Literal["specular", "absorbing", "diffuse"]
ParticleDistribution = Literal["uniform","central", "gaussian", "left_biased_gaussian", "left_wall"]
dimensions = 3

class DSMC_Simulation:
    """Direct Simulation Monte Carlo (DSMC) implementation.
    """

    def __init__(
        self,
        random_seed=None,
    ):
        self.rng = np.random.default_rng(random_seed)
        self._kB = 1.380649e-23
        self.domain_init = False
        self.particle_init = False

    def initialize_domain(self, box_size:float, nr_cells:int, boundary:BoundaryCondition="specular"):
        self.box_size = box_size
        self.nr_cells = nr_cells
        self.cell_size = box_size / nr_cells
        self.boundary = boundary

        # Diffuse wall parameters (used when boundary == "diffuse").
        # Walls are at x=0 (left) and x=box_size (right).
        self.wall_temperature = None
        self.wall_velocity_left = np.zeros(dimensions, dtype=np.float32)
        self.wall_velocity_right = np.zeros(dimensions, dtype=np.float32)

        # Wall momentum-transfer bookkeeping (only meaningful for boundary == "diffuse").
        self.reset_wall_stats()
        self.domain_init = True

    def reset_wall_stats(self) -> None:
        """Reset accumulated wall momentum transfer and counters."""
        self.wall_impulse_left = np.zeros(dimensions, dtype=np.float64)   # impulse imparted to gas [N*s]
        self.wall_impulse_right = np.zeros(dimensions, dtype=np.float64)  # impulse imparted to gas [N*s]
        self.wall_hits_left = 0
        self.wall_hits_right = 0
        self.wall_sampling_time = 0.0

    def particle_weight(self) -> float:
        """Statistical weight: number of real molecules represented by one simulator particle."""
        return float(getattr(self, "particles_per_molecule", 1))

    def wall_area(self) -> float:
        """Area of the x-walls for the current domain."""
        # Domain is a cube with side length box_size.
        return float(self.box_size ** (dimensions - 1))

    def wall_shear_stress_xy(self) -> tuple[float, float]:
        """Return (tau_left, tau_right) estimated from wall momentum flux.

        tau is the shear stress component associated with y-momentum transfer across an x-normal wall.
        Convention here: tau = (impulse imparted to gas in y) / (A * sampling_time).
        """
        if self.wall_sampling_time <= 0.0:
            return 0.0, 0.0
        A = self.wall_area()
        tau_left = float(self.wall_impulse_left[1] / (A * self.wall_sampling_time))
        tau_right = float(self.wall_impulse_right[1] / (A * self.wall_sampling_time))
        return tau_left, tau_right

    #TODO: find a more readable way to implement diffuse wall reflections.
    def configure_diffuse_walls(
        self,
        *,
        wall_temperature: float | None = None,
        wall_velocity_left: np.ndarray | None = None,
        wall_velocity_right: np.ndarray | None = None,
    ) -> None:
        """Configure parameters for the moving diffuse (Maxwell) walls.

        For Couette flow, set equal and opposite tangential wall velocities.
        Example (shear in y):
            wall_velocity_left  = [0, -U/2, 0]
            wall_velocity_right = [0, +U/2, 0]
        """
        if wall_temperature is not None:
            if not np.isfinite(wall_temperature) or wall_temperature <= 0.0:
                raise ValueError("wall_temperature must be a positive finite float")
            self.wall_temperature = float(wall_temperature)
        if wall_velocity_left is not None:
            v = np.asarray(wall_velocity_left, dtype=np.float32)
            if v.shape != (dimensions,):
                raise ValueError(f"wall_velocity_left must have shape ({dimensions},)")
            self.wall_velocity_left = v
        if wall_velocity_right is not None:
            v = np.asarray(wall_velocity_right, dtype=np.float32)
            if v.shape != (dimensions,):
                raise ValueError(f"wall_velocity_right must have shape ({dimensions},)")
            self.wall_velocity_right = v

    def _apply_diffuse_walls(self) -> None:
        """Apply moving diffuse (Maxwell) reflection at x-walls.

        Particles that have crossed x=0 or x=L are re-emitted from the wall with:
        - Normal component sampled from half-range Maxwellian (points into the domain)
        - Tangential components sampled from Maxwellian with mean equal to wall velocity

        This is a simplified reflection model (does not account for the exact wall-hit time
        within the timestep). It is appropriate when dt is small compared to mean free time.
        """
        # Choose wall temperature: default to current gas temperature.
        T_wall = self.temperature if self.wall_temperature is None else self.wall_temperature
        sigma = float(np.sqrt(self._kB * T_wall / self.mass))
        if not np.isfinite(sigma) or sigma <= 0.0:
            return

        eps = 1e-7 * self.box_size

        left = self.positions[:, 0] < 0.0
        if np.any(left):
            n = int(np.count_nonzero(left))
            v_before = self.velocities[left].astype(np.float64, copy=True)
            # Place particles just inside the domain.
            self.positions[left, 0] = eps
            # Sample half-range normal speed (v_x > 0).
            r = np.clip(self.rng.random(n), 1e-12, 1.0)
            vnx = sigma * np.sqrt(-2.0 * np.log(r))
            self.velocities[left, 0] = vnx.astype(np.float32)
            # Tangential components with wall drift.
            drift = self.wall_velocity_left
            self.velocities[left, 1] = (drift[1] + self.rng.normal(0.0, sigma, size=n)).astype(np.float32)
            self.velocities[left, 2] = (drift[2] + self.rng.normal(0.0, sigma, size=n)).astype(np.float32)

            v_after = self.velocities[left].astype(np.float64, copy=False)
            # Impulse imparted to the gas must be scaled by the simulator particle weight.
            self.wall_impulse_left += self.particle_weight() * self.mass * (v_after - v_before).sum(axis=0)
            self.wall_hits_left += n

        right = self.positions[:, 0] >= self.box_size
        if np.any(right):
            n = int(np.count_nonzero(right))
            v_before = self.velocities[right].astype(np.float64, copy=True)
            self.positions[right, 0] = self.box_size - eps
            # Sample half-range normal speed (v_x < 0).
            r = np.clip(self.rng.random(n), 1e-12, 1.0)
            vnx = sigma * np.sqrt(-2.0 * np.log(r))
            self.velocities[right, 0] = (-vnx).astype(np.float32)
            drift = self.wall_velocity_right
            self.velocities[right, 1] = (drift[1] + self.rng.normal(0.0, sigma, size=n)).astype(np.float32)
            self.velocities[right, 2] = (drift[2] + self.rng.normal(0.0, sigma, size=n)).astype(np.float32)

            v_after = self.velocities[right].astype(np.float64, copy=False)
            self.wall_impulse_right += self.particle_weight() * self.mass * (v_after - v_before).sum(axis=0)
            self.wall_hits_right += n

    def set_particle_positions(
        self,
        nr_particles: int,
        distribution_type: ParticleDistribution | str,
        dimensions: int,
    ) -> np.ndarray:
        if distribution_type == "uniform":
            return self.rng.uniform(0.0, self.box_size, size=(nr_particles, dimensions)).astype(np.float32)
        if distribution_type == "gaussian":
            return self.rng.normal(self.box_size / 2, self.box_size / 32, size=(nr_particles, dimensions)).astype(np.float32)

        positions = self.rng.uniform(0.0, self.box_size, size=(nr_particles, dimensions)).astype(np.float32)
        if distribution_type == "central":
            positions[:, 0] = self.box_size / 2
            return positions
        if distribution_type == "left_biased_gaussian":
            positions[:, 0] = self.rng.uniform(0.0, 0.25 * self.box_size, size=nr_particles)
            return positions
        if distribution_type == "left_wall":
            positions[:, 0] = 0.0
            return positions

        raise ValueError(
            f"Unknown particle_distribution: {distribution_type}. "
            "Use 'uniform', 'central', 'gaussian', 'left_biased_gaussian', or 'left_wall'."
        )

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

        self.positions = self.set_particle_positions(
            nr_particles=nr_particles,
            distribution_type=particle_distribution,
            dimensions=dimensions,
        )

        self.velocities = self.rng.normal(0, np.sqrt(self._kB * temperature/self.mass), size=(nr_particles,dimensions)).astype(np.float32)
        self.rotational_energies = np.zeros(nr_particles, dtype=np.float32) 
        self.cell_indices = self.rng.integers(0, self.nr_cells, size=(nr_particles,))
        self.particle_init = True

    def update_cell_indices(self):
        """Assign each particle to a 1D cell index based on its x-position.

        This code currently uses a 1D spatial mesh (nr_cells) along the first axis.
        """
        if not self.domain_init or not self.particle_init:
            raise ValueError("Domain and particles must be initialized before updating cell indices.")
        # Map x in [0, box_size) to integer cell indices in [0, nr_cells-1].
        x = self.positions[:, 0]
        idx = (x / self.cell_size).astype(np.int32)
        self.cell_indices = np.clip(idx, 0, self.nr_cells - 1)

    def select_collision_pairs(self, collision_probability=0.5):
        #TODO: implement proper No-Time-Counter method for selecting collision pairs based on relative velocities and collision cross-sections, instead of using a fixed collision probability.
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

    def perform_collisions(
        self,
        collision_model,
        collision_pairs: list[np.ndarray],
        *,
        check_conservation: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-12,
    ):
        """Perform collisions for the selected pairs of particles using the given collision model.
        
        Args:
            collision_model: Function that takes two velocity vectors and returns new velocity vectors.
            collision_pairs: List of arrays containing the indices of the selected collision pairs for each cell.
            check_conservation: If True, assert pair momentum and total energy conservation per collision.
            rtol: Relative tolerance used for the conservation check.
            atol: Absolute tolerance used for the conservation check.
        """

        for pairs in collision_pairs:
            for i, j in pairs:
                # Get the velocities and rotational energies of the two particles
                v_i = self.velocities[i]
                v_j = self.velocities[j]
                e_rot_i = self.rotational_energies[i]
                e_rot_j = self.rotational_energies[j]

                if check_conservation:
                    p_before = self.mass * (v_i + v_j)
                    e_before = (
                        0.5 * self.mass * (float(np.dot(v_i, v_i)) + float(np.dot(v_j, v_j)))
                        + float(e_rot_i)
                        + float(e_rot_j)
                    )

                # Perform collision using the provided collision model
                new_v_i, new_e_rot_i, new_v_j, new_e_rot_j = collision_model.postsample(v_i, e_rot_i, v_j, e_rot_j, m=self.mass, T=self.temperature)

                if check_conservation:
                    p_after = self.mass * (new_v_i + new_v_j)
                    e_after = (
                        0.5 * self.mass * (float(np.dot(new_v_i, new_v_i)) + float(np.dot(new_v_j, new_v_j)))
                        + float(new_e_rot_i)
                        + float(new_e_rot_j)
                    )
                    if (not np.allclose(p_before, p_after, rtol=rtol, atol=atol)) or (not np.isclose(e_before, e_after, rtol=rtol, atol=atol)):
                        raise RuntimeError(
                            "Collision kernel violated conservation. "
                            f"Δp={np.linalg.norm(p_after - p_before):.3e}, ΔE={e_after - e_before:.3e}"
                        )

                # Update the velocities and rotational energies of the particles
                self.velocities[i] = new_v_i
                self.velocities[j] = new_v_j
                self.rotational_energies[i] = new_e_rot_i
                self.rotational_energies[j] = new_e_rot_j

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
            # Reflect particles that go out of bounds.
            # Use a 2L periodic fold so particles remain inside the domain even
            # when a time step crosses multiple walls.
            for d in range(dimensions):
                period = 2.0 * self.box_size
                folded = np.mod(self.positions[:, d], period)
                reflected = folded >= self.box_size
                self.positions[:, d] = np.where(reflected, period - folded, folded)
                self.velocities[reflected, d] *= -1
        elif self.boundary == "diffuse":
            # Moving diffuse (Maxwell) walls at x=0 and x=L.
            # Only x-walls are treated as solid boundaries; y/z remain periodic via folding.
            self._apply_diffuse_walls()

            # Accumulate sampling time for wall-flux statistics.
            self.wall_sampling_time += float(dt)

            # Fold y/z periodically to keep particles in the box.
            for d in (1, 2):
                self.positions[:, d] = np.mod(self.positions[:, d], self.box_size)
        elif self.boundary == "absorbing":
            # Remove particles that go out of bounds
            mask = np.all((self.positions >= 0) & (self.positions < self.box_size), axis=1)
            self.positions = self.positions[mask]
            self.velocities = self.velocities[mask]
            self.cell_indices = self.cell_indices[mask]
        else:
            raise ValueError(f"Unknown boundary condition: {self.boundary}")

        # Cell assignment must follow the position update.
        # (For absorbing boundaries, arrays are already masked above.)
        self.update_cell_indices()

    def run_simulation(self, collision_model, nr_steps:int, dt:float):
        """Run the DSMC simulation for a given number of steps and time step."""

        start_time = time()
        for step in range(nr_steps):
            self.update_positions(dt)
            collision_pairs = self.select_collision_pairs()
            self.perform_collisions(collision_model, collision_pairs)
        end_time = time()
        print(f"Simulation took {end_time - start_time:.2f} seconds.")
