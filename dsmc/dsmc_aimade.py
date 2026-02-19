import numpy as np


class DSMC_Simulation:
    """Direct Simulation Monte Carlo (DSMC) skeleton implementation.

    This class follows the structure described in dsmc_structure.md and provides
    a minimal, working DSMC loop with simple hard-sphere collision handling.
    """

    def __init__(
        self,
        num_particles,
        box_size,
        cell_size,
        time_step,
        mass=1.0,
        collision_diameter=1.0,
        boundary="periodic",
        seed=None,
    ):
        self.num_particles = int(num_particles)
        self.box_size = float(box_size)
        self.cell_size = float(cell_size)
        self.time_step = float(time_step)
        self.mass = float(mass)
        self.collision_diameter = float(collision_diameter)
        self.boundary = boundary
        self.rng = np.random.default_rng(seed)

        self.positions = np.zeros((self.num_particles, 3), dtype=float)
        self.velocities = np.zeros((self.num_particles, 3), dtype=float)
        self.cell_indices = np.zeros((self.num_particles,), dtype=int)

        self.num_cells_per_dim = int(np.floor(self.box_size / self.cell_size))
        if self.num_cells_per_dim < 1:
            raise ValueError("cell_size must be smaller than box_size.")
        self.num_cells = self.num_cells_per_dim ** 3
        self.cell_volume = (self.box_size / self.num_cells_per_dim) ** 3

        self._cells = [[] for _ in range(self.num_cells)]

        self._kB = 1.380649e-23

    def initialize_domain(self):
        self.num_cells_per_dim = int(np.floor(self.box_size / self.cell_size))
        if self.num_cells_per_dim < 1:
            raise ValueError("cell_size must be smaller than box_size.")
        self.num_cells = self.num_cells_per_dim ** 3
        self.cell_volume = (self.box_size / self.num_cells_per_dim) ** 3
        self._cells = [[] for _ in range(self.num_cells)]

    def initialize_particles(
        self,
        temperature=300.0,
        density_gradient=None,
        temperature_gradient=None,
        max_density=None,
        max_tries=200000,
    ):
        if density_gradient is None:
            self.positions = self.rng.random((self.num_particles, 3)) * self.box_size
        else:
            if not callable(density_gradient):
                raise TypeError("density_gradient must be callable or None.")
            if max_density is None:
                probe = self.rng.random((2048, 3)) * self.box_size
                max_density = float(np.max(density_gradient(probe)))
                if max_density <= 0.0:
                    raise ValueError("density_gradient must be positive somewhere in the domain.")
            positions = np.zeros((self.num_particles, 3), dtype=float)
            accepted = 0
            attempts = 0
            while accepted < self.num_particles and attempts < max_tries:
                batch = min(1024, self.num_particles - accepted)
                trial = self.rng.random((batch, 3)) * self.box_size
                weights = density_gradient(trial) / max_density
                keep = self.rng.random(batch) < weights
                num_keep = int(np.sum(keep))
                if num_keep > 0:
                    positions[accepted : accepted + num_keep] = trial[keep]
                    accepted += num_keep
                attempts += batch
            if accepted < self.num_particles:
                raise RuntimeError("Failed to sample positions from density_gradient.")
            self.positions = positions

        if temperature_gradient is None:
            std = np.sqrt(self._kB * temperature / self.mass)
            self.velocities = self.rng.normal(loc=0.0, scale=std, size=(self.num_particles, 3))
        else:
            if not callable(temperature_gradient):
                raise TypeError("temperature_gradient must be callable or None.")
            temps = np.asarray(temperature_gradient(self.positions), dtype=float)
            if temps.shape not in [(self.num_particles,), (self.num_particles, 1)]:
                raise ValueError("temperature_gradient must return shape (N,) or (N, 1).")
            temps = temps.reshape(self.num_particles)
            if np.any(temps <= 0.0):
                raise ValueError("temperature_gradient must return positive temperatures.")
            std = np.sqrt(self._kB * temps / self.mass)
            self.velocities = self.rng.normal(loc=0.0, scale=1.0, size=(self.num_particles, 3)) * std[:, None]

    def bin_particles(self):
        self._cells = [[] for _ in range(self.num_cells)]
        scaled = self.positions / (self.box_size / self.num_cells_per_dim)
        ijk = np.floor(scaled).astype(int)
        ijk = np.clip(ijk, 0, self.num_cells_per_dim - 1)
        self.cell_indices = (
            ijk[:, 0]
            + self.num_cells_per_dim * ijk[:, 1]
            + (self.num_cells_per_dim ** 2) * ijk[:, 2]
        )
        for idx, cell_id in enumerate(self.cell_indices):
            self._cells[cell_id].append(idx)

    def select_collisionpairs(self):
        pairs = []
        vr_max = []
        sigma = np.pi * (self.collision_diameter ** 2)
        for cell in self._cells:
            n = len(cell)
            if n < 2:
                pairs.append([])
                vr_max.append(0.0)
                continue
            v = self.velocities[cell]
            vr = v[:, None, :] - v[None, :, :]
            vr_mag = np.linalg.norm(vr, axis=2)
            vrmax = np.max(vr_mag)
            num_pairs = int(0.5 * n * (n - 1) * sigma * vrmax * self.time_step / self.cell_volume)
            cell_pairs = []
            for _ in range(num_pairs):
                i, j = self.rng.choice(cell, size=2, replace=False)
                cell_pairs.append((i, j))
            pairs.append(cell_pairs)
            vr_max.append(vrmax)
        return pairs, vr_max

    def perform_collisions(self, pairs, vr_max):
        for cell_idx, cell_pairs in enumerate(pairs):
            if not cell_pairs:
                continue
            vrmax = vr_max[cell_idx]
            if vrmax <= 0.0:
                continue
            for i, j in cell_pairs:
                dv = self.velocities[i] - self.velocities[j]
                vr = np.linalg.norm(dv)
                if vr / vrmax > self.rng.random():
                    vcm = 0.5 * (self.velocities[i] + self.velocities[j])
                    q = 2.0 * self.rng.random() - 1.0
                    phi = 2.0 * np.pi * self.rng.random()
                    vr1 = vr * np.sqrt(1.0 - q * q) * np.cos(phi)
                    vr2 = vr * np.sqrt(1.0 - q * q) * np.sin(phi)
                    vr3 = vr * q
                    dv_new = 0.5 * np.array([vr1, vr2, vr3])
                    self.velocities[i] = vcm + dv_new
                    self.velocities[j] = vcm - dv_new

    def move_particles(self):
        self.positions += self.velocities * self.time_step
        self.apply_boundaries()

    def apply_boundaries(self):
        if self.boundary == "periodic":
            self.positions %= self.box_size
        elif self.boundary == "reflecting":
            for dim in range(3):
                over = self.positions[:, dim] > self.box_size
                under = self.positions[:, dim] < 0.0
                if np.any(over):
                    self.positions[over, dim] = 2.0 * self.box_size - self.positions[over, dim]
                    self.velocities[over, dim] *= -1.0
                if np.any(under):
                    self.positions[under, dim] = -self.positions[under, dim]
                    self.velocities[under, dim] *= -1.0
        else:
            raise ValueError("boundary must be 'periodic' or 'reflecting'.")

    def sample_macroscopic_properties(self):
        v2 = np.sum(self.velocities ** 2, axis=1)
        mean_v2 = np.mean(v2)
        temperature = (self.mass * mean_v2) / (3.0 * self._kB)
        density = self.num_particles / (self.box_size ** 3)
        pressure = density * self._kB * temperature
        mean_velocity = np.mean(self.velocities, axis=0)
        return {
            "temperature": temperature,
            "pressure": pressure,
            "mean_velocity": mean_velocity,
            "density": density,
        }

    def simulate(self, num_steps, sample_interval=1):
        history = []
        for step in range(int(num_steps)):
            self.bin_particles()
            pairs, vr_max = self.select_collisionpairs()
            self.perform_collisions(pairs, vr_max)
            self.move_particles()
            if step % sample_interval == 0:
                history.append(self.sample_macroscopic_properties())
        return history

    def save_state(self, path):
        np.savez(
            path,
            positions=self.positions,
            velocities=self.velocities,
            box_size=self.box_size,
            cell_size=self.cell_size,
            time_step=self.time_step,
            mass=self.mass,
            collision_diameter=self.collision_diameter,
            boundary=self.boundary,
        )

    def load_state(self, path):
        data = np.load(path, allow_pickle=True)
        self.positions = data["positions"]
        self.velocities = data["velocities"]
        self.box_size = float(data["box_size"])
        self.cell_size = float(data["cell_size"])
        self.time_step = float(data["time_step"])
        self.mass = float(data["mass"])
        self.collision_diameter = float(data["collision_diameter"])
        self.boundary = str(data["boundary"])
        self.num_particles = self.positions.shape[0]
        self.cell_indices = np.zeros((self.num_particles,), dtype=int)
        self.initialize_domain()
