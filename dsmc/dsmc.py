import numpy as np


class DSMC_Simulation:
    """Direct Simulation Monte Carlo (DSMC) implementation.
    """

    def __init__(
        self,
        time_step,
        seed=None,
    ):
        self.time_step = float(time_step)
        self.rng = np.random.default_rng(seed)
        self._kB = 1.380649e-23

    def initialize_particles(self, nr_particles, temperature=300.0):
        self.positions = np.zeros((nr_particles,3),dtype=np.float32)
        self.velocities = np.zeros((nr_particles,3),dtype=np.float32)
        self.cell_indices = np.zeros(nr_particles,dtype=np.int32)

    def initialize_domain(self, box_size, cell_size, boundary="specular"):
        self.box_size = float(box_size)
        self.cell_size = float(cell_size)
        self.boundary = boundary
        self._cells = None

