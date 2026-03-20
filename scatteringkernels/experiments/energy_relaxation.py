from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from visualization.plot import plot_energy_relaxation

# --- simulation parameters ---
pressure = 1  # Pa
box_size = 1e-5  # m
volume = box_size**3  # m^3
temperature = 300  # K
gas_constant = 8.314  # J/(mol*K)
molecules_per_particle = (
    pressure * volume / (gas_constant * temperature)
)  # ideal gas law: n = PV/RT
mass = molecules_per_particle * 2.016 / 6.022e23

# --- set up collision model ---
bl_model = borgnakke_larssen_model(randomseed=42)

# --- set up DSMC simulation ---
dsmc = DSMC_Simulation(random_seed=42)
dsmc.create_box(box_size=1e-5)
dsmc.create_grid(nr_cells=100)
dsmc.create_particles(
    nr_particles=10000,
    mass=mass,
    temperature=temperature,
    particle_distribution="uniform",
)

velocities = dsmc.velocities
positions = dsmc.positions

dsmc.run_simulation(
    nr_steps=5000,
    dt=1e-5,
    collision_model=bl_model,
)

energy_history = dsmc.get_energy_history()

plot_energy_relaxation(energy_history)
