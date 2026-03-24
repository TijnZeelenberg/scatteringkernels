# Energy relaxation experiment
from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn import MixtureDensityNetwork
from visualization.plot import plot_energy_relaxation, plot_total_energy

# --- simulation parameters ---
pressure = 100  # Pa
box_size = 7.5e-6  # m
volume = box_size**3  # m^3
dt = 1e-9  # time step (s)
nr_steps = 30000  # number of time steps
trans_temperature = 300  # K
rot_temperature = 300  # K
gas_constant = 8.314  # J/(mol*K)
kB = 1.380649e-23  # J/K
mass = 3.34e-27  # mass of H2 molecule (kg)
d_H2 = 2.89e-10  # effective diameter of H2 (m)
nr_particles = 10000  # number of simulated particles

n_real = pressure / (kB * trans_temperature)  # number density (1/m^3)
n_sim = nr_particles / volume  # simulated number density (1/m^3)
Fn = n_real / n_sim  # scaling factor for collision frequency

# --- set up collision model ---
bl = borgnakke_larssen_model(randomseed=42)
mdn = MixtureDensityNetwork(
    input_dim=3, output_dim=2, num_mixtures=5, hidden_dim=128, randomseed=42
)
mdn.load_model("results/models/mdn_H2H2.pth")

# --- set up DSMC simulation ---
dsmc = DSMC_Simulation(random_seed=42)
dsmc.create_box(box_size=box_size)
dsmc.create_grid(x_cells=10, y_cells=10, z_cells=10)
dsmc.create_particles(
    nr_particles=nr_particles,
    Fn=Fn,
    mass=mass,
    d=d_H2,
    trans_temperature=trans_temperature,
    rot_temperature=rot_temperature,
    particle_distribution="uniform",
)

dsmc.run_simulation(
    nr_steps=nr_steps,
    dt=dt,
    collision_model=bl,
)

# --- plot energy relaxation ---
stats = dsmc.get_stats()
plot_energy_relaxation(stats)
