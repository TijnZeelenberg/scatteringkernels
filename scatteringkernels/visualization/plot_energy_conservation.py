import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from physics.dsmc import DSMC_Simulation
from physics.borgnakkelarssen_model import borgnakke_larssen_model
from machinelearning.mdn_model import MixtureDensityNetwork
from config.experiment_config import ExperimentConfig


############################
# User-editable parameters #
############################

# MDN model checkpoint
MDN_MODEL_PATH = "results/models/mdn_H2H2.pth"

# Simulation setup
SEED = 42
NR_PARTICLES = 5000
MOLECULES_PER_PARTICLE = 100  # statistical weight (does not change molecular speed scale)
TEMPERATURE = 300.0
BOX_SIZE = 1e-3
NR_CELLS = 50
DT = 4e-7
N_STEPS = 3000

# Collision selection (placeholder; affects collision rate)
COLLISION_PROBABILITY = 0.1

# Diagnostics
CHECK_CONSERVATION_PER_COLLISION = False
PLOT_RELATIVE_DRIFT = True


def molecular_mass_h2() -> float:
    # 2.016 g/mol = 2.016e-3 kg/mol
    return float(2.016e-3 / 6.022e23)


def total_energy(sim: DSMC_Simulation, *, weighted: bool) -> float:
    """Total energy = translational + rotational.

    If weighted=True, returns physical total energy (scaled by statistical weight).
    If weighted=False, returns energy per simulator particle sum (still should be conserved).
    """
    weight = float(sim.particles_per_molecule) if weighted else 1.0
    kinetic = 0.5 * sim.mass * float(weight) * float(np.sum(np.sum(sim.velocities.astype(np.float64) ** 2, axis=1)))
    rotational = float(weight) * float(np.sum(sim.rotational_energies.astype(np.float64)))
    return kinetic + rotational


def run_energy_trace(*, collision_model, label: str) -> dict:
    sim = DSMC_Simulation(random_seed=SEED)
    sim.initialize_domain(box_size=BOX_SIZE, nr_cells=NR_CELLS, boundary="specular")
    sim.initialize_particles(
        nr_particles=NR_PARTICLES,
        molecules_per_particle=MOLECULES_PER_PARTICLE,
        mass=molecular_mass_h2(),
        temperature=TEMPERATURE,
        particle_distribution="uniform",
    )

    e_weighted = np.empty(N_STEPS + 1, dtype=np.float64)
    e_unweighted = np.empty(N_STEPS + 1, dtype=np.float64)
    e_weighted[0] = total_energy(sim, weighted=True)
    e_unweighted[0] = total_energy(sim, weighted=False)

    for t in tqdm(range(1, N_STEPS + 1), desc=f"{label}: sim", leave=False):
        sim.update_positions(DT)
        pairs = sim.select_collision_pairs(collision_probability=COLLISION_PROBABILITY)
        sim.perform_collisions(collision_model, pairs, check_conservation=CHECK_CONSERVATION_PER_COLLISION)
        e_weighted[t] = total_energy(sim, weighted=True)
        e_unweighted[t] = total_energy(sim, weighted=False)

    return {
        "label": label,
        "e_weighted": e_weighted,
        "e_unweighted": e_unweighted,
    }


def build_mdn() -> MixtureDensityNetwork:
    config = ExperimentConfig()
    mdn = MixtureDensityNetwork(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        num_mixtures=config.num_mixtures,
        hidden_dim=config.hidden_dim,
        randomseed=config.random_seed,
    )
    mdn.load_model(MDN_MODEL_PATH)
    return mdn


def main() -> None:
    bl = borgnakke_larssen_model(rng=np.random.default_rng(SEED))
    mdn = build_mdn()

    res_bl = run_energy_trace(collision_model=bl, label="BL")
    res_mdn = run_energy_trace(collision_model=mdn, label="MDN")

    t = np.arange(N_STEPS + 1) * DT

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)

    def plot_trace(res: dict, key: str, name: str):
        E = res[key]
        if PLOT_RELATIVE_DRIFT:
            y = (E - E[0]) / E[0]
            ax.plot(t, y, linewidth=1.5, label=f"{name} (rel. drift)")
        else:
            ax.plot(t, E, linewidth=1.5, label=name)

    # Weighted energy corresponds to physical total energy.
    plot_trace(res_bl, "e_weighted", "BL")
    plot_trace(res_mdn, "e_weighted", "MDN")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("(E - E0) / E0" if PLOT_RELATIVE_DRIFT else "total energy [J]")
    ax.set_title("Total energy conservation (specular box)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Print end-of-run drift numbers for quick inspection.
    for res in (res_bl, res_mdn):
        E = res["e_weighted"]
        drift = float((E[-1] - E[0]) / E[0])
        print(f"{res['label']} weighted relative drift: {drift:.3e}")

    plt.show()


if __name__ == "__main__":
    main()
