"""Beta-MDN wf-sweep visualization — thin wrapper around `run_wf_sweep_experiments`."""

from __future__ import annotations

import matplotlib.pyplot as plt

from physics.species import Species
from visualization.wfsweep import run_wf_sweep_experiments


if __name__ == "__main__":
    run_wf_sweep_experiments(
        kind="beta_mdn",
        tag="H2_400000",
        species=Species.H2(),
        sparta_path="data/sparta_H2_energy_relaxationVHS_zinv0151.dat",
    )
    plt.show()
