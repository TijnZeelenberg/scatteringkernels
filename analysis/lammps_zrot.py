"""Compute Z_rot for H2 and O2 from LAMMPS energy-relaxation traces.

Fits the analytical solution of the Jeans equation to T_trans - T_rot:

    dT(t) = dT_0 * exp(-lambda * t)

where lambda = (1 + zeta_rot/3) / tau_rot = (5/3) / tau_rot for diatomics.
Z_rot = tau_rot / tau_coll,  tau_coll = 1 / (n * sigma * <v_r>).
"""

import numpy as np
from scipy.optimize import curve_fit

from physics.species import Species

KB = 1.380649e-23
N_MOLECULES = 20_000
BOX_SIZE = 1.0e-7  # m
ZETA_ROT = 2       # linear diatomic


def _load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, skiprows=1, ndmin=2)
    return arr[:, 1], arr[:, 2], arr[:, 3]  # t, T_trans, T_rot


def _fit(t: np.ndarray, dT: np.ndarray) -> float:
    def model(t, lam, dT0):
        return dT0 * np.exp(-lam * t)

    (lam, _), _ = curve_fit(model, t, dT, p0=[1e9, dT[0]], maxfev=10_000)
    return float(lam)


def compute_zrot(path: str, species: Species) -> dict:
    t, T_trans, T_rot = _load(path)

    T_eq = (3.0 * T_trans[0] + ZETA_ROT * T_rot[0]) / (3.0 + ZETA_ROT)
    dT = T_trans - T_rot

    lam = _fit(t, dT)
    tau_rot = (1.0 + ZETA_ROT / 3.0) / lam

    n = N_MOLECULES / BOX_SIZE**3
    v_r = np.sqrt(8.0 * KB * T_eq / (np.pi * species.mass / 2.0))
    sigma = np.pi * species.diameter**2
    tau_coll = 1.0 / (n * sigma * v_r)

    Z_rot = tau_rot / tau_coll
    return {"T_eq": T_eq, "lambda": lam, "tau_rot": tau_rot, "tau_coll": tau_coll, "Z_rot": Z_rot}


CONFIGS = {
    "H2": (Species.H2(), "data/lammps/h2_energy_relaxation.dat"),
    "O2": (Species.O2(), "data/lammps/o2_energy_relaxation.dat"),
}

for name, (species, path) in CONFIGS.items():
    r = compute_zrot(path, species)
    print(f"{name}:  T_eq = {r['T_eq']:.1f} K  |  lambda = {r['lambda']:.3e} /s"
          f"  |  tau_rot = {r['tau_rot']:.3e} s  |  tau_coll = {r['tau_coll']:.3e} s"
          f"  |  Z_rot = {r['Z_rot']:.3f}")
