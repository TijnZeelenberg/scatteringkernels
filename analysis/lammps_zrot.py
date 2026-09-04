"""Compute the inverse rotational collision number 1/Z_rot from a LAMMPS
energy-relaxation trace.

Method
------
For a two-temperature (trans, rot) system with energy conservation
    (3/2) T_trans + (zeta_rot/2) T_rot = const
the Jeans relaxation
    dT_rot/dt = (T_trans - T_rot) / tau_rot
combined with conservation gives
    d(T_trans - T_rot)/dt = -(1 + zeta_rot/3) (T_trans - T_rot) / tau_rot
i.e. (T_trans - T_rot) decays exponentially at rate
    lambda = (1 + zeta_rot/3) / tau_rot
For a linear diatomic (zeta_rot = 2): lambda = (5/3)/tau_rot.

Z_rot is defined by tau_rot = Z_rot * tau_coll, with tau_coll = 1/nu_coll and
    nu_coll = n * sigma * <v_r>,   sigma = pi d^2 (hard-sphere/VHS at d_ref),
    <v_r>   = sqrt(8 k_B T_eq / (pi mu)),   mu = m/2 .
Therefore
    1/Z_rot = tau_coll / tau_rot = (3 lambda) / (5 nu_coll)   for diatomics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from physics.species import Species

KB = 1.380649e-23


def fit_relaxation_rate(t: np.ndarray, dT: np.ndarray, floor_frac: float = 0.05):
    """Linear fit of log(dT) vs t over the resolved decay window.

    Drops samples where dT has fallen below `floor_frac` of its initial value
    (noise-dominated tail) and any non-positive dT.
    """
    threshold = floor_frac * abs(dT[0])
    mask = dT > threshold
    if mask.sum() < 10:
        raise RuntimeError(
            f"Not enough points above the noise floor ({mask.sum()} kept); "
            "lower floor_frac or check the input data."
        )
    slope = np.polyfit(t[mask], np.log(dT[mask]), 1)[0]
    return float(-slope), mask


def compute_inverse_zrot(
    lammps_path: str | Path,
    species: Species,
    n_molecules: int,
    box_size: float,
    zeta_rot: int = 2,
    floor_frac: float = 0.05,
) -> dict:
    arr = np.loadtxt(str(lammps_path), skiprows=1, ndmin=2)
    t = arr[:, 1]
    T_trans = arr[:, 2]
    T_rot = arr[:, 3]

    T_eq = (3.0 * T_trans[0] + zeta_rot * T_rot[0]) / (3.0 + zeta_rot)
    dT = T_trans - T_rot

    decay_rate, mask = fit_relaxation_rate(t, dT, floor_frac=floor_frac)
    tau_rot = (1.0 + zeta_rot / 3.0) / decay_rate

    n_density = n_molecules / box_size**3
    mu = species.mass / 2.0
    v_r_mean = np.sqrt(8.0 * KB * T_eq / (np.pi * mu))
    sigma = np.pi * species.diameter**2
    nu_coll = n_density * sigma * v_r_mean
    tau_coll = 1.0 / nu_coll

    Z_rot = tau_rot / tau_coll
    inv_Z_rot = 1.0 / Z_rot

    return {
        "T_trans_0": float(T_trans[0]),
        "T_rot_0": float(T_rot[0]),
        "T_eq": float(T_eq),
        "decay_rate": float(decay_rate),
        "tau_rot": float(tau_rot),
        "n_density": float(n_density),
        "v_r_mean": float(v_r_mean),
        "sigma": float(sigma),
        "nu_coll": float(nu_coll),
        "tau_coll": float(tau_coll),
        "Z_rot": float(Z_rot),
        "inv_Z_rot": float(inv_Z_rot),
        "n_fit_points": int(mask.sum()),
        "n_total_points": len(t),
    }


def _format_report(r: dict) -> str:
    return (
        f"Initial T_trans / T_rot      : {r['T_trans_0']:.1f} / {r['T_rot_0']:.1f} K\n"
        f"Equipartition T_eq           : {r['T_eq']:.1f} K\n"
        f"Exponential decay rate lambda: {r['decay_rate']:.3e} /s\n"
        f"Rotational relaxation tau_rot: {r['tau_rot']:.3e} s\n"
        f"Number density n             : {r['n_density']:.3e} /m^3\n"
        f"Mean relative speed <v_r>    : {r['v_r_mean']:.1f} m/s\n"
        f"Cross-section sigma = pi d^2 : {r['sigma']:.3e} m^2\n"
        f"Collision freq nu_coll       : {r['nu_coll']:.3e} /s\n"
        f"Mean collision time tau_coll : {r['tau_coll']:.3e} s\n"
        f"Fit window                   : {r['n_fit_points']} / {r['n_total_points']} samples\n"
        f"------------------------------\n"
        f"Z_rot                        : {r['Z_rot']:.3f}\n"
        f"1 / Z_rot                    : {r['inv_Z_rot']:.4f}\n"
    )


def main():
    p = argparse.ArgumentParser(
        description="Compute 1/Z_rot from a LAMMPS energy-relaxation trace."
    )
    p.add_argument(
        "--lammps",
        type=str,
        default="lammps/output/lammps_H2_energy_relaxation.dat",
        help="Path to the LAMMPS energy-relaxation .dat file.",
    )
    p.add_argument("--species", type=str, default="H2", choices=["H2", "O2"])
    p.add_argument(
        "--N", type=int, default=20000, help="number of molecules in the LAMMPS run"
    )
    p.add_argument("--L", type=float, default=2.0e-8, help="cubic box edge [m]")
    p.add_argument(
        "--floor-frac",
        type=float,
        default=0.05,
        help="Drop dT samples below this fraction of dT(0) when fitting (noise floor).",
    )
    args = p.parse_args()

    species = Species.H2() if args.species == "H2" else Species.O2()
    r = compute_inverse_zrot(
        lammps_path=args.lammps,
        species=species,
        n_molecules=args.N,
        box_size=args.L,
        floor_frac=args.floor_frac,
    )
    print(_format_report(r))


if __name__ == "__main__":
    main()
