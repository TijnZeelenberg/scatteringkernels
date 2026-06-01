"""Generate a LAMMPS data file for rigid H2 with separately controlled
translational and rotational temperatures.

For each molecule we sample:
  * V_com from Maxwell-Boltzmann at T_trans (3D, on molecule mass)
  * omega from Maxwell-Boltzmann at T_rot, projected onto the plane
    perpendicular to the bond axis (2 rotational DOF for a linear molecule)
Atom positions and velocities are then computed as
  v_atom = V_com + omega x r_rel
so the rigid-body constraint is already satisfied at t = 0.
"""

from pathlib import Path

import numpy as np

# Physical constants (SI). Atom mass matches `mass 1 1.67372e-27` in in.h2relaxation.
KB = 1.380649e-23  # Boltzmann constant
M_ATOM = 1.67372e-27  # mass of a single H atom
D_BOND = 7.4e-11  # [m] bond length of H2 (0.074 nm)

# --- Parameters ---
N_MOL = 20000  # number of H2 molecules
L = 1.0e-7  # cubic box edge [m]
T_TRANS = 300.0  # initial translational temperature [K]
T_ROT = 100.0  # initial rotational temperature [K]
SEED = 42
OUT = Path(__file__).parent / "h2_init.data"


def generate_initial_state(N_mol, L, T_trans, T_rot, seed):
    M_mol = 2 * M_ATOM
    I_perp = M_ATOM * D_BOND**2 / 2

    rng = np.random.default_rng(seed)

    n_side = int(np.ceil(N_mol ** (1 / 3)))
    spacing = L / n_side
    grid_1d = (np.arange(n_side) + 0.5) * spacing
    gx, gy, gz = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
    com = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])[:N_mol]

    phi = rng.uniform(0, 2 * np.pi, N_mol)
    costheta = rng.uniform(-1, 1, N_mol)
    sintheta = np.sqrt(1 - costheta**2)
    bond_axis = np.column_stack(
        [
            sintheta * np.cos(phi),
            sintheta * np.sin(phi),
            costheta,
        ]
    )

    sigma_v = np.sqrt(KB * T_trans / M_mol)
    V_com = rng.normal(0.0, sigma_v, (N_mol, 3))
    V_com -= V_com.mean(axis=0)

    sigma_omega = np.sqrt(KB * T_rot / I_perp)
    omega3 = rng.normal(0.0, sigma_omega, (N_mol, 3))
    parallel = np.einsum("ij,ij->i", omega3, bond_axis)[:, None] * bond_axis
    omega = omega3 - parallel

    half = (D_BOND / 2) * bond_axis
    r1 = com + half
    r2 = com - half
    v1 = V_com + np.cross(omega, half)
    v2 = V_com + np.cross(omega, -half)

    T_trans_real = M_mol * np.mean(np.sum(V_com**2, axis=1)) / (3 * KB)
    T_rot_real = I_perp * np.mean(np.sum(omega**2, axis=1)) / (2 * KB)

    return r1, r2, v1, v2, T_trans_real, T_rot_real


def write_lammps_data(path, r1, r2, v1, v2, L):
    N_mol = r1.shape[0]
    N_atoms = 2 * N_mol

    atom_id = np.arange(1, N_atoms + 1)
    mol_id = np.repeat(np.arange(1, N_mol + 1), 2)
    atype = np.ones(N_atoms, dtype=int)
    positions = np.empty((N_atoms, 3))
    positions[0::2] = r1
    positions[1::2] = r2
    velocities = np.empty((N_atoms, 3))
    velocities[0::2] = v1
    velocities[1::2] = v2

    atoms_arr = np.column_stack([atom_id, mol_id, atype, positions])
    vel_arr = np.column_stack([atom_id, velocities])
    bonds_arr = np.column_stack(
        [
            np.arange(1, N_mol + 1),
            np.ones(N_mol, dtype=int),
            np.arange(1, N_atoms, 2),
            np.arange(2, N_atoms + 1, 2),
        ]
    )

    with open(path, "w") as f:
        f.write("LAMMPS data file: rigid H2 with split T_trans, T_rot\n\n")
        f.write(f"{N_atoms} atoms\n")
        f.write(f"{N_mol} bonds\n")
        f.write("1 atom types\n")
        f.write("1 bond types\n\n")
        f.write(f"0.0 {L:.9e} xlo xhi\n")
        f.write(f"0.0 {L:.9e} ylo yhi\n")
        f.write(f"0.0 {L:.9e} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write(f"1 {M_ATOM:.9e}\n\n")
        f.write("Atoms # molecular\n\n")
        np.savetxt(f, atoms_arr, fmt="%d %d %d %.9e %.9e %.9e")
        f.write("\nVelocities\n\n")
        np.savetxt(f, vel_arr, fmt="%d %.9e %.9e %.9e")
        f.write("\nBonds\n\n")
        np.savetxt(f, bonds_arr, fmt="%d %d %d %d")


if __name__ == "__main__":
    r1, r2, v1, v2, T_trans_real, T_rot_real = generate_initial_state(
        N_MOL, L, T_TRANS, T_ROT, SEED
    )
    write_lammps_data(OUT, r1, r2, v1, v2, L)

    print(f"Wrote {2 * N_MOL} atoms / {N_MOL} bonds to {OUT}")
    print(f"  Box       : {L * 1e9:.2f} nm  (L = {L:.3e} m)")
    print(f"  T_trans   : target {T_TRANS:.1f} K  realised {T_trans_real:.1f} K")
    print(f"  T_rot     : target {T_ROT:.1f} K  realised {T_rot_real:.1f} K")
