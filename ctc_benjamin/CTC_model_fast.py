import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import multiprocessing
import time

import numpy as np
import numba as nb
import pandas as pd

# Constants
m_H = 1.6738e-27
m_H2 = m_H * 2
sigma_LJ = 3.06e-10
kB = 1.38064852e-23
K_to_EV = 8.617333262145e-5
ev_to_J = 1.60217667e-19
epsilon_LJ = 34.0 * K_to_EV * ev_to_J  # LJ well depth [J]
d_H2 = 0.741e-10
I_mol = 0.5 * (d_H2**2) * m_H

# Simulation settings
dt = 0.1e-15
tsim = 5e-12
nsteps = int(tsim / dt)
ncoll = 40000

T_min = 50
T_max = 500
T_equilibrium = 0.5 * (T_min + T_max)
sample_temperature_mixture = False

m1 = m_H2
m2 = m_H2

outputfile = "data/H2H2_collisionsV2.csv"
varNames = [
    "Etr",
    "Erot1_in",
    "Erot2_in",
    "Etr_out",
    "Erot1_out",
    "Erot2_out",
]


# ─── Numba-compiled helpers (matching CTC_utils exactly) ──────────────────


@nb.njit(cache=True)
def norm3(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@nb.njit(cache=True)
def norm2(a, b):
    return np.sqrt(a * a + b * b)


@nb.njit(cache=True)
def intraatomic_force_nb(xi, xj):
    """Matches CTC_utils.intraatomic_force exactly."""
    drij = norm3(xi - xj)
    if drij < 1e-30:
        return np.zeros(3)

    drijxy = norm2(xi[0] - xj[0], xi[1] - xj[1])
    theta_ij = np.arctan2(xj[2] - xi[2], drijxy)
    phi_ij = np.arctan2(xj[1] - xi[1], xj[0] - xi[0])

    # Lennart-Jones force
    f_mag = -4.0 * epsilon_LJ * (
        6.0 * sigma_LJ**6 / drij**7 - 12.0 * sigma_LJ**12 / drij**13
    )

    f_z = np.sin(theta_ij) * f_mag
    f_xy = np.cos(theta_ij) * f_mag
    f_x = np.cos(phi_ij) * f_xy
    f_y = np.sin(phi_ij) * f_xy

    return np.array([-f_x, -f_y, -f_z])


@nb.njit(cache=True)
def matvec_transpose(R, f):
    """Compute f @ R  (i.e. R^T @ f in column-vector convention).
    For (3,) f and (3,3) R, result[j] = sum_i f[i]*R[i,j]."""
    out = np.empty(3)
    for j in range(3):
        out[j] = f[0] * R[0, j] + f[1] * R[1, j] + f[2] * R[2, j]
    return out


@nb.njit(cache=True)
def get_moments_nb(F13, F14, F23, F24, R1, R2, d):
    """Matches CTC_utils.get_moments exactly.

    Transforms forces to body-fixed frame, then computes torques
    for a linear molecule aligned on the z-axis."""
    half_d = 0.5 * d

    # Transform forces to body frame of molecule 1: f_body = f_lab @ R
    f13_r = matvec_transpose(R1, F13)
    f14_r = matvec_transpose(R1, F14)
    f23_r = matvec_transpose(R1, F23)
    f24_r = matvec_transpose(R1, F24)

    # Newton's 3rd law, transform to body frame of molecule 2
    f31_r = matvec_transpose(R2, -F13)
    f41_r = matvec_transpose(R2, -F14)
    f32_r = matvec_transpose(R2, -F23)
    f42_r = matvec_transpose(R2, -F24)

    # Molecule 1: atom1 at +z, atom2 at -z in body frame
    f_on_atom1 = f13_r + f14_r
    f_on_atom2 = f23_r + f24_r
    m1_x = half_d * (f_on_atom2[1] - f_on_atom1[1])
    m1_y = half_d * (f_on_atom1[0] - f_on_atom2[0])
    m1 = np.array([m1_x, m1_y, 0.0])

    # Molecule 2: atom3 at +z, atom4 at -z in body frame
    f_on_atom3 = f31_r + f32_r
    f_on_atom4 = f41_r + f42_r
    m2_x = half_d * (f_on_atom4[1] - f_on_atom3[1])
    m2_y = half_d * (f_on_atom3[0] - f_on_atom4[0])
    m2 = np.array([m2_x, m2_y, 0.0])

    return m1, m2


@nb.njit(cache=True)
def get_rdot_nb(w, R):
    """Matches CTC_utils.get_rdot: Rdot = R @ w_tilde.

    w_tilde = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
    """
    wx, wy, wz = w[0], w[1], w[2]
    Rdot = np.empty((3, 3))
    for i in range(3):
        Rdot[i, 0] = R[i, 1] * wz - R[i, 2] * wy
        Rdot[i, 1] = -R[i, 0] * wz + R[i, 2] * wx
        Rdot[i, 2] = R[i, 0] * wy - R[i, 1] * wx
    return Rdot


@nb.njit(cache=True)
def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@nb.njit(cache=True)
def normalize3(v):
    n = norm3(v)
    if n < 1e-30:
        return np.array([1.0, 0.0, 0.0])
    return v / n


@nb.njit(cache=True)
def cross3(a, b):
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


@nb.njit(cache=True)
def reorthonormalize_rotation(R):
    # Keep R on SO(3) to avoid geometric drift in long integrations.
    e0 = normalize3(R[:, 0])
    c1 = R[:, 1]
    u1 = c1 - dot3(c1, e0) * e0
    if norm3(u1) < 1e-30:
        if np.abs(e0[0]) < 0.9:
            u1 = np.array([1.0, 0.0, 0.0]) - e0[0] * e0
        else:
            u1 = np.array([0.0, 1.0, 0.0]) - e0[1] * e0
    e1 = normalize3(u1)
    e2 = normalize3(cross3(e0, e1))
    e1 = normalize3(cross3(e2, e0))
    out = np.empty((3, 3))
    out[:, 0] = e0
    out[:, 1] = e1
    out[:, 2] = e2
    return out


@nb.njit(cache=True)
def random_rotation_matrix(u1, u2, u3):
    """Haar-uniform rotation from 3 independent U(0,1) samples."""
    two_pi = 2.0 * np.pi
    q1 = np.sqrt(1.0 - u1) * np.sin(two_pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(two_pi * u2)
    q3 = np.sqrt(u1) * np.sin(two_pi * u3)
    q4 = np.sqrt(u1) * np.cos(two_pi * u3)

    x, y, z, w = q1, q2, q3, q4
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty((3, 3))
    R[0, 0] = 1.0 - 2.0 * (yy + zz)
    R[0, 1] = 2.0 * (xy - wz)
    R[0, 2] = 2.0 * (xz + wy)
    R[1, 0] = 2.0 * (xy + wz)
    R[1, 1] = 1.0 - 2.0 * (xx + zz)
    R[1, 2] = 2.0 * (yz - wx)
    R[2, 0] = 2.0 * (xz - wy)
    R[2, 1] = 2.0 * (yz + wx)
    R[2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


# ─── Main collision kernel ────────────────────────────────────────────────


@nb.njit(cache=True)
def run_collision_numba(
    Etr_init,
    Erot1_initial,
    Erot2_initial,
    frac11,
    frac21,
    b,
    b_phi,
    r1_u1,
    r1_u2,
    r1_u3,
    r2_u1,
    r2_u2,
    r2_u3,
    sign_o11,
    sign_o12,
    sign_o21,
    sign_o22,
):
    """Run a single collision simulation, fully JIT-compiled."""

    vtr = np.sqrt(Etr_init / m_H2)

    Er11 = frac11 * Erot1_initial
    Er12 = (1.0 - frac11) * Erot1_initial
    Er21 = frac21 * Erot2_initial
    Er22 = (1.0 - frac21) * Erot2_initial

    omega_11 = sign_o11 * np.sqrt(2.0 * Er11 / I_mol)
    omega_12 = sign_o12 * np.sqrt(2.0 * Er12 / I_mol)
    omega_21 = sign_o21 * np.sqrt(2.0 * Er21 / I_mol)
    omega_22 = sign_o22 * np.sqrt(2.0 * Er22 / I_mol)

    omega_1 = np.array([omega_11, omega_12, 0.0])
    omega_2 = np.array([omega_21, omega_22, 0.0])

    by = b * np.cos(b_phi)
    bz = b * np.sin(b_phi)
    X1 = np.array([-2.0 * sigma_LJ, -0.5 * by, -0.5 * bz])
    X2 = np.array([2.0 * sigma_LJ, 0.5 * by, 0.5 * bz])

    half_d = 0.5 * d_H2

    R1 = random_rotation_matrix(r1_u1, r1_u2, r1_u3)
    R2 = random_rotation_matrix(r2_u1, r2_u2, r2_u3)

    # Atom positions in lab frame
    X11 = X1 + R1[:, 2] * half_d
    X12 = X1 - R1[:, 2] * half_d
    X21 = X2 + R2[:, 2] * half_d
    X22 = X2 - R2[:, 2] * half_d

    V1 = np.array([vtr, 0.0, 0.0])
    V2 = np.array([-vtr, 0.0, 0.0])

    for step in range(nsteps):
        dx = X1[0] - X2[0]
        dy = X1[1] - X2[1]
        dz = X1[2] - X2[2]
        dr = np.sqrt(dx * dx + dy * dy + dz * dz)

        if dr > 5.0 * sigma_LJ and step > 0:
            break

        # Inter-molecular atom-atom forces
        F13 = intraatomic_force_nb(X11, X21)
        F14 = intraatomic_force_nb(X11, X22)
        F23 = intraatomic_force_nb(X12, X21)
        F24 = intraatomic_force_nb(X12, X22)

        F1 = F13 + F14 + F23 + F24
        F2 = -F1

        M1, M2 = get_moments_nb(F13, F14, F23, F24, R1, R2, d_H2)

        # Velocity Verlet: half-step
        v1_half = V1 + (F1 / m1) * (0.5 * dt)
        v2_half = V2 + (F2 / m2) * (0.5 * dt)
        omega_1_half = omega_1 + (M1 / I_mol) * (0.5 * dt)
        omega_2_half = omega_2 + (M2 / I_mol) * (0.5 * dt)
        R1_half = R1 + get_rdot_nb(omega_1, R1) * (0.5 * dt)
        R2_half = R2 + get_rdot_nb(omega_2, R2) * (0.5 * dt)

        # Full-step positions and orientations
        X1 = X1 + v1_half * dt
        X2 = X2 + v2_half * dt
        R1 = R1 + get_rdot_nb(omega_1_half, R1_half) * dt
        R2 = R2 + get_rdot_nb(omega_2_half, R2_half) * dt
        R1 = reorthonormalize_rotation(R1)
        R2 = reorthonormalize_rotation(R2)

        # Update atom positions
        X11 = X1 + R1[:, 2] * half_d
        X12 = X1 - R1[:, 2] * half_d
        X21 = X2 + R2[:, 2] * half_d
        X22 = X2 - R2[:, 2] * half_d

        # Half-step forces
        F13h = intraatomic_force_nb(X11, X21)
        F14h = intraatomic_force_nb(X11, X22)
        F23h = intraatomic_force_nb(X12, X21)
        F24h = intraatomic_force_nb(X12, X22)
        F1h = F13h + F14h + F23h + F24h
        F2h = -F1h

        M1h, M2h = get_moments_nb(F13h, F14h, F23h, F24h, R1, R2, d_H2)

        # Full-step velocities
        V1 = v1_half + (F1h / m1) * (0.5 * dt)
        V2 = v2_half + (F2h / m2) * (0.5 * dt)
        omega_1 = omega_1_half + (M1h / I_mol) * (0.5 * dt)
        omega_2 = omega_2_half + (M2h / I_mol) * (0.5 * dt)

    # Final energies
    Ekin_final = np.linalg.norm(V1) ** 2 * 0.5 * m1 + np.linalg.norm(V2) ** 2 * 0.5 * m2
    Erot1_final = 0.5 * I_mol * (omega_1[0] ** 2 + omega_1[1] ** 2)
    Erot2_final = 0.5 * I_mol * (omega_2[0] ** 2 + omega_2[1] ** 2)

    return Ekin_final, Erot1_final, Erot2_final


# ─── Python wrapper for random init + multiprocessing ─────────────────────


def run_collision(i):
    rng = np.random.default_rng(i)

    T_trans = T_min + rng.random() * (T_max - T_min)
    T_rot   = T_min + rng.random() * (T_max - T_min)


    # Collision pair relative translational energy distribution.
    Etr_init = rng.gamma(shape=1.5, scale=T_trans*kB)

    # Canonical rotational mode energies for linear rotor (two quadratic modes).
    Er11 = rng.gamma(shape=0.5, scale=T_rot*kB)
    Er12 = rng.gamma(shape=0.5, scale=T_rot*kB)
    Er21 = rng.gamma(shape=0.5, scale=T_rot*kB)
    Er22 = rng.gamma(shape=0.5, scale=T_rot*kB)
    Erot1_initial = Er11 + Er12
    Erot2_initial = Er21 + Er22

    b_max = 1.5 * sigma_LJ
    b = b_max * np.sqrt(rng.random()) 
    b_phi = 2.0 * np.pi * rng.random()

    frac11 = Er11 / Erot1_initial 
    frac21 = Er21 / Erot2_initial 

    sign_o11 = 1.0 if rng.random() > 0.5 else -1.0
    sign_o12 = 1.0 if rng.random() > 0.5 else -1.0
    sign_o21 = 1.0 if rng.random() > 0.5 else -1.0
    sign_o22 = 1.0 if rng.random() > 0.5 else -1.0

    r1_u1 = rng.random()
    r1_u2 = rng.random()
    r1_u3 = rng.random()
    r2_u1 = rng.random()
    r2_u2 = rng.random()
    r2_u3 = rng.random()

    Etr_final, Erot1_final, Erot2_final = run_collision_numba(
        Etr_init,
        Erot1_initial,
        Erot2_initial,
        frac11,
        frac21,
        b,
        b_phi,
        r1_u1,
        r1_u2,
        r1_u3,
        r2_u1,
        r2_u2,
        r2_u3,
        sign_o11,
        sign_o12,
        sign_o21,
        sign_o22,
    )
    return [
        Etr_init,
        Erot1_initial,
        Erot2_initial,
        Etr_final,
        Erot1_final,
        Erot2_final,
    ]


if __name__ == "__main__":
    from tqdm import tqdm

    # Trigger JIT compilation before timing / multiprocessing
    print("Compiling Numba kernels...")
    _ = run_collision(0)
    print("Done.")

    print(f"You have {os.cpu_count()} CPU cores available.")
    num_processes = 6
    print(f"Using {num_processes} CPU cores for simulation.")

    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_results = list(
            tqdm(
                pool.imap_unordered(run_collision, range(ncoll)),
                total=ncoll,
                desc="Running collisions",
                unit="collisions",
            )
        )
    df = pd.DataFrame(all_results, columns=pd.Index(varNames))
    df.to_csv(outputfile, index=False)

    print(df.head(1))
    print(f"--- {time.time() - start_time:.1f} seconds ---")
