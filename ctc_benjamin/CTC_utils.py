from math import pi, acos, cos, sin
import numpy as np
from numpy.typing import NDArray


def randomrotationmatrix(seed: float) -> np.ndarray:
    """Generate random 3D rotation matrix according to ZYX Euler angles

    Args:
        float seed: used to seed the function for reproducability

    Returns:
        3x3 np.array of Euler angles
    """
    psi = seed * 2 * pi
    theta = 0
    phi = acos(1 - 2 * seed)

    Rz = np.array([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
    Ry = np.array(
        [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    )
    Rx = np.array([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
    return Rz @ Ry @ Rx


def lennartjones_potential(dist: float, sigma_LJ: float, kB: float) -> float:
    """Calculate Lennard-Jones potential energy
    Args:
        float distance: distance between two atoms [m]
    Returns:
        float: Lennard-Jones potential energy [J]
    """
    K_to_EV = 8.617333262145e-5  # eV/K
    ev_to_K = 1.60217667e-19  # K/eV

    epsilon = 34 * K_to_EV * ev_to_K  # Depth of the potential well [J]

    potential = 4 * epsilon * ((sigma_LJ / dist) ** 12 - (sigma_LJ / dist) ** 6)

    return potential


def lennartjones_force(dist: float, sigma_LJ: float, kB: float) -> float:
    """Calculate the 12-6 Lennard-Jones force
    Args:
        float distance: distance between two atoms [m]
    Returns:
        float: Lennard-Jones force [N]
    """
    K_to_EV = 8.617333262145e-5  # eV/K
    ev_to_K = 1.60217667e-19  # K/eV
    epsilon = 34 * K_to_EV * ev_to_K  # Depth of the potential well [J]
    force = (
        -4
        * epsilon
        * ((6 * sigma_LJ**6) / (dist**7) - (12 * sigma_LJ**12) / (dist**13))
    )
    return force


def intraatomic_force(
    xi: NDArray, xj: NDArray, sigma_LJ: float, kB: float
) -> np.ndarray:
    """
    Computes the intra-atomic force between two atoms.

    Inputs:
      xi, xj : (3,) numpy arrays representing position vectors (X, Y, Z)

    Outputs:
      fij    : (3,) numpy array containing the interatomic forces
    """
    # Calculate interatomic distance
    drij = np.linalg.norm(xi - xj)

    # Geometrical computations
    # xi[0:2] takes the first two elements (X, Y)
    drijxy = np.linalg.norm(xi[0:2] - xj[0:2])

    # Indices: 0->X, 1->Y, 2->Z
    theta_ij = np.arctan2(xj[2] - xi[2], drijxy)
    phi_ij = np.arctan2(xj[1] - xi[1], xj[0] - xi[0])

    # Compute net force magnitude (Assumes LJ function is defined)
    f_mag = lennartjones_force(float(drij), sigma_LJ, kB)

    # Decompose the force
    f_z = np.sin(theta_ij) * f_mag
    f_xy = np.cos(theta_ij) * f_mag

    f_x = np.cos(phi_ij) * f_xy
    f_y = np.sin(phi_ij) * f_xy

    return np.array([-f_x, -f_y, -f_z])


def get_moments(f13_tr, f14_tr, f23_tr, f24_tr, r1, r2, d_h2):
    """
    Computes the total moment (torque) vectors for two molecules in the
    body-fixed coordinate system.

    Inputs:
        fij_tr : (3,) np.array, interatomic forces
        r1, r2 : (3, 3) np.array, rotation matrices
        d_h2   : float, bond length

    Outputs:
        m1, m2 : (3,) np.array, total moments (Nx, Ny, Nz) in N*m
    """
    # Transform forces to body-fixed frames (assuming row vector convention)
    f13_r = f13_tr @ r1
    f14_r = f14_tr @ r1
    f23_r = f23_tr @ r1
    f24_r = f24_tr @ r1

    # Newton's 3rd law for Molecule 2
    f31_r = -f13_tr @ r2
    f41_r = -f14_tr @ r2
    f32_r = -f23_tr @ r2
    f42_r = -f24_tr @ r2

    # --- Molecule 1 ---
    # Sum forces acting on Atom 1 and Atom 2 respectively
    f_on_atom1 = f13_r + f14_r
    f_on_atom2 = f23_r + f24_r

    # Calculate Moments (Torque = r x F).
    # Simplified for linear molecule aligned on Z-axis.
    # m_x = -z * F_y, m_y = z * F_x
    half_d = d_h2 / 2.0
    m1_x = half_d * (f_on_atom2[1] - f_on_atom1[1])
    m1_y = half_d * (f_on_atom1[0] - f_on_atom2[0])
    m1 = np.array([m1_x, m1_y, 0.0])

    # --- Molecule 2 ---
    f_on_atom3 = f31_r + f32_r
    f_on_atom4 = f41_r + f42_r

    m2_x = half_d * (f_on_atom4[1] - f_on_atom3[1])
    m2_y = half_d * (f_on_atom3[0] - f_on_atom4[0])
    m2 = np.array([m2_x, m2_y, 0.0])

    return m1, m2


def get_rdot(w, R):
    """
    Computes the derivative of the rotation matrix R given angular velocity w.

    Inputs:
        w : (3,) np.array, Angular velocity vector (Body Frame)
        R : (3,3) np.array, Rotation matrix

    Outputs:
        rdot : (3,3) np.array, Time derivative of R
    """
    # Unpack for readability
    wx, wy, wz = w

    # Construct skew-symmetric matrix (cross-product matrix)
    w_tilde = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])

    # Compute derivative
    # Note: Using @ for matrix multiplication
    return R @ w_tilde


def get_wdot(M, I):
    """
    Computes the derivative of angular velocity w given moment M and moment of inertia I.

    Inputs:
        M : (3,) np.array, Moment vector (Body Frame)
        I : float, Moment of inertia (assumed scalar for symmetric top)
    Outputs:
        wdot : (3,) np.array, Time derivative of angular velocity
    """

    # Unpack moments
    M1 = M[0]
    M2 = M[1]

    # Compute angular acceleration according to Newton's 2nd law
    omega_1_dot = M1 / I
    omega_2_dot = M2 / I
    omega_3_dot = 0.0  # No torque about symmetry axis
    return np.array([omega_1_dot, omega_2_dot, omega_3_dot])
