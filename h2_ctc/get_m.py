"""Torque / angular momentum vectors in body-fixed frames.

Python translation of getM.m.
"""

import numpy as np


def get_m(F13tr, F14tr, F23tr, F24tr, R1, R2, dH2):
    """Compute torque vectors for both molecules in their body-fixed frames.

    Forces on molecule 1 atoms are projected into molecule 1's body frame
    via R1 (and similarly for molecule 2 via R2). The z-component of the
    torque is zero because the molecule axis is aligned with the body-frame
    z-axis.

    Parameters
    ----------
    F13tr, F14tr, F23tr, F24tr : array_like, shape (3,)
        Pairwise forces in the inertial frame [N]:
          F13 = force on atom 1 (mol 1) from atom 3 (mol 2), etc.
    R1, R2 : np.ndarray, shape (3, 3)
        Rotation matrices of molecules 1 and 2.
    dH2 : float
        Bond length of H2 [m].

    Returns
    -------
    M1, M2 : np.ndarray, shape (3,)
        Torque vectors [N·m] for molecules 1 and 2 in their body frames.
    """
    # Rotate inertial-frame forces into each molecule's body frame.
    # MATLAB: F13_r = F13tr * R1  (row-vec * matrix = row-vec)
    # NumPy:  F13_r = F13tr @ R1  (1-D array @ 2-D matrix → same result)
    F13_r = F13tr @ R1
    F14_r = F14tr @ R1
    F23_r = F23tr @ R1
    F24_r = F24tr @ R1

    F31_r = -F13tr @ R2
    F41_r = -F14tr @ R2
    F32_r = -F23tr @ R2
    F42_r = -F24tr @ R2

    # Torque components (x = index 0, y = index 1, z = index 2)
    # Note: MATLAB uses 1-based indexing, so MATLAB (1) → Python [0], (2) → [1]
    M1 = np.zeros(3)
    M1[0] = -dH2/2 * (F13_r[1] + F14_r[1]) + dH2/2 * (F23_r[1] + F24_r[1])
    M1[1] =  dH2/2 * (F13_r[0] + F14_r[0]) - dH2/2 * (F23_r[0] + F24_r[0])
    M1[2] = 0.0

    M2 = np.zeros(3)
    M2[0] = -dH2/2 * (F31_r[1] + F32_r[1]) + dH2/2 * (F41_r[1] + F42_r[1])
    M2[1] =  dH2/2 * (F31_r[0] + F32_r[0]) - dH2/2 * (F41_r[0] + F42_r[0])
    M2[2] = 0.0

    return M1, M2
