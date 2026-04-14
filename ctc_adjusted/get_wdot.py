"""Angular acceleration of a linear diatomic molecule.

Python translation of getWdot.m.
"""

import numpy as np


def get_wdot(M, I):
    """Compute angular acceleration from torque and moment of inertia.

    Only the x and y components of torque are non-zero (z-axis is the
    molecular axis in the body frame). The z-component of angular
    acceleration is therefore always zero.

    Parameters
    ----------
    M : array_like, shape (3,)
        Torque vector [N·m] in the body frame.
    M[0] = Mx, M[1] = My, M[2] = 0.
    I : float
        Principal moment of inertia [kg·m²].

    Returns
    -------
    np.ndarray, shape (3,)
        Angular acceleration vector [rad/s²].
    """
    M = np.asarray(M, dtype=float)
    return np.array([M[0] / I, M[1] / I, 0.0])
