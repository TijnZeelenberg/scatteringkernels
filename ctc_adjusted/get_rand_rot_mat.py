"""Random rotation matrix (ZYX Euler angles).

Python translation of getRandRotMat.m.
"""

import numpy as np


def get_rand_rot_mat():
    """Generate a uniformly random 3×3 rotation matrix using ZYX Euler angles.

    * psi  (z-rotation): uniform in [0, 2π)
    * theta (y-rotation): fixed at 0
    * phi  (x-rotation): distributed as acos(1 − 2·u), u ~ Uniform(0,1),
      which gives a uniform distribution on the sphere for the polar angle.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix R = Rz(psi) @ Ry(theta) @ Rx(phi).
    """
    psi   = np.random.rand() * 2.0 * np.pi   # azimuthal angle [rad]
    theta = 0.0                               # polar tilt [rad]
    phi   = np.arccos(1.0 - 2.0 * np.random.rand())  # elevation angle [rad]

    cp, sp = np.cos(psi),   np.sin(psi)
    ct, st = np.cos(theta), np.sin(theta)
    cf, sf = np.cos(phi),   np.sin(phi)

    Rz = np.array([[ cp, -sp, 0.0],
                   [ sp,  cp, 0.0],
                   [0.0, 0.0, 1.0]])

    Ry = np.array([[ ct, 0.0,  st],
                   [0.0, 1.0, 0.0],
                   [-st, 0.0,  ct]])

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cf, -sf],
                   [0.0,  sf,  cf]])

    return Rz @ Ry @ Rx
