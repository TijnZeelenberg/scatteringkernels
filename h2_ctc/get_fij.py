"""Interatomic force vector between two atoms.

Python translation of getFij.m.
"""

import numpy as np
from lj import LJ


def get_fij(Xi, Xj):
    """Compute the interatomic force vector on atom i due to atom j.

    Uses the Lennard-Jones (12-6) potential. The force is decomposed
    into (x, y, z) components via spherical angles.

    Parameters
    ----------
    Xi : array_like, shape (3,)
        Position of atom i [m].
    Xj : array_like, shape (3,)
        Position of atom j [m].

    Returns
    -------
    np.ndarray, shape (3,)
        Force vector [N] acting on atom i from atom j.
    """
    Xi = np.asarray(Xi, dtype=float)
    Xj = np.asarray(Xj, dtype=float)

    drij  = np.linalg.norm(Xi - Xj)
    drijxy = np.linalg.norm(Xi[:2] - Xj[:2])

    thetaij = np.arctan2(Xj[2] - Xi[2], drijxy)   # polar angle from XY-plane
    phiij   = np.arctan2(Xj[1] - Xi[1], Xj[0] - Xi[0])  # azimuthal angle in XY-plane

    Fmag = LJ(drij)

    Fijz  = np.sin(thetaij) * Fmag
    Fijxy = np.cos(thetaij) * Fmag
    Fijx  = np.cos(phiij) * Fijxy
    Fijy  = np.sin(phiij) * Fijxy

    return np.array([-Fijx, -Fijy, -Fijz])
