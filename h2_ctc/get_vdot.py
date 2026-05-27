"""Translational acceleration of a rigid body (Newton's 2nd law).

Python translation of getVdot.m.
"""

import numpy as np


def get_vdot(F, m):
    """Compute acceleration from net force and mass.

    Parameters
    ----------
    F : array_like, shape (3,)
        Net force vector [N].
    m : float
        Mass [kg].

    Returns
    -------
    np.ndarray, shape (3,)
        Acceleration vector [m/s²].
    """
    return np.asarray(F, dtype=float) / m
