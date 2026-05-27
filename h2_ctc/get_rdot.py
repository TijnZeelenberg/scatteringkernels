"""Time derivative of a rotation matrix.

Python translation of getRdot.m.
"""

import numpy as np


def get_rdot(w, R):
    """Compute the time derivative of rotation matrix R given angular velocity w.

    The skew-symmetric matrix W̃ of ω is constructed and Ṙ = R · W̃.

    Parameters
    ----------
    w : array_like, shape (3,)
        Angular velocity vector [rad/s] in the body frame.
    R : np.ndarray, shape (3, 3)
        Current rotation matrix.

    Returns
    -------
    Rdot : np.ndarray, shape (3, 3)
        Time derivative of R.
    """
    w = np.asarray(w, dtype=float)

    wtilde = np.array([
        [ 0.0,  -w[2],  w[1]],
        [ w[2],  0.0,  -w[0]],
        [-w[1],  w[0],  0.0 ]
    ])

    return R @ wtilde
