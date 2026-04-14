"""Lennard-Jones (12-6) force and potential energy functions.

Python translation of LJ.m and LJ_e.m.
"""

import numpy as np

_K_TO_EV = 0.00008617328149741   # Kelvin to eV conversion
_EV_TO_J = 1.60217662e-19        # eV to Joule conversion
_SIGMA   = 3.06e-10              # LJ zero-crossing distance [m]
_EPS_J   = 34.00 * _K_TO_EV * _EV_TO_J  # Well-depth [J]


def LJ(r):
    """Compute the Lennard-Jones (12-6) interatomic force magnitude.

    Parameters
    ----------
    r : float
        Interatomic distance [m].

    Returns
    -------
    float
        Interatomic force [N]. Negative = attractive.
    """
    return -4.0 * _EPS_J * (6.0 * _SIGMA**6 / r**7 - 12.0 * _SIGMA**12 / r**13)


def LJ_e(r):
    """Compute the Lennard-Jones (12-6) interatomic potential energy.

    Parameters
    ----------
    r : float
        Interatomic distance [m].

    Returns
    -------
    float
        Potential energy [J].
    """
    return 4.0 * _EPS_J * (((_SIGMA / r)**12) - ((_SIGMA / r)**6))
