from math import pi, acos, cos, sin
import numpy as np


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
