import numpy as np
from numpy.linalg import norm
import pandas as pd
import time
from random import random, seed
from math import sqrt
from CTC_utils import *

start_time = time.time()
seed(42)  # randomseed for reproducability


# Constants
m_H = 1.6738e-27  # Hydrogen atom mass [kg]
m_H2 = m_H * 2  # Hydrogen molecule mass [kg]
sigma_LJ = 3.06 * 1e-10  # LJ length-parameter [m]
kB = 1.38064852e-23  # Boltzmann constant [m^2*kg/s^2/K]
mu_H2 = m_H2 * m_H2 / (m_H2 + m_H2)  # Reduced mass H2 [kg]
d_H2 = 0.741 * 1e-100  # Interatomic distance for H2 molecule [m]
I = 0.5 * (d_H2**2) * m_H  # Hydrogen moment of inertia [kg m^2]

# Simulation settings
dt = 0.1e-15  # Time-step [s]
dt2 = dt * dt  # Time-step squared [s^2]
tsim = 2e-12  # Max. simulation time [s]
nsteps = tsim / dt  # Max. number of steps
ncoll = 1000  # Number of collisios
Etr_K_max = 5900  # Maximum translational velocity [K]
Etr_min = 100  # Minimum translational energy [K]
Erot_K_max = 3000  # Maximum rotational energy [K]

# Molecule 1 with 2 atoms 1 and 2
m11 = m_H  # Atom mass 1 [kg]
m12 = m_H  # Atom mass 2 [kg]
m1 = m11 + m12  # Molecular mass [kg]

# Molecule 2 with 2 atoms 1 and 2
m21 = m_H  # Atom mass 1 [kg]
m22 = m_H  # Atom mass 2 [kg]
m2 = m21 + m22  # Molecular mass [kg]

# data storage
varNames = ["b", "Etr", "Er1", "Er2", "Etrp", "Er1p", "Er2p"]
b_arr = np.zeros((ncoll, len(varNames)))
dataNames = np.array(
    [
        "Ekin1",
        "Ekin2",
        "Erot1",
        "Erot2",
        "Elj13",
        "Elj14",
        "Elj23",
        "Elj24",
        "dr12v",
        "dr13v",
        "dr14v",
        "dr23v",
        "dr24v",
        "dr34v",
        "drABv",
    ]
)

for i in range(ncoll):
    if i % 100 == 0:
        print(f"iteration {i} of {ncoll}")

    # Translational energy
    Etr_K = Etr_min + random() * Etr_K_max  # Translational energy [K]
    vtr = sqrt(Etr_K * kB / m_H2)  # Velocity [m/s]

    bmax = 1.5 * sigma_LJ  # Max impact parameter
    b = random() * bmax  # Impact parameter
    b_arr[i] = b / sigma_LJ

    # Rotational energies
    Erot_tot_1 = random() * Erot_K_max * kB  # Rotational energy of particle 1 [J]
    Erot_tot_2 = random() * Erot_K_max * kB  # Rotational energy of particle 2 [J]

    frac11 = random()  # Fraction of rotational energy in mode 1
    frac21 = random()  # Fraction of rotational energy in mode 2

    Er11 = frac11 * Erot_tot_1  # Rotational energy for particle 1 mode 1
    Er12 = (1 - frac11) * Erot_tot_1  # Rotational energy for particle 1 mode 2
    Er21 = frac21 * Erot_tot_2  # Rotational energy for particle 2 mode 1
    Er22 = (1 - frac21) * Erot_tot_2  # Rotational energy for particle 2 mode 1

    # Angular velocities omega_nm of particle n in mode m
    omega_11 = ((random() > 0.5) * 2 - 1) * sqrt(2 * Er11 / I)
    omega_12 = ((random() > 0.5) * 2 - 1) * sqrt(2 * Er12 / I)
    omega_21 = ((random() > 0.5) * 2 - 1) * sqrt(2 * Er21 / I)
    omega_22 = ((random() > 0.5) * 2 - 1) * sqrt(2 * Er22 / I)

    # Angular velocity vectors (molecule rotates around x and y axis)
    omega_1 = np.array([omega_11, omega_12, 0])
    omega_2 = np.array([omega_21, omega_22, 0])

    # Initialize Center of Mass of each molecule relative to global frame
    X1 = np.array([sigma_LJ * -2, 0, -b / 2])  # Position molecule 1 Center of Mass[m]
    X2 = np.array([sigma_LJ * 2, 0, b / 2])  # Position molecule 2 Center of Mass[m]

    # Initialize atom positions relative to Center of Mass of each particle
    X11_0 = np.array([0, 0, 0.5 * d_H2])  # Atom position 1  [m]
    X12_0 = np.array([0, 0, -0.5 * d_H2])  # Atom position 2 [m]
    X21_0 = np.array([0, 0, 0.5 * d_H2])  # Atom position 3 [m]
    X22_0 = np.array([0, 0, -0.5 * d_H2])  # Atom position 4 [m]

    # Initialize random orientation of each molecule
    R1 = randomrotationmatrix(random())
    R2 = randomrotationmatrix(random())

    # Compute atom positions relative to Center of Mass
    Xv11 = R1 @ np.transpose(X11_0)
    Xv12 = R1 @ np.transpose(X12_0)
    Xv21 = R2 @ np.transpose(X21_0)
    Xv22 = R2 @ np.transpose(X22_0)

    # Compute atom positions relative to global frame
    X11 = X1 + np.transpose(Xv11)
    X12 = X1 + np.transpose(Xv12)
    X21 = X2 + np.transpose(Xv21)
    X22 = X2 + np.transpose(Xv22)

    V1 = np.array([vtr, 0, 0])  # Particle 1 moves in the positive x direction
    V2 = np.array([-vtr, 0, 0])  # Particle 2 moves in the negative x direction

    # Preallocation data arrays
    Ekin1 = np.zeros((1, round(nsteps)))
    Ekin2 = np.zeros((1, round(nsteps)))
    Erot1 = np.zeros((1, round(nsteps)))
    Erot2 = np.zeros((1, round(nsteps)))
    Elj13 = np.zeros((1, round(nsteps)))
    Elj14 = np.zeros((1, round(nsteps)))
    Elj23 = np.zeros((1, round(nsteps)))
    Elj24 = np.zeros((1, round(nsteps)))
    dr12v = np.zeros((1, round(nsteps)))
    dr13v = np.zeros((1, round(nsteps)))
    dr14v = np.zeros((1, round(nsteps)))
    dr23v = np.zeros((1, round(nsteps)))
    dr24v = np.zeros((1, round(nsteps)))
    dr34v = np.zeros((1, round(nsteps)))
    drABv = np.zeros((1, round(nsteps)))

    dr = 0
    step = 0

    # Collision simulation
    while dr <= 5 * sigma_LJ:
        step += 1
        dr = norm(X1 - X2)
        drABv[step] = dr

        # Extracting values at timestep t
        Ekin1[step] = 0.5 * m1 * (norm(V1) ** 2)
        Ekin2[step] = 0.5 * m2 * (norm(V2) ** 2)
        Erot1[step] = 0.5 * I * (omega_1[0] ** 2 + omega_1[1] ** 2)
        Erot1[step] = 0.5 * I * (omega_2[0] ** 2 + omega_2[1] ** 2)


print("--- %s seconds ---", (time.time() - start_time))
