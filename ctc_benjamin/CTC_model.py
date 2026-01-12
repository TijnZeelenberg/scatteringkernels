import os

# Make sure each process uses only one thread for numpy operations (since we are implementing multiprocessing ourselves)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import multiprocessing
import time
from math import sqrt
from random import random, seed  # for randomly initializing particles

import numpy as np
import pandas as pd  # for storing the data
from CTC_utils import *
from numpy.linalg import norm

start_time = time.time()

# Constants
m_H = 1.6738e-27  # Hydrogen atom mass [kg]
m_H2 = m_H * 2  # Hydrogen molecule mass [kg]
sigma_LJ = 3.06 * 1e-10  # LJ length-parameter [m]
kB = 1.38064852e-23  # Boltzmann constant [m^2*kg/s^2/K]
mu_H2 = m_H2 * m_H2 / (m_H2 + m_H2)  # Reduced mass H2 [kg]
d_H2 = 0.741 * 1e-10  # Interatomic distance for H2 molecule [m]
I = 0.5 * (d_H2**2) * m_H  # Hydrogen moment of inertia [kg m^2]

# Simulation settings
dt = 0.1e-15  # Time-step [s]
dt2 = dt * dt  # Time-step squared [s^2]
tsim = 5e-12  # Max. simulation time [s]
nsteps = tsim / dt  # Max. number of steps
ncoll = 2000  # Number of collisios
Etr_K_max = 5900  # Maximum translational velocity [K]
Etr_min = 100  # Minimum translational energy [K] (to avoid molecules barely moving)
Erot_K_max = 3000  # Maximum rotational energy [K]

# Molecule 1 with 2 atoms 1 and 2
weight_1 = 1
m11 = m_H * weight_1   # Atom mass 1 [kg]
m12 = m_H * weight_1   # Atom mass 2 [kg]
m1 = m11 + m12  # Molecular mass [kg]

# Molecule 2 with 2 atoms 1 and 2
weight_2 = 1
m21 = m_H * weight_2 # Atom mass 1 [kg]
m22 = m_H * weight_2 # Atom mass 2 [kg]
m2 = m21 + m22  # Molecular mass [kg]

# data storage
outputfile = f"CTC_simulation_results_m{weight_1}_m{weight_2}.csv"

varNames = [
    "b",
    "Etr_init_K",
    "Er1_init_K",
    "Er2_init_K",
    "Etr1_final_K",
    "Etr2_final_K",
    "Er1_final_K",
    "Er2_final_K",
    "V1_init",
    "V1_final",
    "V2_init",
    "V2_final",
]


def run_collision(i):
    if i % 50 == 0:
        print(f"Running collision {i}")
    seed(i)

    # Translational energy of each particle [K]
    Etr_init_K = Etr_min + random() * Etr_K_max
    vtr = sqrt(Etr_init_K * kB / m_H2)  # Velocity of each particle [m/s]

    bmax = 1.5 * sigma_LJ  # Max impact parameter
    b = random() * bmax  # Impact parameter

    # Rotational energies
    Erot1_initial = random() * Erot_K_max * kB  # Rotational energy of particle 1 [J]
    Erot2_initial = random() * Erot_K_max * kB  # Rotational energy of particle 2 [J]

    frac11 = random()  # Fraction of rotational energy in mode 1
    frac21 = random()  # Fraction of rotational energy in mode 2

    Er11 = frac11 * Erot1_initial  # Rotational energy for particle 1 mode 1
    Er12 = (1 - frac11) * Erot1_initial  # Rotational energy for particle 1 mode 2
    Er21 = frac21 * Erot2_initial  # Rotational energy for particle 2 mode 1
    Er22 = (1 - frac21) * Erot2_initial  # Rotational energy for particle 2 mode 1

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

    V1_init = np.array([vtr, 0, 0])  # Particle 1 moves in the positive x direction
    V2_init = np.array([-vtr, 0, 0])  # Particle 2 moves in the negative x direction

    Ekin1 = 0.5 * m1 * (norm(V1_init) ** 2)
    Ekin2 = 0.5 * m2 * (norm(V2_init) ** 2)
    Erot1 = 0.5 * I * (omega_1[0] ** 2 + omega_1[1] ** 2)
    Erot2 = 0.5 * I * (omega_2[0] ** 2 + omega_2[1] ** 2)

    V1 = V1_init.copy()
    V2 = V2_init.copy()
    dr = 0
    step = 0
    # Collision simulation
    while dr <= 5 * sigma_LJ:
        step += 1
        if step + 1 >= nsteps:
            print(f"Collision {i} took too long to drift apart. Continuing...")
            break
        dr = norm(X1 - X2)

        # Compute interaction forces between atoms of different molecules
        F13tr = intraatomic_force(X11, X21, sigma_LJ, kB)
        F14tr = intraatomic_force(X11, X22, sigma_LJ, kB)
        F23tr = intraatomic_force(X12, X21, sigma_LJ, kB)
        F24tr = intraatomic_force(X12, X22, sigma_LJ, kB)

        # Total force on each molecule
        F1 = F13tr + F14tr + F23tr + F24tr
        F2 = -F1

        # Compute momenta in the body-fixed frames using the forces in inertial frame
        # and the orientations of the molecules.
        # Momenta are computed at timestep t.
        M1, M2 = get_moments(F13tr, F14tr, F23tr, F24tr, R1, R2, d_H2)

        # Update velocities and angular velocities using Verlet algorithm
        # Compute half-step velocities
        v1_half = V1 + (F1 / m1) * (0.5 * dt)
        v2_half = V2 + (F2 / m2) * (0.5 * dt)
        omega_1_half = omega_1 + (M1 / I) * (0.5 * dt)
        omega_2_half = omega_2 + (M2 / I) * (0.5 * dt)
        R1_half = R1 + get_rdot(omega_1, R1) * (0.5 * dt)
        R2_half = R2 + get_rdot(omega_2, R2) * (0.5 * dt)

        # Update positions at timestep (t+dt)
        X1 = X1 + v1_half * dt
        X2 = X2 + v2_half * dt
        # Update orientations at timestep (t+dt)
        R1 = R1 + get_rdot(omega_1_half, R1_half) * dt
        R2 = R2 + get_rdot(omega_2_half, R2_half) * dt

        # Update atomic positions at timestep (t+dt) using Center of Mass and rotation matrices
        Xv11 = R1 @ np.transpose(X11_0)
        Xv12 = R1 @ np.transpose(X12_0)
        Xv21 = R2 @ np.transpose(X21_0)
        Xv22 = R2 @ np.transpose(X22_0)

        X11 = X1 + np.transpose(Xv11)
        X12 = X1 + np.transpose(Xv12)
        X21 = X2 + np.transpose(Xv21)
        X22 = X2 + np.transpose(Xv22)

        # Compute half-step forces
        F13trhalf = intraatomic_force(X11, X21, sigma_LJ, kB)
        F14trhalf = intraatomic_force(X11, X22, sigma_LJ, kB)
        F23trhalf = intraatomic_force(X12, X21, sigma_LJ, kB)
        F24trhalf = intraatomic_force(X12, X22, sigma_LJ, kB)
        F1_half = F13trhalf + F14trhalf + F23trhalf + F24trhalf
        F2_half = -F1_half

        M1_half, M2_half = get_moments(
            F13trhalf, F14trhalf, F23trhalf, F24trhalf, R1, R2, d_H2
        )

        # Update velocities and angular velocities at timestep (t+dt)
        V1 = v1_half + (F1_half / m1) * (0.5 * dt)
        V2 = v2_half + (F2_half / m2) * (0.5 * dt)
        omega_1 = omega_1_half + (M1_half / I) * (0.5 * dt)
        omega_2 = omega_2_half + (M2_half / I) * (0.5 * dt)

        # Extracting values at timestep t
        Ekin1 = 0.5 * m1 * (norm(V1) ** 2)
        Ekin2 = 0.5 * m2 * (norm(V2) ** 2)
        Erot1 = 0.5 * I * (omega_1[0] ** 2 + omega_1[1] ** 2)
        Erot2 = 0.5 * I * (omega_2[0] ** 2 + omega_2[1] ** 2)

    # Store the initial and final energies in a list
    collision_results = [
        b / sigma_LJ,  # b (normalized)
        Etr_init_K,  # Initial translational energy of each molecule (already in K)
        Erot1_initial / kB,  # Initial rotational energy of molecule 1
        Erot2_initial / kB,  # Initial rotational energy of molecule 2
        Ekin1 / kB,  # Final kinetic energy of molecule 1
        Ekin2 / kB,  # Final kinetic energy of molecule 2
        Erot1 / kB,  # Final rotational energy of molecule 1
        Erot2 / kB,  # Final rotational energy of molecule 2
        norm(V1_init),  # Initial speed of molecule 1 
        norm(V1),  # Final speed of molecule 1
        norm(V2_init),  # Initial speed of molecule 2
        norm(V2),  # Final speed of molecule 2
    ]
    return collision_results


if __name__ == "__main__":
    print(f"You have {os.cpu_count()} CPU cores available.")
    num_processes = 6
    print(f"Using {num_processes} CPU cores for simulation.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        all_results = pool.map(run_collision, range(ncoll))
    df = pd.DataFrame(all_results, columns=pd.Index(varNames))
    df.to_csv(outputfile, index=False)

    print(df)
    print("--- %s seconds ---", (time.time() - start_time))
