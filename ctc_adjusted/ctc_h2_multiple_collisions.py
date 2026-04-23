import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from lj import LJ_e
from get_fij import get_fij
from get_m import get_m
from get_rand_rot_mat import get_rand_rot_mat
from get_rdot import get_rdot
from get_vdot import get_vdot
from get_wdot import get_wdot
from dscatter import dscatter


# ---------------------------------------------------------------------------
# Number of collisions to simulate
# ---------------------------------------------------------------------------
ncoll = 20000
savefile = "data/H2H2_collisions.npy"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
m_H = 1.6738e-27  # Hydrogen atom mass [kg]
m_H2 = m_H * 2  # Hydrogen molecule mass [kg]
sigma_LJ = 3.06e-10  # LJ length-parameter [m]
kB = 1.38064852e-23  # Boltzmann constant [J/K]
d_H2 = 0.741e-10  # H2 bond length [m]
I = 0.5 * d_H2**2 * m_H  # Moment of inertia [kg·m²]

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
dt = 0.1e-15  # Time-step [s]
tsim = 2e-12  # Max simulation time [s]
n_steps = int(tsim / dt)  # Max number of steps


# ---------------------------------------------------------------------------
# Single-collision worker (runs in a subprocess)
# ---------------------------------------------------------------------------
def run_collision(seed: int):
    """Simulate one H2-H2 collision and return scalar result tuple."""
    np.random.seed(seed)

    # Sample total translational energy uniformly between 50 K and 6000 K
    Etr_K_max = 5950.0
    frac_tr = np.random.rand()
    Etr_tot = (frac_tr * Etr_K_max)
    # Split total translational energy randomly between the two molecules
    frac_tr1 = np.random.rand()
    Etr_K1 =  frac_tr1 * Etr_tot + 50.0  # Ensure minimum energy of 50 K for molecule 1
    Etr_J1 = Etr_K1 * kB
    vtr1 = np.sqrt(2.0 * Etr_J1 / m_H2)
    Etr_K2 = (1.0 - frac_tr1) * Etr_tot + 50.0  # Ensure minimum energy of 50 K for molecule 2
    Etr_J2 = Etr_K2 * kB
    vtr2 = np.sqrt(2.0 * Etr_J2 / m_H2)

    bmax = 1.2 * sigma_LJ
    b = np.random.rand() * bmax
    b_norm = b / sigma_LJ

    Erot_K_max = 3000.0
    Erot_tot_1 = np.random.rand() * Erot_K_max * kB
    Erot_tot_2 = np.random.rand() * Erot_K_max * kB

    frac11 = np.random.rand()
    frac21 = np.random.rand()

    Er11 = frac11 * Erot_tot_1
    Er12 = (1.0 - frac11) * Erot_tot_1
    Er21 = frac21 * Erot_tot_2
    Er22 = (1.0 - frac21) * Erot_tot_2

    def sign():
        return (np.random.rand() > 0.5) * 2 - 1

    w11 = sign() * np.sqrt(2.0 * Er11 / I)
    w12 = sign() * np.sqrt(2.0 * Er12 / I)
    w21 = sign() * np.sqrt(2.0 * Er21 / I)
    w22 = sign() * np.sqrt(2.0 * Er22 / I)

    w1 = np.array([w11, w12, 0.0])
    w2 = np.array([w21, w22, 0.0])

    m1 = 2.0 * m_H
    m2 = 2.0 * m_H

    X1 = np.array([-2.0 * sigma_LJ, 0.0, -b / 2.0])
    X2 = np.array([2.0 * sigma_LJ, 0.0, b / 2.0])

    X11_0 = np.array([0.0, 0.0, 0.5 * d_H2])
    X12_0 = np.array([0.0, 0.0, -0.5 * d_H2])
    X21_0 = np.array([0.0, 0.0, 0.5 * d_H2])
    X22_0 = np.array([0.0, 0.0, -0.5 * d_H2])

    R1 = get_rand_rot_mat()
    R2 = get_rand_rot_mat()

    X11 = X1 + R1 @ X11_0
    X12 = X1 + R1 @ X12_0
    X21 = X2 + R2 @ X21_0
    X22 = X2 + R2 @ X22_0

    V1 = np.array([vtr1, 0.0, 0.0])
    V2 = np.array([-vtr2, 0.0, 0.0])

    # Only the first and last energy values are needed
    Ekin1_0 = Ekin2_0 = Erot1_0 = Erot2_0 = Etrans_rel_0= 0.0
    Ekin1_last = Ekin2_last = Erot1_last = Erot2_last = Etrans_rel_last = 0.0

    dr = 0.0
    step = 0

    while dr <= 5.0 * sigma_LJ:
        dr = np.linalg.norm(X1 - X2)

        ek1 = 0.5 * m1 * np.dot(V1, V1)
        ek2 = 0.5 * m2 * np.dot(V2, V2)
        g = V1 - V2
        ek_rel = 0.5 * (m1 * m2 / (m1 + m2)) * np.dot(g, g)
        er1 = 0.5 * I * (w1[0] ** 2 + w1[1] ** 2)
        er2 = 0.5 * I * (w2[0] ** 2 + w2[1] ** 2)

        if step == 0:
            Ekin1_0, Ekin2_0, Erot1_0, Erot2_0, Etrans_rel_0 = ek1, ek2, er1, er2, ek_rel

        Ekin1_last, Ekin2_last, Erot1_last, Erot2_last, Etrans_rel_last = ek1, ek2, er1, er2, ek_rel

        F13tr = get_fij(X11, X21)
        F14tr = get_fij(X11, X22)
        F23tr = get_fij(X12, X21)
        F24tr = get_fij(X12, X22)

        F1 = F13tr + F14tr + F23tr + F24tr
        F2 = -F1

        M1, M2 = get_m(F13tr, F14tr, F23tr, F24tr, R1, R2, d_H2)

        V1_ = V1 + 0.5 * dt * get_vdot(F1, m1)
        V2_ = V2 + 0.5 * dt * get_vdot(F2, m2)

        X1 = X1 + dt * V1_
        X2 = X2 + dt * V2_

        R1_ = R1 + 0.5 * dt * get_rdot(w1, R1)
        R2_ = R2 + 0.5 * dt * get_rdot(w2, R2)

        w1_ = w1 + 0.5 * dt * get_wdot(M1, I)
        w2_ = w2 + 0.5 * dt * get_wdot(M2, I)

        R1 = R1 + dt * get_rdot(w1_, R1_)
        R2 = R2 + dt * get_rdot(w2_, R2_)

        X11 = X1 + R1 @ X11_0
        X12 = X1 + R1 @ X12_0
        X21 = X2 + R2 @ X21_0
        X22 = X2 + R2 @ X22_0

        F13tr_ = get_fij(X11, X21)
        F14tr_ = get_fij(X11, X22)
        F23tr_ = get_fij(X12, X21)
        F24tr_ = get_fij(X12, X22)

        F1_ = F13tr_ + F14tr_ + F23tr_ + F24tr_
        F2_ = -F1_

        M1_, M2_ = get_m(F13tr_, F14tr_, F23tr_, F24tr_, R1, R2, d_H2)

        V1 = V1_ + 0.5 * dt * get_vdot(F1_, m1)
        V2 = V2_ + 0.5 * dt * get_vdot(F2_, m2)

        w1 = w1_ + 0.5 * dt * get_wdot(M1_, I)
        w2 = w2_ + 0.5 * dt * get_wdot(M2_, I)

        step += 1
        if step >= n_steps:
            break

    return (
        (Ekin1_0 + Ekin2_0) / kB,  # Etr before
        Ekin1_0 / kB,  # EtrA before
        Ekin2_0 / kB,  # EtrB before
        (Ekin1_last + Ekin2_last) / kB,  # Etr after
        Ekin1_last / kB,  # EtrA after
        Ekin2_last / kB,  # EtrB after
        Erot1_0 / kB,  # Er1 before
        Erot2_0 / kB,  # Er2 before
        Erot1_last / kB,  # Er1 after
        Erot2_last / kB,  # Er2 after
        Etrans_rel_0 / kB,  # Etrans_rel before
        Etrans_rel_last / kB,  # Etrans_rel after
        b_norm,  # b / sigma_LJ
    )


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Main collision loop — parallelised over available CPU cores
    # -----------------------------------------------------------------------
    t_start = time.time()

    n_workers = multiprocessing.cpu_count()
    print(f"Running {ncoll} collisions on {n_workers} workers …")

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = list(
            tqdm(pool.imap(run_collision, range(ncoll)), total=ncoll, desc="Collisions")
        )

    elapsed = time.time() - t_start
    print(f"\nDone. Elapsed: {elapsed:.1f} s")

    # Unpack results
    (
        Etrrvec,
        EtrAvec,
        EtrBvec,
        Etrrpvec,
        EtrApvec,
        EtrBpvec,
        Er1vec,
        Er2vec,
        Er1pvec,
        Er2pvec,
        Etrans_rel_vec,
        Etrans_relpvec,
        bvec,
    ) = map(np.array, zip(*results))

    # -----------------------------------------------------------------------
    # Assemble results table
    # -----------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "Etr": Etrans_rel_vec,
            "Er1": Er1vec,
            "Er2": Er2vec,
            "Etrp": Etrans_relpvec,
            "Er1p": Er1pvec,
            "Er2p": Er2pvec,
        }
    )

    # save as numpy npy
    np.save(savefile, df.to_numpy())
    print(f"Saved {savefile}")

    # -----------------------------------------------------------------------
    # Inelastic collision fraction
    # -----------------------------------------------------------------------
    E_xchanged = df["Etr"] / df["Etrp"]
    inelastic_count = int(((E_xchanged > 0.99) & (E_xchanged < 1.01)).sum())
    inelastic_frac = inelastic_count / len(E_xchanged)
    print(f"Inelastic fraction (|ΔEtr/Etr| < 1%): {inelastic_frac:.3f}")

    # -----------------------------------------------------------------------
    # Visualise pre- and post-collisional energies
    # ----------------------------------------------------------------------

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].scatter(df["Etr"], df["Etrp"])
