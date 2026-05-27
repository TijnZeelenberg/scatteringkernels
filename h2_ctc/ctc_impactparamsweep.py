import time

import numpy as np
import paths
import pandas as pd
from numba import njit, prange
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Physical constants  (module-level so numba can see them as compile-time
# constants inside @njit functions)
# ---------------------------------------------------------------------------
_m_H = 1.6738e-27  # H atom mass [kg]
_m_H2 = _m_H * 2  # H2 mass [kg]
_sigma = 3.06e-10  # LJ σ [m]
_kB = 1.38064852e-23  # Boltzmann [J/K]
_d_H2 = 0.741e-10  # bond length [m]
_I = 0.5 * (_d_H2**2) * _m_H  # moment of inertia [kg·m²]
_eps_J = 34.00 * 0.00008617328149741 * 1.60217662e-19  # LJ ε [J]
_dt = 0.1e-15  # timestep [s]
_nstep = int(2e-12 / _dt)  # max integration steps

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
ncoll = 1000000  # number of collisions to simulate
seed = 42
dist = "uniform"  # 'uniform' | 'mb' | 'ntc'
T_eq = 6000.0  # equilibrium temperature [K] — used by mb and ntc distributions
E_rel_max = 2000.0  # upper cap [K] — used by uniform distribution
Erotmax = E_rel_max / 2  # max initial rotational energy [K]
bfac_sweep = [
    1.0,
    1.05
    1.1,
    1.15,
    1.2,
    1.25,
    1.3,
    1.35,
    1.4,
    1.45,
    1.5,
    1.55,
    1.6,
    1.65,
    1.7,
    1.75,
    1.8,
]  # impact-parameter sweep values

_DIST_IDS = {"uniform": 0, "mb": 1, "ntc": 2}

# ---------------------------------------------------------------------------
# JIT-compiled building blocks
# ---------------------------------------------------------------------------


@njit(cache=True)
def _lj_force(r):
    """LJ (12-6) force magnitude [N]."""
    return -4.0 * _eps_J * (6.0 * _sigma**6 / r**7 - 12.0 * _sigma**12 / r**13)


@njit(cache=True)
def _fij(Xi, Xj):
    """Force vector on atom i due to atom j [N]."""
    dx = Xi[0] - Xj[0]
    dy = Xi[1] - Xj[1]
    dz = Xi[2] - Xj[2]
    drij = np.sqrt(dx * dx + dy * dy + dz * dz)
    drijxy = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(Xj[2] - Xi[2], drijxy)
    phi = np.arctan2(Xj[1] - Xi[1], Xj[0] - Xi[0])
    Fmag = _lj_force(drij)
    Fijxy = np.cos(theta) * Fmag
    return np.array([-np.cos(phi) * Fijxy, -np.sin(phi) * Fijxy, -np.sin(theta) * Fmag])


@njit(cache=True)
def _torques(F13, F14, F23, F24, R1, R2, dH2):
    """Torques on both molecules in their body frames [N·m]."""
    F13_r = F13 @ R1
    F14_r = F14 @ R1
    F23_r = F23 @ R1
    F24_r = F24 @ R1
    F31_r = -F13 @ R2
    F41_r = -F14 @ R2
    F32_r = -F23 @ R2
    F42_r = -F24 @ R2
    h = dH2 / 2.0
    M1 = np.array(
        [
            -h * (F13_r[1] + F14_r[1]) + h * (F23_r[1] + F24_r[1]),
            h * (F13_r[0] + F14_r[0]) - h * (F23_r[0] + F24_r[0]),
            0.0,
        ]
    )
    M2 = np.array(
        [
            -h * (F31_r[1] + F32_r[1]) + h * (F41_r[1] + F42_r[1]),
            h * (F31_r[0] + F32_r[0]) - h * (F41_r[0] + F42_r[0]),
            0.0,
        ]
    )
    return M1, M2


@njit(cache=True)
def _rdot(w, R):
    """Time derivative of rotation matrix R given body-frame ω."""
    wt = np.zeros((3, 3))
    wt[0, 1] = -w[2]
    wt[0, 2] = w[1]
    wt[1, 0] = w[2]
    wt[1, 2] = -w[0]
    wt[2, 0] = -w[1]
    wt[2, 1] = w[0]
    return R @ wt


@njit(cache=True)
def _rand_rot_mat():
    """Uniformly random rotation matrix (ZYX Euler, θ=0)."""
    psi = np.random.random() * 2.0 * np.pi
    phi = np.arccos(1.0 - 2.0 * np.random.random())
    cp, sp = np.cos(psi), np.sin(psi)
    cf, sf = np.cos(phi), np.sin(phi)
    Rz = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cf, -sf], [0.0, sf, cf]])
    return Rz @ Rx  # Ry = I when theta = 0


# ---------------------------------------------------------------------------
# Single-collision kernel
# ---------------------------------------------------------------------------


@njit(cache=True)
def _run_one(seed, dist_id, T_eq, E_rel_max, bmax):
    """
    Simulate one H2-H2 collision and return a 13-element result array.

    Layout (energies in K, i.e. divided by kB):
      [0]  total KE before        [1]  mol-1 KE before    [2]  mol-2 KE before
      [3]  total KE after         [4]  mol-1 KE after     [5]  mol-2 KE after
      [6]  E_rot1 before          [7]  E_rot2 before
      [8]  E_rot1 after           [9]  E_rot2 after
      [10] E_rel (trans) before   [11] E_rel (trans) after
      [12] b / σ

    dist_id: 0=uniform over [0, E_rel_max], 1=Maxwell-Boltzmann Gamma(3/2, T_eq),
             2=NTC-matched Gamma(2, T_eq)
    v_COM is set to zero: molecules approach with equal and opposite speeds.
    """
    np.random.seed(seed)

    # ---- initial relative translational energy (E_rel, COM frame) ----------
    if dist_id == 0:  # uniform
        E_rel = np.random.random() * E_rel_max
    elif dist_id == 1:  # Maxwell-Boltzmann: p(E) ∝ √E · exp(-E/T) = Gamma(3/2, T)
        E_rel = np.random.gamma(1.5, T_eq)
    else:  # NTC-matched: p(E) ∝ E · exp(-E/T) = Gamma(2, T)
        E_rel = np.random.gamma(2.0, T_eq)
    if E_rel < 1.0:
        E_rel = 1.0
    # v_COM = 0 → equal and opposite speeds; E_rel = m_H2 * v²  (μ = m/2, v_rel = 2v)
    v = np.sqrt(E_rel * _kB / _m_H2)

    b = np.sqrt(np.random.random()) * bmax
    b_norm = b / _sigma

    # ---- initial rotational energies ---------------------------------------
    Erot1 = np.random.random() * Erotmax * _kB
    Erot2 = np.random.random() * Erotmax * _kB
    f11 = np.random.random()
    f21 = np.random.random()

    s1 = 1.0 if np.random.random() > 0.5 else -1.0
    s2 = 1.0 if np.random.random() > 0.5 else -1.0
    s3 = 1.0 if np.random.random() > 0.5 else -1.0
    s4 = 1.0 if np.random.random() > 0.5 else -1.0

    w1 = np.array(
        [
            s1 * np.sqrt(2.0 * f11 * Erot1 / _I),
            s2 * np.sqrt(2.0 * (1.0 - f11) * Erot1 / _I),
            0.0,
        ]
    )
    w2 = np.array(
        [
            s3 * np.sqrt(2.0 * f21 * Erot2 / _I),
            s4 * np.sqrt(2.0 * (1.0 - f21) * Erot2 / _I),
            0.0,
        ]
    )

    # ---- molecule COM positions and velocities -----------------------------
    X1 = np.array([-2.0 * _sigma, 0.0, -b / 2.0])
    X2 = np.array([2.0 * _sigma, 0.0, b / 2.0])
    V1 = np.array([v, 0.0, 0.0])
    V2 = np.array([-v, 0.0, 0.0])

    hb = 0.5 * _d_H2
    X11_0 = np.array([0.0, 0.0, hb])
    X12_0 = np.array([0.0, 0.0, -hb])
    X21_0 = np.array([0.0, 0.0, hb])
    X22_0 = np.array([0.0, 0.0, -hb])

    R1 = _rand_rot_mat()
    R2 = _rand_rot_mat()
    X11 = X1 + R1 @ X11_0
    X12 = X1 + R1 @ X12_0
    X21 = X2 + R2 @ X21_0
    X22 = X2 + R2 @ X22_0

    # ---- bookkeeping -------------------------------------------------------
    ek1_0 = ek2_0 = er1_0 = er2_0 = erel_0 = 0.0
    ek1_f = ek2_f = er1_f = er2_f = erel_f = 0.0

    dr = 0.0
    step = 0

    # ---- velocity-Verlet integration loop ----------------------------------
    # Structure mirrors the original: dr is computed at the top of the loop
    # body; the loop exits once dr > 5σ (condition checked against the
    # previous iteration's dr, same as the original `while dr <= 5σ` guard).
    while dr <= 5.0 * _sigma:
        dx = X1[0] - X2[0]
        dy = X1[1] - X2[1]
        dz = X1[2] - X2[2]
        dr = np.sqrt(dx * dx + dy * dy + dz * dz)

        ek1 = 0.5 * _m_H2 * (V1[0] ** 2 + V1[1] ** 2 + V1[2] ** 2)
        ek2 = 0.5 * _m_H2 * (V2[0] ** 2 + V2[1] ** 2 + V2[2] ** 2)
        gx = V1[0] - V2[0]
        gy = V1[1] - V2[1]
        gz = V1[2] - V2[2]
        erel = 0.5 * (_m_H2 * 0.5) * (gx * gx + gy * gy + gz * gz)  # μ = m/2
        er1 = 0.5 * _I * (w1[0] ** 2 + w1[1] ** 2)
        er2 = 0.5 * _I * (w2[0] ** 2 + w2[1] ** 2)

        if step == 0:
            ek1_0 = ek1
            ek2_0 = ek2
            er1_0 = er1
            er2_0 = er2
            erel_0 = erel
        ek1_f = ek1
        ek2_f = ek2
        er1_f = er1
        er2_f = er2
        erel_f = erel

        # --- forces at current positions ---
        F13 = _fij(X11, X21)
        F14 = _fij(X11, X22)
        F23 = _fij(X12, X21)
        F24 = _fij(X12, X22)
        F1 = F13 + F14 + F23 + F24
        M1, M2 = _torques(F13, F14, F23, F24, R1, R2, _d_H2)

        # --- half-kick ---
        idt = 0.5 * _dt
        V1_ = V1 + (idt / _m_H2) * F1
        V2_ = V2 - (idt / _m_H2) * F1  # F2 = -F1
        R1_ = R1 + idt * _rdot(w1, R1)
        R2_ = R2 + idt * _rdot(w2, R2)
        w1_ = np.array([w1[0] + idt * M1[0] / _I, w1[1] + idt * M1[1] / _I, 0.0])
        w2_ = np.array([w2[0] + idt * M2[0] / _I, w2[1] + idt * M2[1] / _I, 0.0])

        # --- full-step positions / orientations ---
        X1 = X1 + _dt * V1_
        X2 = X2 + _dt * V2_
        R1 = R1 + _dt * _rdot(w1_, R1_)
        R2 = R2 + _dt * _rdot(w2_, R2_)

        X11 = X1 + R1 @ X11_0
        X12 = X1 + R1 @ X12_0
        X21 = X2 + R2 @ X21_0
        X22 = X2 + R2 @ X22_0

        # --- forces at new positions ---
        F13_ = _fij(X11, X21)
        F14_ = _fij(X11, X22)
        F23_ = _fij(X12, X21)
        F24_ = _fij(X12, X22)
        F1_ = F13_ + F14_ + F23_ + F24_
        M1_, M2_ = _torques(F13_, F14_, F23_, F24_, R1, R2, _d_H2)

        # --- full-kick ---
        V1 = V1_ + (idt / _m_H2) * F1_
        V2 = V2_ - (idt / _m_H2) * F1_
        w1 = np.array([w1_[0] + idt * M1_[0] / _I, w1_[1] + idt * M1_[1] / _I, 0.0])
        w2 = np.array([w2_[0] + idt * M2_[0] / _I, w2_[1] + idt * M2_[1] / _I, 0.0])

        step += 1
        if step >= _nstep:
            break

    # ---- pack result -------------------------------------------------------
    out = np.empty(13)
    out[0] = (ek1_0 + ek2_0) / _kB
    out[1] = ek1_0 / _kB
    out[2] = ek2_0 / _kB
    out[3] = (ek1_f + ek2_f) / _kB
    out[4] = ek1_f / _kB
    out[5] = ek2_f / _kB
    out[6] = er1_0 / _kB
    out[7] = er2_0 / _kB
    out[8] = er1_f / _kB
    out[9] = er2_f / _kB
    out[10] = erel_0 / _kB
    out[11] = erel_f / _kB
    out[12] = b_norm
    return out


# ---------------------------------------------------------------------------
# Parallel runners
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _run_chunk(seed_offset, count, dist_id, T_eq, E_rel_max, bmax):
    """Run *count* collisions starting at seed *seed_offset*, in parallel."""
    results = np.empty((count, 13))
    for i in prange(count):
        results[i] = _run_one(seed_offset + i, dist_id, T_eq, E_rel_max, bmax)
    return results


def run_all_collisions(
    ncoll,
    seed=0,
    chunk_size=1000,
    dist="uniform",
    T_eq=2200.0,
    E_rel_max=20100.0,
    bfac=1.1,
):
    """Run ncoll collisions with a tqdm progress bar.

    Work is split into chunks of *chunk_size* so that tqdm can update
    between chunks.

    Parameters
    ----------
    dist : 'uniform' | 'mb' | 'ntc'
        E_rel sampling distribution. 'uniform' draws uniformly over [0, E_rel_max];
        'mb' draws from Gamma(3/2, T_eq) (Maxwell-Boltzmann); 'ntc' draws from
        Gamma(2, T_eq) (NTC-selected collision distribution).
    T_eq : float
        Equilibrium temperature [K], used by mb and ntc distributions.
    E_rel_max : float
        Upper cap [K] for the uniform distribution.
    bfac : float
        Impact-parameter multiplier; collisions are sampled with b ∈ [0, bfac·σ].

    Returns
    -------
    results : np.ndarray, shape (ncoll, 13)
    """
    dist_id = _DIST_IDS[dist]
    bmax = bfac * _sigma
    results = np.empty((ncoll, 13))
    with tqdm(total=ncoll, unit="collision") as bar:
        offset = 0
        while offset < ncoll:
            n = min(chunk_size, ncoll - offset)
            results[offset : offset + n] = _run_chunk(
                seed + offset, n, dist_id, T_eq, E_rel_max, bmax
            )
            offset += n
            bar.update(n)
    return results


# ---------------------------------------------------------------------------
# Main — sweep over bfac values
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # tag to indicate the distribution of the data
    if dist == "uniform":
        dist_tag = f"uniform_Erelmax{E_rel_max:.0f}"
    else:
        dist_tag = f"{dist}_Teq{T_eq:.0f}"

    # Trigger JIT compilation once before the timed sweep.
    print("Compiling (first run only) …")
    _ = _run_chunk(0, 1, _DIST_IDS[dist], T_eq, E_rel_max, bfac_sweep[0] * _sigma)
    print("Compilation done.\n")

    for bfac in bfac_sweep:
        t0 = time.time()
        print(
            f"Running {ncoll} collisions (bfac={bfac}, dist={dist}, T_eq={T_eq:.0f} K) …"
        )
        raw = run_all_collisions(
            ncoll,
            chunk_size=1000,
            seed=seed,
            dist=dist,
            T_eq=T_eq,
            E_rel_max=E_rel_max,
            bfac=bfac,
        )
        elapsed = time.time() - t0
        print(
            f"Done. Elapsed: {elapsed:.1f} s  ({elapsed / ncoll * 1e3:.2f} ms/collision)\n"
        )

        # ---- assemble DataFrame --------------------------------------------
        df = pd.DataFrame(
            {
                "Etr": raw[:, 10],  # relative translational KE before [K]
                "Er1": raw[:, 6],  # molecule-1 rotational KE before  [K]
                "Er2": raw[:, 7],  # molecule-2 rotational KE before  [K]
                "Etrp": raw[:, 11],  # relative translational KE after  [K]
                "Er1p": raw[:, 8],  # molecule-1 rotational KE after   [K]
                "Er2p": raw[:, 9],  # molecule-2 rotational KE after   [K]
            }
        )

        bfac_tag = str(bfac).replace(".", "_")
        savefile = paths.ensure_parent(
            paths.DATA_DIR
            / "ctc/H2/impactparam"
            / f"H2_collisions_b{bfac_tag}_{dist_tag}_ncoll{ncoll}_seed{seed}.npy"
        )
        np.save(savefile, df.to_numpy())
        print(f"Saved {savefile}")

        ratio = df["Etr"] / df["Etrp"]
        elastic_count = int(((ratio > 0.99) & (ratio < 1.01)).sum())
        inelastic_frac = 1.0 - elastic_count / len(ratio)
        print(f"Inelastic fraction (|ΔEtr/Etr| > 1%): {inelastic_frac:.3f}\n")
