"""
Quantify agreement between MD (LAMMPS), DSMC (SPARTA), and ml-DSMC (BL) rotational
relaxation curves via relative RMSE vs the LAMMPS reference.

Relative RMSE = RMSE / ΔT  where ΔT = T_rot(t=0) - T_rot(t=t_max)

Each comparison curve is linearly interpolated onto the LAMMPS time grid before
computing the residuals. The evaluation window matches the plot xlims:
  H2: 0–2 ns,  O2: 0–4 ns.
"""

import numpy as np

lammps_h2 = np.loadtxt("data/lammps/h2_energy_relaxation.dat", skiprows=1)
sparta_h2 = np.loadtxt("data/sparta/h2_energy_relaxation.dat")
bl_h2 = np.loadtxt("data/ml-dsmc/bl/h2_energy_relaxation.dat", skiprows=1)

lammps_o2 = np.loadtxt("data/lammps/o2_energy_relaxation.dat", skiprows=1)
sparta_o2 = np.loadtxt("data/sparta/o2_energy_relaxation.dat")
bl_o2 = np.loadtxt("data/ml-dsmc/bl/o2_energy_relaxation.dat", skiprows=1)


def relative_rmse(t_ref, y_ref, t_other, y_other, t_max):
    """RMSE of y_other vs y_ref on the ref grid over [0, t_max], normalised by ΔT."""
    mask = t_ref <= t_max
    t_eval = t_ref[mask]
    y_ref_crop = y_ref[mask]
    y_interp = np.interp(t_eval, t_other, y_other)
    delta_T = abs(y_ref_crop[-1] - y_ref[0])
    rmse = np.sqrt(np.mean((y_interp - y_ref_crop) ** 2))
    return rmse, delta_T, rmse / delta_T


H2_T_MAX = 2.0  # ns, matches plot xlim
O2_T_MAX = 4.0

def time_to_95(t, y):
    """Time for y to reach 95% of its total change from initial to final value."""
    T_init, T_final = y[0], y[-1]
    T_95 = T_init + 0.95 * (T_final - T_init)
    # find first crossing
    idx = np.argmax(y >= T_95) if T_final > T_init else np.argmax(y <= T_95)
    if idx == 0:
        return float("nan")
    return float(np.interp(T_95, y[idx - 1 : idx + 1], t[idx - 1 : idx + 1]))


print(f"{'Species':<8} {'Method':<16} {'t_95 [ns]':>10}")
print("-" * 36)
for species, method, arr in [
    ("H2", "MD (LAMMPS)",   lammps_h2),
    ("H2", "BL (SPARTA)",   sparta_h2),
    ("H2", "ml-DSMC (BL)",  bl_h2),
    ("O2", "MD (LAMMPS)",   lammps_o2),
    ("O2", "BL (SPARTA)",   sparta_o2),
    ("O2", "ml-DSMC (BL)",  bl_o2),
]:
    t = arr[:, 1] * 1e9
    y = arr[:, 3]
    print(f"{species:<8} {method:<16} {time_to_95(t, y):>10.3f}")

print()

vs_lammps = [
    ("H2", "BL (SPARTA)",  lammps_h2, sparta_h2, H2_T_MAX),
    ("H2", "ml-DSMC (BL)", lammps_h2, bl_h2,    H2_T_MAX),
    ("O2", "BL (SPARTA)",  lammps_o2, sparta_o2, O2_T_MAX),
    ("O2", "ml-DSMC (BL)", lammps_o2, bl_o2,    O2_T_MAX),
]

vs_sparta = [
    ("H2", "ml-DSMC (BL)", sparta_h2, bl_h2,  H2_T_MAX),
    ("O2", "ml-DSMC (BL)", sparta_o2, bl_o2,  O2_T_MAX),
]

print(f"{'Species':<8} {'vs. ref':<8} {'Method':<16} {'RMSE [K]':>10} {'ΔT [K]':>8} {'rRMSE':>8}")
print("-" * 62)
for species, method, ref, other, t_max in vs_lammps:
    t_ref   = ref[:, 1] * 1e9
    y_ref   = ref[:, 3]
    t_other = other[:, 1] * 1e9
    y_other = other[:, 3]
    rmse, delta_T, rrmse = relative_rmse(t_ref, y_ref, t_other, y_other, t_max)
    print(f"{species:<8} {'LAMMPS':<8} {method:<16} {rmse:>10.2f} {delta_T:>8.1f} {rrmse:>8.1%}")

print()
for species, method, ref, other, t_max in vs_sparta:
    t_ref   = ref[:, 1] * 1e9
    y_ref   = ref[:, 3]
    t_other = other[:, 1] * 1e9
    y_other = other[:, 3]
    rmse, delta_T, rrmse = relative_rmse(t_ref, y_ref, t_other, y_other, t_max)
    print(f"{species:<8} {'SPARTA':<8} {method:<16} {rmse:>10.2f} {delta_T:>8.1f} {rrmse:>8.1%}")
