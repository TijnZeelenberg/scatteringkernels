"""Green-Kubo viscosity from DSMC stress autocorrelations.

Given a `stats` dict from `DSMC_Simulation.get_stats()` (which records
Pxy/Pxz/Pyz at every timestep), `green_kubo_viscosity` returns the integrated
shear viscosity. Independent script `H2viscosity.py` calls these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paths


_KB = 1.380649e-23  # J/K


@dataclass
class ViscosityResult:
    """Green-Kubo viscosity decomposition for later plotting/analysis."""

    viscosity: float          # Pa·s
    T_eq: float               # equilibrium translational temperature, K
    acf_xy: np.ndarray
    acf_xz: np.ndarray
    acf_yz: np.ndarray
    acf_avg: np.ndarray
    dt: float
    volume: float


def autocorrelation_fft(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized stress autocorrelation up to `max_lag` time steps, via FFT."""
    n = len(signal)
    padded = np.zeros(2 * n)
    padded[:n] = signal
    fft = np.fft.rfft(padded)
    acf_full = np.fft.irfft(fft * np.conj(fft))[:max_lag]
    counts = np.arange(n, n - max_lag, -1)
    return acf_full / counts


def green_kubo_viscosity(
    stats: dict,
    *,
    dt: float,
    volume: float,
    equilibration_steps: int = 50,
    max_lag: int = 100,
) -> ViscosityResult:
    """Compute Green-Kubo shear viscosity from DSMC stress traces.

    Args:
        stats: dict from `DSMC_Simulation.get_stats()` with Pxy/Pxz/Pyz arrays.
        dt: simulation timestep in seconds.
        volume: simulation cell volume in m^3.
        equilibration_steps: number of initial steps to discard.
        max_lag: maximum ACF lag (in steps) to integrate over.
    """
    pxy = np.asarray(stats["Pxy"][equilibration_steps:]).copy()
    pxz = np.asarray(stats["Pxz"][equilibration_steps:]).copy()
    pyz = np.asarray(stats["Pyz"][equilibration_steps:]).copy()
    pxy -= pxy.mean()
    pxz -= pxz.mean()
    pyz -= pyz.mean()

    acf_xy = autocorrelation_fft(pxy, max_lag)
    acf_xz = autocorrelation_fft(pxz, max_lag)
    acf_yz = autocorrelation_fft(pyz, max_lag)
    acf_avg = (acf_xy + acf_xz + acf_yz) / 3.0

    T_eq = float(np.mean(stats["T_trans_mean"][equilibration_steps:]))
    viscosity = (volume / (_KB * T_eq)) * float(np.trapezoid(acf_avg, dx=dt))

    return ViscosityResult(
        viscosity=viscosity,
        T_eq=T_eq,
        acf_xy=acf_xy,
        acf_xz=acf_xz,
        acf_yz=acf_yz,
        acf_avg=acf_avg,
        dt=dt,
        volume=volume,
    )


def plot_acf(
    result: ViscosityResult,
    *,
    output_path: str | Path,
    reference_viscosity_micro: float | None = None,
    reference_label: str | None = None,
):
    """Plot the stress ACFs and the cumulative Green-Kubo integral, save figure."""
    lags = np.arange(len(result.acf_avg)) * result.dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(lags * 1e6, result.acf_xy / result.acf_avg[0], alpha=0.4, label="Pxy")
    ax1.plot(lags * 1e6, result.acf_xz / result.acf_avg[0], alpha=0.4, label="Pxz")
    ax1.plot(lags * 1e6, result.acf_yz / result.acf_avg[0], alpha=0.4, label="Pyz")
    ax1.plot(lags * 1e6, result.acf_avg / result.acf_avg[0], "k-", lw=2, label="Average")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("Normalized ACF")
    ax1.set_title(f"Stress Autocorrelation (T_eq = {result.T_eq:.1f} K)")
    ax1.legend()

    prefactor = result.volume / (_KB * result.T_eq)
    running = prefactor * np.cumsum(result.acf_avg) * result.dt
    ax2.plot(lags * 1e6, running * 1e6, "k-", lw=2)
    ax2.axhline(result.viscosity * 1e6, color="r", ls="--", lw=1,
                label=f"Final: {result.viscosity:.2e} Pa·s")
    if reference_viscosity_micro is not None:
        ax2.axhline(reference_viscosity_micro, color="g", ls="--", lw=1,
                    label=reference_label or "Reference")
    ax2.set_xlabel("Lag time (μs)")
    ax2.set_ylabel("η (μPa·s)")
    ax2.set_title("Cumulative Green-Kubo Viscosity")
    ax2.legend()
    plt.tight_layout()

    fig.savefig(paths.ensure_parent(output_path), dpi=150)
    return fig
