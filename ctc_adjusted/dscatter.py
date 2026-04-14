"""Density-colored scatter plot.

Python translation of dscatter.m (MathWorks, 2003-2004).

The original MATLAB implementation uses a Whittaker/P-spline smoother
(penalized least squares) to estimate the 2-D point density. This port
replicates the same approach: build a 2-D histogram, apply the same
P-spline smoother along each axis, then color each scatter point by its
smoothed density value.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


# ---------------------------------------------------------------------------
# Internal smoother (direct translation of MATLAB smooth1D)
# ---------------------------------------------------------------------------

def _smooth1D(Y, lam):
    """Whittaker P-spline smoother applied column-wise.

    Solves  (I + P) Z = Y  where
        P = lam² · D2ᵀD2  +  2·lam · D1ᵀD1
    and D1, D2 are first- and second-order finite-difference matrices.

    Parameters
    ----------
    Y : np.ndarray, shape (m, n)
    lam : float
        Smoothing strength (≈ number of bins to smooth over).

    Returns
    -------
    Z : np.ndarray, shape (m, n)
    """
    m = Y.shape[0]
    E  = np.eye(m)
    D1 = np.diff(E, n=1, axis=0)   # (m-1, m)
    D2 = np.diff(D1, n=1, axis=0)  # (m-2, m)
    P  = lam**2 * (D2.T @ D2) + 2.0 * lam * (D1.T @ D1)
    Z  = solve(E + P, Y)
    return Z


def dscatter(X, Y, msize=10, smoothing=20, bins=None, ax=None):
    """Scatter plot colored by local point density.

    Parameters
    ----------
    X, Y : array_like, shape (N,)
        Data coordinates.
    msize : float, optional
        Marker size (default 10).
    smoothing : float, optional
        Smoothing factor λ (default 20, roughly 20 bins).
    bins : list of 2 ints, optional
        [nx, ny] histogram bins. Defaults to min(unique values, 200) per axis.
    ax : matplotlib Axes, optional
        Target axes. Uses plt.gca() if None.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()

    if ax is None:
        ax = plt.gca()

    if bins is None:
        nx = min(len(np.unique(X)), 200)
        ny = min(len(np.unique(Y)), 200)
        bins = [nx, ny]

    nbins = bins  # [nx, ny]

    # --- 2-D histogram (normalised) ----------------------------------------
    # np.histogram2d: first axis → X, second axis → Y
    # MATLAB accumarray(bin, 1, nbins([2 1])) stores rows=Y-bins, cols=X-bins
    # We match that layout: H shape = (ny, nx)
    edges_x = np.linspace(X.min(), X.max(), nbins[0] + 1)
    edges_y = np.linspace(Y.min(), Y.max(), nbins[1] + 1)

    # Bin indices (0-based, clamped)
    bin_x = np.clip(np.digitize(X, edges_x) - 1, 0, nbins[0] - 1)
    bin_y = np.clip(np.digitize(Y, edges_y) - 1, 0, nbins[1] - 1)

    H = np.zeros((nbins[1], nbins[0]))
    np.add.at(H, (bin_y, bin_x), 1)
    H /= len(X)  # normalise

    # --- Smooth (replicating MATLAB: G = smooth1D(H, nbins(2)/lambda);
    #                                  F = smooth1D(G', nbins(1)/lambda)' )
    G = _smooth1D(H,   nbins[1] / smoothing)
    F = _smooth1D(G.T, nbins[0] / smoothing).T

    Fmax = F.max()
    if Fmax > 0:
        F /= Fmax

    # --- Assign density value to each point ---------------------------------
    col = F[bin_y, bin_x]

    sc = ax.scatter(X, Y, c=col, s=msize, cmap="jet", marker="o")
    plt.colorbar(sc, ax=ax)
    return ax
