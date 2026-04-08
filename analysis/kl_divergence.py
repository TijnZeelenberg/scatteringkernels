from scipy.stats import gaussian_kde, entropy
import numpy as np


def kl_divergence(p_samples, q_samples, n_grid=1000):
    """Computes the KL divergence D_KL(p || q) by estimating densities with Gaussian KDE
    on a shared grid, then calling scipy.stats.entropy.

    Args:
        p_samples: 1D array-like of samples from distribution p
        q_samples: 1D array-like of samples from distribution q
        n_grid: number of evaluation points for the shared grid

    Returns:
        kl_div (float): The KL divergence D_KL(p || q).
    """
    p_samples = np.asarray(p_samples).ravel()
    q_samples = np.asarray(q_samples).ravel()

    x_min = min(p_samples.min(), q_samples.min())
    x_max = max(p_samples.max(), q_samples.max())
    grid = np.linspace(x_min, x_max, n_grid)

    p_pdf = gaussian_kde(p_samples)(grid)
    q_pdf = gaussian_kde(q_samples)(grid)

    return float(entropy(p_pdf, q_pdf))