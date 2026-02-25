from scipy.stats import gaussian_kde
import numpy as np
import torch
from torch.nn.functional import kl_div


def kl_divergence(p, q):
    """Computes the KL divergence between two distributions p and q using Gaussian KDE for density estimation.
    
    Args:
        p (torch.Tensor): First distribution, shape (n_samples, n_features).
        q (torch.Tensor): Second distribution, shape (n_samples, n_features).

    Returns:
        kl_div (float): The KL divergence D_KL(p || q).
    """
    return kl_div(p, q, reduction='mean')