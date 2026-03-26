from sklearn.mixture import GaussianMixture
from typing import Literal

import torch

CovarianceType = Literal['full', 'tied', 'diag', 'spherical']

class GaussianMixtureModel:
    """
    Gaussian Mixture Model for modeling scattering kernels.
    This model fits a mixture of Gaussians to the scattering kernel data, allowing it to capture complex distributions.
    
    Args:
        n_components (int): Number of Gaussian mixtures to use.
        covariance_type (str, optional): Type of covariance parameters. Default is 'full'.
    
    Returns:
        gmm (GaussianMixture): Fitted Gaussian Mixture Model.
    """
    
    def __init__(self, n_components: int, covariance_type: CovarianceType = 'full'):
        self.n_components: int = n_components
        self.covariance_type: CovarianceType = covariance_type
        self.gmm: GaussianMixture | None = None

    def fit(self, x):
        """
        Fits the GMM to the data.
        
        Args:
            x (torch.Tensor): Training data, shape (n_samples, n_features).
        """
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type)
        x = x.cpu().numpy()
        self.gmm.fit(x)

    def predict(self, x):
        """
        Predicts the labels for the data samples in X using the fitted GMM.
        
        Args:
            x (torch.Tensor): Input data, shape (n_samples, n_features).
        
        Returns:
            labels (array): Predicted labels for each sample, shape (n_samples,).
        """
        if self.gmm is None:
            raise ValueError("Model has not been fitted yet.")
        x = x.cpu().numpy()
        return self.gmm.predict(x)

    def sample(self, num_samples):
        """
        Generates random samples from the fitted GMM.
        
        Args:
            num_samples (int): Number of samples to generate.
        
        Returns:
            samples (array): Generated samples, shape (n_samples, n_features).
        """
        if self.gmm is None:
            raise ValueError("Model has not been fitted yet.")
        samples, _ = self.gmm.sample(num_samples)

        # Convert samples back to torch.Tensor
        samples = torch.tensor(samples)

        return samples