import numpy as np
import torch
import matplotlib.pyplot as plt

# Sample from the trained MDN
def sample_mdn(model, inputdata):
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(inputdata)

        # N is number of data points, K is number of mixtures, D is output dimension
        N, K, D = mu.shape
        samples = np.zeros((N, D))

        # Reconstruct distribution per data point
        for i in range(N):
            # Sample a mixture component based on pi
            component = np.random.choice(K, p=pi[i].numpy())

            # Sample from the selected Gaussian
            sampled_point = np.random.normal(
                loc=mu[i, component].numpy(),
                scale=sigma[i, component].numpy()
            )
            samples[i] = sampled_point


    return samples


# Visualize MDN scattering vs CTC scattering
def plot_scattering_comparison(ctc_data, mdn_model, num_samples=1000):
    """Visualize the comparison between CTC scattering data and MDN model predictions.
    Args:
        ctc_data (np.ndarray): array of shape (N, 5) containing CTC data with columns
                               [E_c, eta_tr_in, eta_r_A_in, eta_tr_out, eta_r_A_out].
        mdn_model (MixtureDensityNetwork): trained MDN model for predictions.
        num_samples (int): number of samples to draw from the MDN for visualization.
    Returns:
        None
        """
    
    samples = sample_mdn(mdn_model, num_samples)

    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].set_title('CTC Data')
    ax[0, 0].scatter(e_tr_in, e_tr_out, alpha=0.5, label='CTC Data')
    ax[0, 0].set_xlabel(r"$\eta_{tr}$")
    ax[0, 0].set_ylabel(r"$\eta_{tr}'$")

    ax[1, 0].scatter(e_r_A_in, e_r_A_out, alpha=0.5, label='CTC Data')
    ax[1, 0].set_xlabel(r"$\eta_{r,A}$")
    ax[1, 0].set_ylabel(r"$\eta_{r,A}'$")

    ax[0,1].set_title('MDN Predictions')
    ax[0, 1].scatter(e_tr_in, mdn_samples[:,0], alpha=0.5, label='CTC Data')
    ax[0, 1].set_xlabel(r"$\eta_{r,A}$")
    ax[0, 1].set_ylabel(r"$\eta_{r,A}'$")

    ax[1, 1].scatter(e_tr_in, mdn_samples[:,1], alpha=0.5, label='CTC Data')
    ax[1, 1].set_xlabel(r"$\eta_{r,A}$")
    ax[1, 1].set_ylabel(r"$\eta_{r,A}'$")

    plt.tight_layout()
    plt.show()
    return