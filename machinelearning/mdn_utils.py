import numpy as np
import torch
import matplotlib.pyplot as plt

# Sample from the trained MDN (handles the same normalization used during training)
def sample_mdn(model, inputdata, in_mean, in_std, out_mean, out_std):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(inputdata, dtype=torch.float32)
        x = (x - in_mean) / in_std
        pi, mu, sigma = model(x)

        # Move to CPU numpy for sampling
        pi_np = pi.detach().cpu().numpy()
        mu_np = mu.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        out_mean_np = out_mean.detach().cpu().numpy().reshape(-1)
        out_std_np = out_std.detach().cpu().numpy().reshape(-1)

        # N is number of data points, K is number of mixtures, D is output dimension
        N, K, D = mu_np.shape
        samples = np.zeros((N, D), dtype=np.float32)

        for i in range(N):
            component = np.random.choice(K, p=pi_np[i])
            y_norm = np.random.normal(loc=mu_np[i, component], scale=sigma_np[i, component])
            # Denormalize back to original output space
            y = y_norm * out_std_np + out_mean_np
            samples[i] = y

    return samples


def plot_scattering_comparison(ctc_data, mdn_model):
    """Visualize the comparison between CTC scattering data and MDN model predictions.
    Args:
        ctc_data (np.ndarray): array of shape (N, 5) with columns
            [E_c, eta_tr_in, eta_r_A_in, eta_tr_out, eta_r_A_out].
        mdn_model (MixtureDensityNetwork): trained MDN model for predictions.
    """
    samples = sample_mdn(
        model=mdn_model,
        inputdata=ctc_data[:, :3],
        in_mean=in_mean,
        in_std=in_std,
        out_mean=out_mean,
        out_std=out_std,
    )

    dotsize = 4
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # CTC ground truth
    ax[0, 0].set_title('CTC Data')
    ax[0, 0].scatter(ctc_data[:, 1], ctc_data[:, 3], alpha=0.5, s=dotsize)
    ax[0,0].set_ylim(0,1)
    ax[0,0].set_xlim(0,1)
    ax[0, 0].set_xlabel(r"$\eta_{tr}$")
    ax[0, 0].set_ylabel(r"$\eta_{tr}'$")

    ax[1, 0].scatter(ctc_data[:, 2], ctc_data[:, 4], alpha=0.5, s=dotsize)
    ax[1,0].set_ylim(0,1)
    ax[1,0].set_xlim(0,1)
    ax[1, 0].set_xlabel(r"$\eta_{r,A}$")
    ax[1, 0].set_ylabel(r"$\eta_{r,A}'$")

    # MDN predictions
    ax[0, 1].set_title('MDN Predictions')
    ax[0, 1].scatter(ctc_data[:, 1], samples[:, 0], alpha=0.5, s=dotsize)
    ax[0,1].set_ylim(0,1)
    ax[0,1].set_xlim(0,1)
    ax[0, 1].set_xlabel(r"$\eta_{tr}$")
    ax[0, 1].set_ylabel(r"$\eta_{tr}'$")

    ax[1, 1].scatter(ctc_data[:, 2], samples[:, 1], alpha=0.5, s=dotsize)
    ax[1,1].set_ylim(0,1)
    ax[1,1].set_xlim(0,1)
    ax[1, 1].set_xlabel(r"$\eta_{r,A}$")
    ax[1, 1].set_ylabel(r"$\eta_{r,A}'$")

    plt.tight_layout()
    plt.show()