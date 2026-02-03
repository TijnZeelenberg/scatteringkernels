import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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


def plot_scattering_comparison(ctc_data, mdn_model, in_mean=None, in_std=None, out_mean=None, out_std=None):
    """Visualize the comparison between CTC scattering data and MDN model predictions.
    Args:
        ctc_data (np.ndarray): array of shape (N, 5) with columns
            [E_c, eta_tr_in, eta_r_A_in, eta_tr_out, eta_r_A_out].
        mdn_model (MixtureDensityNetwork): trained MDN model for predictions.
        in_mean (torch.Tensor | np.ndarray | None): input normalization mean.
        in_std (torch.Tensor | np.ndarray | None): input normalization std.
        out_mean (torch.Tensor | np.ndarray | None): output normalization mean.
        out_std (torch.Tensor | np.ndarray | None): output normalization std.
    """
    eps = 1e-8
    inputs = ctc_data[:, :3]
    outputs = ctc_data[:, 3:]

    if in_mean is None:
        in_mean = torch.tensor(inputs.mean(axis=0, keepdims=True), dtype=torch.float32)
    elif not isinstance(in_mean, torch.Tensor):
        in_mean = torch.tensor(in_mean, dtype=torch.float32)

    if in_std is None:
        in_std_np = inputs.std(axis=0, ddof=0, keepdims=True)
        in_std = torch.tensor(np.maximum(in_std_np, eps), dtype=torch.float32)
    elif not isinstance(in_std, torch.Tensor):
        in_std = torch.tensor(in_std, dtype=torch.float32)

    if out_mean is None:
        out_mean = torch.tensor(outputs.mean(axis=0, keepdims=True), dtype=torch.float32)
    elif not isinstance(out_mean, torch.Tensor):
        out_mean = torch.tensor(out_mean, dtype=torch.float32)

    if out_std is None:
        out_std_np = outputs.std(axis=0, ddof=0, keepdims=True)
        out_std = torch.tensor(np.maximum(out_std_np, eps), dtype=torch.float32)
    elif not isinstance(out_std, torch.Tensor):
        out_std = torch.tensor(out_std, dtype=torch.float32)
    
    samples = sample_mdn(
        model=mdn_model,
        inputdata=ctc_data[:, :3],
        in_mean=in_mean,
        in_std=in_std,
        out_mean=out_mean,
        out_std=out_std,
    )

    dotsize = 4
    alpha = 1.0
    colormap = 'viridis'
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # CTC ground truth
    ax[0, 0].set_title('CTC Data')
    xy1 = np.vstack([ctc_data[:, 1], ctc_data[:, 3]])
    z1 = gaussian_kde(xy1)(xy1)   # density per point
    idx1 = z1.argsort()          # plot low-density first
    x1, y1 = ctc_data[:, 1][idx1], ctc_data[:, 3][idx1]
    ax[0, 0].scatter(x1, y1, c=z1[idx1], cmap=colormap, alpha=alpha, s=dotsize)
    ax[0,0].set_ylim(0,1)
    ax[0,0].set_xlim(0,1)
    ax[0, 0].set_xlabel(r"$\eta_{tr}$")
    ax[0, 0].set_ylabel(r"$\eta_{tr}'$")

    xy2 = np.vstack([ctc_data[:, 2], ctc_data[:, 4]])
    z2 = gaussian_kde(xy2)(xy2)   # density per point
    idx2 = z2.argsort()          # plot low-density first
    x2, y2 = ctc_data[:, 2][idx2], ctc_data[:, 4][idx2]
    ax[1, 0].scatter(x2, y2, c=z2[idx2], cmap=colormap, alpha=alpha, s=dotsize)
    ax[1,0].set_ylim(0,1)
    ax[1,0].set_xlim(0,1)
    ax[1, 0].set_xlabel(r"$\eta_{r,A}$")
    ax[1, 0].set_ylabel(r"$\eta_{r,A}'$")

    # MDN predictions
    ax[0, 1].set_title('MDN Predictions')
    xy3 = np.vstack([ctc_data[:, 1], samples[:, 0]])
    z3 = gaussian_kde(xy3)(xy3)   # density per point
    idx3 = z3.argsort()          # plot low-density first
    x3, y3 = ctc_data[:, 1][idx3], samples[:, 0][idx3]
    ax[0, 1].scatter(x3, y3, c=z3[idx3], cmap=colormap, alpha=alpha, s=dotsize)
    ax[0,1].set_ylim(0,1)
    ax[0,1].set_xlim(0,1)
    ax[0, 1].set_xlabel(r"$\eta_{tr}$")
    ax[0, 1].set_ylabel(r"$\eta_{tr}'$")

    xy4 = np.vstack([ctc_data[:, 2], samples[:, 1]])
    z4 = gaussian_kde(xy4)(xy4)   # density per point
    idx4 = z4.argsort()          # plot low-density first
    x4, y4 = ctc_data[:, 2][idx4], samples[:, 1][idx4]
    ax[1, 1].scatter(x4, y4, c=z4[idx4], cmap=colormap, alpha=alpha, s=dotsize)
    ax[1,1].set_ylim(0,1)
    ax[1,1].set_xlim(0,1)
    ax[1, 1].set_xlabel(r"$\eta_{r,A}$")
    ax[1, 1].set_ylabel(r"$\eta_{r,A}'$")

    plt.tight_layout()
    plt.show()
