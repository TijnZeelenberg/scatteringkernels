import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from mdn import mdn_loss

def data_preparation(DATAFILE, nr_samples=40000):
    # import data
    rawdata = np.loadtxt(DATAFILE, delimiter=',', skiprows=1)[:nr_samples,:]

    # Make train/validation split
    train_size = int(0.7 * rawdata.shape[0])
    val_size = rawdata.shape[0] - train_size

    # Convert to the variable set (Ec, \eta_trans, \eta_rot_A)
    inputdata = np.zeros((rawdata.shape[0], 3))
    inputdata[:,0] = np.sum(rawdata[:,0:3], axis=1)
    inputdata[:,1] = rawdata[:,0]/inputdata[:,0] 
    inputdata[:,2] = rawdata[:,1] / np.sum(rawdata[:,1:3], axis=1)

    outputdata = np.zeros((rawdata.shape[0], 2))
    outputdata[:,0] = rawdata[:,3]/np.sum(rawdata[:,3:6], axis=1)
    outputdata[:,1] = rawdata[:,4]/ np.sum(rawdata[:,4:6], axis=1)

    # Create Dataloaders for training and validation (with normalization)
    inputs = torch.tensor(inputdata, dtype=torch.float32)
    outputs = torch.tensor(outputdata, dtype=torch.float32)
    return inputs, outputs, train_size, val_size


def create_dataloaders(inputs, outputs, train_size, val_size, batch_size):
    """
    Prepare DataLoaders for training and validation with normalization.
    
    :param dataset: input tensors
    :param train_size: number of training samples
    :param val_size: number of validation samples
    :param batch_size: batch size for DataLoaders
    :return: train_loader, val_loader, in_mean, in_std, out_mean, out_std
    """

    dataset = TensorDataset(inputs, outputs)

    # Create train/validation split 
    generator = torch.Generator().manual_seed(0)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_idx = torch.tensor(train_subset.indices, dtype=torch.long)
    val_idx = torch.tensor(val_subset.indices, dtype=torch.long)

    # Compute normalization statistics on training data only
    eps = 1e-8
    in_mean = inputs[train_idx].mean(dim=0, keepdim=True)
    in_std = inputs[train_idx].std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    out_mean = outputs[train_idx].mean(dim=0, keepdim=True)
    out_std = outputs[train_idx].std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)

    # Normalize tensors, then re-wrap
    inputs_norm = (inputs - in_mean) / in_std
    outputs_norm = (outputs - out_mean) / out_std
    dataset_norm = TensorDataset(inputs_norm, outputs_norm)

    train_dataset = Subset(dataset_norm, train_idx.tolist())
    val_dataset = Subset(dataset_norm, val_idx.tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, in_mean, in_std, out_mean, out_std


# Train the model
def train_model(model, train_loader, val_loader, in_mean, in_std, out_mean, out_std, num_epochs=50):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50
    train_loss_hist = np.zeros(num_epochs)
    val_loss_hist = np.zeros(num_epochs)

    print(f"Starting training using {train_loader.dataset.__len__()} training samples and {val_loader.dataset.__len__()} validation samples.")
    for epoch in tqdm(range(num_epochs), desc="Training MDN", unit="epoch", colour="green"):
        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, outputs in train_loader:
            optimizer.zero_grad()
            pi, mu, sigma = model(inputs)
            loss = mdn_loss(pi, mu, sigma, outputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_hist[epoch] = train_loss

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, outputs in val_loader:
                pi, mu, sigma = model(inputs)
                loss = mdn_loss(pi, mu, sigma, outputs)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_loss_hist[epoch] = val_loss
    return train_loss_hist, val_loss_hist


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


def plot_scattering_comparison(ctc_data,mdn_model, nr_samples=10000, in_mean=None, in_std=None, out_mean=None, out_std=None):
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
    if nr_samples > ctc_data.shape[0]:
        print(f"Number of samples requested ({nr_samples}) exceeds available CTC data ({ctc_data.shape[0]}). Using all available data.")
        nr_samples = ctc_data.shape[0]
    
    inputs = ctc_data[:nr_samples, :3]
    outputs = ctc_data[:nr_samples, 3:]

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
        inputdata=ctc_data[:nr_samples, :3],
        in_mean=in_mean,
        in_std=in_std,
        out_mean=out_mean,
        out_std=out_std,
    )

    dotsize = 3
    alpha = 1.0
    colormap = 'viridis'
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # CTC ground truth
    ax[0, 0].set_title('CTC Data')
    xy1 = np.vstack([ctc_data[:nr_samples, 1], ctc_data[:nr_samples, 3]])
    z1 = gaussian_kde(xy1)(xy1)   # density per point
    idx1 = z1.argsort()          # plot low-density first
    x1, y1 = ctc_data[:nr_samples, 1][idx1], ctc_data[:nr_samples, 3][idx1]
    ax[0, 0].scatter(x1, y1, c=z1[idx1], cmap=colormap, alpha=alpha, s=dotsize)
    ax[0,0].set_ylim(0,1)
    ax[0,0].set_xlim(0,1)
    ax[0, 0].set_xlabel(r"$\epsilon_{tr}$")
    ax[0, 0].set_ylabel(r"$\epsilon_{tr}'$")
    print("Finished plot 1")

    xy2 = np.vstack([ctc_data[:nr_samples, 2], ctc_data[:nr_samples, 4]])
    z2 = gaussian_kde(xy2)(xy2)   # density per point
    idx2 = z2.argsort()          # plot low-density first
    x2, y2 = ctc_data[:nr_samples, 2][idx2], ctc_data[:nr_samples, 4][idx2]
    ax[1, 0].scatter(x2, y2, c=z2[idx2], cmap=colormap, alpha=alpha, s=dotsize)
    ax[1,0].set_ylim(0,1)
    ax[1,0].set_xlim(0,1)
    ax[1, 0].set_xlabel(r"$\epsilon_{r,A}$")
    ax[1, 0].set_ylabel(r"$\epsilon_{r,A}'$")
    print("Finished plot 2")

    # MDN predictions
    ax[0, 1].set_title('MDN Predictions')
    xy3 = np.vstack([ctc_data[:nr_samples, 1], samples[:nr_samples, 0]])
    z3 = gaussian_kde(xy3)(xy3)   # density per point
    idx3 = z3.argsort()          # plot low-density first
    x3, y3 = ctc_data[:nr_samples, 1][idx3], samples[:nr_samples, 0][idx3]
    ax[0, 1].scatter(x3, y3, c=z3[idx3], cmap=colormap, alpha=alpha, s=dotsize)
    ax[0,1].set_ylim(0,1)
    ax[0,1].set_xlim(0,1)
    ax[0, 1].set_xlabel(r"$\epsilon_{tr}$")
    ax[0, 1].set_ylabel(r"$\epsilon_{tr}'$")
    print("Finished plot 3")

    xy4 = np.vstack([ctc_data[:nr_samples, 2], samples[:nr_samples, 1]])
    z4 = gaussian_kde(xy4)(xy4)   # density per point
    idx4 = z4.argsort()          # plot low-density first
    x4, y4 = ctc_data[:nr_samples, 2][idx4], samples[:nr_samples, 1][idx4]
    ax[1, 1].scatter(x4, y4, c=z4[idx4], cmap=colormap, alpha=alpha, s=dotsize)
    ax[1,1].set_ylim(0,1)
    ax[1,1].set_xlim(0,1)
    ax[1, 1].set_xlabel(r"$\epsilon_{r,A}$")
    ax[1, 1].set_ylabel(r"$\epsilon_{r,A}'$")
    print("Finished plot 4")

    plt.tight_layout()
    plt.savefig("scattering_comparison.png", dpi=300)
    plt.show()