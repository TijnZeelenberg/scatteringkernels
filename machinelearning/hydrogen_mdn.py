import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn as nn
import torch.nn.functional as F

gas = "H2H2"
batch_size = 128
hidden_dim = 128

# import data
if gas == "O2O2":
    DATA = "datasets/O2O2_collisions.csv"
elif gas == "H2H2":
    DATA = "datasets/H2H2_collisions.csv"
rawdata = np.loadtxt(DATA, delimiter=',', skiprows=1)

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

print("Input mean:", in_mean.flatten().tolist())
print("Input std :", in_std.flatten().tolist())
print("Output mean:", out_mean.flatten().tolist())
print("Output std :", out_std.flatten().tolist())

# Model Definition
class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures, hidden_dim=hidden_dim):
        super().__init__()
        self.K = num_mixtures
        self.D = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.pi_layer = nn.Linear(hidden_dim, self.K)
        self.mu_layer = nn.Linear(hidden_dim, self.K * self.D)
        self.sigma_layer = nn.Linear(hidden_dim, self.K * self.D)

    def forward(self, x):
        h = self.net(x)

        # Mixture weights
        pi = F.softmax(self.pi_layer(h), dim=-1)

        # Means
        mu = self.mu_layer(h)
        mu = mu.view(-1, self.K, self.D)

        # Standard deviations
        sigma = self.sigma_layer(h)
        sigma = F.softplus(sigma) + 1e-6  # Ensure positivity
        # sigma = torch.exp(sigma)
        sigma = sigma.view(-1, self.K, self.D)

        return pi, mu, sigma

# Define loss function
def mdn_loss(pi, mu, sigma, y):
    """
    Computes the negative log-likelihood loss for a Mixture Density Network.
    Args:
        pi: Mixture weights, shape (batch_size, K)
        mu: Means of the mixtures, shape (batch_size, K, D)
        sigma: Standard deviations of the mixtures, shape (batch_size, K, D)
        y: Target values, shape (batch_size, D)
    """
    y = y.unsqueeze(1)  # Shape (batch_size, 1, D)

    # Gaussian probability density function
    log_prob = -0.5 * (
        torch.sum(((y - mu) / sigma) ** 2, dim=2)
        + torch.sum(torch.log(sigma ** 2), dim=2)
        + mu.size(2) * torch.log(torch.tensor(2 * torch.pi))
    )  # Shape (batch_size, K)

    # Weighted log probabilities
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    log_sum_exp = torch.logsumexp(weighted_log_prob, dim=1) 

    return -torch.mean(log_sum_exp)

# Training the model
if __name__ == "__main__":
    K = 5
    output_dim = 2

    model = MixtureDensityNetwork(
        input_dim=3,
        output_dim=output_dim,
        num_mixtures=K
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50
    train_loss_hist = np.zeros(num_epochs)
    val_loss_hist = np.zeros(num_epochs)

    for epoch in range(num_epochs):
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
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

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
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss_hist[epoch]:.4f}")

    # save model
    if gas == "O2O2":
        torch.save(model.state_dict(), "trainedmodels/mdn_o2o2_collision_model.pth")
    elif gas == "H2H2":
        torch.save(model.state_dict(), "trainedmodels/mdn_h2h2_collision_model.pth")


    # Plot loss
    import matplotlib.pyplot as plt

    plt.plot(range(1, num_epochs + 1), train_loss_hist, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_hist, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MDN Loss')
    plt.title('Training Loss over Epochs')
    plt.grid()
    plt.show()
    plt.savefig(f'trainedmodels/mdn_{gas}_collision_loss.jpg')