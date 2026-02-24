import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# Model Definition
class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network for modeling scattering kernels.
    This model predicts a mixture of Gaussians for the scattering kernel, allowing it to capture complex distributions.
    
    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output (e.g., scattering angles).
        num_mixtures (int): Number of Gaussian mixtures to use.
        hidden_dim (int, optional): Number of hidden units in the fully connected layers. Default is 128.
    
    Returns:
        pi (torch.Tensor): Mixture weights, shape (batch_size, num_mixtures).
        mu (torch.Tensor): Means of the mixtures, shape (batch_size, num_mixtures, output_dim).
        sigma (torch.Tensor): Standard deviations of the mixtures, shape (batch_size, num_mixtures, output_dim).
    """
    
    def __init__(self, input_dim, output_dim, num_mixtures, hidden_dim=128):
        super().__init__()
        self.K = num_mixtures
        self.D = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.pi_layer = nn.Linear(hidden_dim, self.K) # Mixture weights
        self.mu_layer = nn.Linear(hidden_dim, self.K * self.D) # Mixture means
        self.sigma_layer = nn.Linear(hidden_dim, self.K * self.D) # Mixture standard deviations

        self.input_mean: torch.Tensor | None = None
        self.input_std: torch.Tensor | None = None
        self.output_mean: torch.Tensor | None = None
        self.output_std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
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
    
    def create_dataloaders(self, X, y, batch_size, shuffle, trainval_split, random_seed): 
        """
        Creates a DataLoader for the given dataset.
        
        Args:
            X (torch.Tensor): Input features, shape (n_samples, input_dim).
            y (torch.Tensor): Target values, shape (n_samples, output_dim).
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at every epoch.
        Returns:
            dataloader (DataLoader): DataLoader for the dataset.
        """
        # Normalize the data
        self.input_mean = X.mean(dim=0)
        self.input_std = X.std(dim=0) + 1e-6
        self.output_mean = y.mean(dim=0)
        self.output_std = y.std(dim=0) + 1e-6
        X = (X - self.input_mean) / self.input_std
        y = (y - self.output_mean) / self.output_std

        # Split the dataset into training and validation sets
        dataset = TensorDataset(X, y)
        train_size = int(trainval_split * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=generator) 

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def train_model(self, train_loader:DataLoader, val_loader:DataLoader, optimizer:torch.optim.Optimizer, num_epochs, lr):
        """
        Trains the Mixture Density Network using the provided training data.
        
        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            num_epochs (int): Number of epochs to train the model.
            lr (float): Learning rate for the optimizer.
        """
        # Training loop
        self.train_loss_history = []
        self.val_loss_history = []
        for epoch in tqdm(range(num_epochs), unit="epoch"):
            self.train()
            total_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pi, mu, sigma = self.forward(x_batch)
                loss = mdn_loss(pi, mu, sigma, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.train_loss_history.append(avg_loss)

            # Validation loop
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    pi_val, mu_val, sigma_val = self.forward(x_val)
                    val_loss += mdn_loss(pi_val, mu_val, sigma_val, y_val).item()
            avg_val_loss = val_loss / len(val_loader)
            self.val_loss_history.append(avg_val_loss)

        return self.train_loss_history, self.val_loss_history

        

    def predict(self, x: torch.Tensor):
        """
        Predicts the mixture parameters for the input data.
        
        Args:
            x (torch.Tensor): Input data, shape (batch_size, input_dim).
        
        Returns:
            [pi, mu, sigma]: Tuple containing mixture weights, means, and standard deviations.
                pi (torch.Tensor): Mixture weights, shape (batch_size, num_mixtures).
                mu (torch.Tensor): Means of the mixtures, shape (batch_size, num_mixtures, output_dim).
                sigma (torch.Tensor): Standard deviations of the mixtures, shape (batch_size, num_mixtures, output_dim).
        """
        return self.forward(x)

    def sample(self, x: torch.Tensor):
        """
        Generates samples from the predicted mixture of Gaussians.
        
        Args:
            x (torch.Tensor): Input features, shape (batch_size, input_dim).
            num_samples_per_input (int): Number of samples to generate per input sample.
        Returns:
            samples (torch.Tensor): Generated samples, shape (batch_size, n_samples, output_dim).
        """
        if self.input_mean is None or self.input_std is None or self.output_mean is None or self.output_std is None:
            raise ValueError("Normalization parameters are not set. Ensure the model has been trained or loaded before sampling.")

        self.eval()
        with torch.no_grad():
            # Normalize the input
            x = (x - self.input_mean) / self.input_std

            pi, mu, sigma = self.forward(x)

            # Sample one component per input according to the mixture weights
            components = torch.multinomial(pi, num_samples=1, replacement=True).squeeze(1) 

            # select mu and sigma for the chosen components
            mu_sel = mu[torch.arange(mu.size(0)), components]
            sigma_sel = sigma[torch.arange(sigma.size(0)), components]

            # Draw Gaussian samples
            samples = mu_sel + torch.randn_like(mu_sel) * sigma_sel

            # De-normalize outputs
            samples = samples * self.output_std + self.output_mean 
            return samples

    def save_model(self, path):
        """
        Saves the model state dictionary to a .pt file.
        
        Args:
            path (str): Path to save the model, must end with .pt.
        """
        if self.input_mean is None or self.input_std is None or self.output_mean is None or self.output_std is None:
            raise ValueError("Model has not been trained yet. Cannot save untrained model.")
        model_dict = {
            "state_dict": self.state_dict(),
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "output_mean": self.output_mean,
            "output_std": self.output_std,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history
        }
        torch.save(model_dict, path)

    def load_model(self, path):
        """
        Loads the model state dictionary from a .pt file.
        
        Args:
            path (str): Path to load the model from, must end with .pt.
        """
        model_dict = torch.load(path)
        self.load_state_dict(model_dict["state_dict"])
        self.input_mean = model_dict["input_mean"]
        self.input_std = model_dict["input_std"]
        self.output_mean = model_dict["output_mean"]
        self.output_std = model_dict["output_std"]


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
