import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm


# Model Definition
class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network for modeling scattering kernels.
    This model predicts a mixture of Gaussians for the scattering kernel, allowing it to capture complex distributions.

    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output
        num_mixtures (int): Number of Gaussian mixtures to use.
        hidden_dim (int, optional): Number of hidden units in the fully connected layers. Default is 128.

    Returns:
        pi (torch.Tensor): Mixture weights, shape (batch_size, num_mixtures).
        mu (torch.Tensor): Means of the mixtures, shape (batch_size, num_mixtures, output_dim).
        sigma (torch.Tensor): Standard deviations of the mixtures, shape (batch_size, num_mixtures, output_dim).
    """

    def __init__(self, input_dim, output_dim, num_mixtures, hidden_dim, randomseed, dropout=0.0):
        super().__init__()
        self.rng = np.random.default_rng(randomseed)
        self.K = num_mixtures
        self.D = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pi_layer = nn.Linear(hidden_dim, self.K)  # Mixture weights
        self.mu_layer = nn.Linear(hidden_dim, self.K * self.D)  # Mixture means
        self.sigma_layer = nn.Linear(
            hidden_dim, self.K * self.D
        )  # Mixture standard deviations

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

    def _param_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        """Return (device, dtype) of the model parameters."""
        try:
            p = next(self.parameters())
        except StopIteration:
            return torch.device("cpu"), torch.float32
        return p.device, p.dtype

    def _cast_normalization_tensors(self) -> None:
        """Keep normalization tensors on same device/dtype as model parameters."""
        device, dtype = self._param_device_dtype()
        for attr in ("input_mean", "input_std", "output_mean", "output_std"):
            t = getattr(self, attr)
            if t is not None:
                setattr(self, attr, t.to(device=device, dtype=dtype))

    def create_dataloaders(
        self, X, y, batch_size, shuffle, trainval_split, random_seed
    ):
        """
        Creates a DataLoader for the given dataset.

        Args:
            X (torch.Tensor): Input features
            y (torch.Tensor): Target values
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
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs,
        lr,
    ):
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
        Generates data samples from the predicted mixture of Gaussians.

        Args:
            x (torch.Tensor): Input features
            num_samples_per_input (int): Number of samples to generate per input sample.
        Returns:
            samples (torch.Tensor): Generated data samples
        """
        if (
            self.input_mean is None
            or self.input_std is None
            or self.output_mean is None
            or self.output_std is None
        ):
            raise ValueError(
                "Normalization parameters are not set. Ensure the model has been trained or loaded before sampling."
            )

        device, dtype = self._param_device_dtype()
        self._cast_normalization_tensors()
        x = x.to(device=device, dtype=dtype)

        self.eval()
        with torch.no_grad():
            # Normalize the input
            x = (x - self.input_mean) / self.input_std

            # Guard against pathological inputs (e.g. Etot=0 -> NaNs).
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x.clamp(min=-1e4, max=1e4)

            pi, mu, sigma = self.forward(x)

            # Ensure probabilities are valid for multinomial.
            pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
            pi = torch.clamp(pi, min=0.0)
            pi_sum = pi.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(pi, 1.0 / pi.size(-1))
            pi = torch.where(pi_sum > 0, pi / pi_sum, uniform)

            # Sample one component per input according to the mixture weights
            component = torch.multinomial(pi, num_samples=1, replacement=True).squeeze(
                1
            )

            # select mu and sigma for the chosen components
            mu_sel = mu[torch.arange(mu.size(0)), component]
            sigma_sel = sigma[torch.arange(sigma.size(0)), component]

            # Draw Gaussian samples
            samples = mu_sel + torch.randn_like(mu_sel) * sigma_sel

            # De-normalize outputs
            samples = samples * self.output_std + self.output_mean
            return samples

    def _sample_unit_direction(self, shape):
        """
        Samples random unit vectors uniformly distributed on the surface of a hyper sphere.

        Args:
            shape (tuple): Desired shape of the output tensor, should be (batch_size, output_dim).

        Returns:
            directions (torch.Tensor): Sampled unit direction vectors, shape (batch_size, output_dim).
        """
        while True:
            d = self.rng.normal(size=shape)
            n = np.linalg.norm(d)
            if n > 0.0:
                return d / n

    def collide(self, velocity_i, e_rot_i, velocity_j, e_rot_j, m):
        if velocity_i.shape != velocity_j.shape:
            raise ValueError("Input velocity vectors must have the same shape.")

        # Compute precollisional energy fractions
        # Etot here includes center-of-mass kinetic energy (because it uses absolute velocities).
        g = velocity_i - velocity_j
        E_rel = 0.25 * m * np.sum(g**2, axis=1)
        Etot = float(E_rel + e_rot_i + e_rot_j)
        Erot = float(e_rot_i + e_rot_j)

        # Degenerate (near-zero) collisions: keep state unchanged.
        if not np.isfinite(Etot) or Etot <= 0.0:
            return velocity_i, e_rot_i, velocity_j, e_rot_j

        eta_tr = E_rel / Etot
        eta_rot_A = e_rot_i / Erot

        if not (np.isfinite(eta_tr) and np.isfinite(eta_rot_A)):
            return velocity_i, e_rot_i, velocity_j, e_rot_j

        # Sample new energy fractions from the predicted mixture of Gaussians
        device, dtype = self._param_device_dtype()
        input_features = torch.tensor(
            [[Etot, eta_tr, eta_rot_A]],
            device=device,
            dtype=dtype,
        )
        etap_tr, etap_rot_A = (
            self.sample(input_features).squeeze(0).detach().cpu().numpy()
        )

        # Physical constraints: energy fractions must lie in [0, 1].
        etap_tr = float(np.clip(etap_tr, 0.0, 1.0))
        etap_rot_A = float(np.clip(etap_rot_A, 0.0, 1.0))

        # Enforce momentum conservation by working in the COM frame.
        # Center-of-mass velocity (equal masses assumed)
        V = 0.5 * (velocity_i + velocity_j)
        E_com = float(m * np.dot(V, V))

        # TODO: take another look at this code 2026-03-31
        # Interpret the sampled translational fraction as target total translational energy,
        # but COM kinetic energy is fixed, so only the relative part can change.
        Etr_target = float(np.clip(etap_tr * Etot, 0.0, Etot))
        E_rel_post = max(0.0, Etr_target - E_com)

        # Relative translational energy cannot exceed the available non-COM energy.
        E_available = max(0.0, Etot - E_com)
        E_rel_post = float(np.clip(E_rel_post, 0.0, E_available))

        # Remaining energy goes into rotation.
        E_rot_pool = float(max(0.0, E_available - E_rel_post))
        E_rot_i_post = float(etap_rot_A * E_rot_pool)
        E_rot_j_post = float((1.0 - etap_rot_A) * E_rot_pool)

        # Sample isotropic relative velocity direction and set magnitude from E_rel_post.
        direction = self._sample_unit_direction(velocity_i.shape)
        g_mag = float(np.sqrt(max(0.0, 4.0 * E_rel_post / m)))
        g_post = direction * g_mag

        v_i_post = V + 0.5 * g_post
        v_j_post = V - 0.5 * g_post
        return v_i_post, E_rot_i_post, v_j_post, E_rot_j_post

    def batch_collide(self, velocity_i, e_rot_i, velocity_j, e_rot_j, m):
        # --- Compute COM-frame energies ---
        V = 0.5 * (velocity_i + velocity_j)  # (N, 3)
        g = velocity_i - velocity_j  # (N, 3)
        Erel = 0.25 * m * np.sum(g**2, axis=1)  # (N,)
        Etot = Erel + e_rot_i + e_rot_j  # (N,) — only redistributable energy
        Erot_total = e_rot_i + e_rot_j  # (N,) — total rotational energy

        # --- COM-frame fractions (safe denominator; invalid rows masked below) ---
        safe_Etot = np.where(Etot > 0, Etot, 1.0)
        safe_Erot = np.where(Erot_total > 0, Erot_total, 1.0)
        eta_tr = Erel / safe_Etot
        eta_rot_A = e_rot_i / safe_Erot

        # --- Degenerate collision mask ---
        valid = (
            np.isfinite(Etot)
            & (Etot > 0)
            & np.isfinite(eta_tr)
            & np.isfinite(eta_rot_A)
        )

        v_i_post = velocity_i.copy()
        v_j_post = velocity_j.copy()
        e_rot_i_post = e_rot_i.copy()
        e_rot_j_post = e_rot_j.copy()

        if not np.any(valid):
            return v_i_post, e_rot_i_post, v_j_post, e_rot_j_post

        idx = np.where(valid)[0]
        input_features = np.stack(
            [Etot[idx], eta_tr[idx], eta_rot_A[idx]],
            axis=1,
        )

        device, dtype = self._param_device_dtype()
        input_tensor = torch.tensor(input_features, device=device, dtype=dtype)

        samples = self.sample(input_tensor).detach().cpu().numpy()
        xi_rel_post = np.clip(samples[:, 0], 0.0, 1.0)
        xi_rot_A_post = np.clip(samples[:, 1], 0.0, 1.0)

        # --- Reconstruct post-collision state ---
        E_avail_v = Etot[idx]
        V_v = V[idx]

        E_rel_post = xi_rel_post * E_avail_v
        E_rot_pool = E_avail_v - E_rel_post
        e_rot_i_post[idx] = xi_rot_A_post * E_rot_pool
        e_rot_j_post[idx] = E_rot_pool - e_rot_i_post[idx]

        # Isotropic random directions
        raw = self.rng.normal(size=(len(idx), 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        directions = raw / norms

        g_mag = np.sqrt(np.maximum(0.0, 4.0 * E_rel_post / m))
        g_post = directions * g_mag[:, None]

        v_i_post[idx] = V_v + 0.5 * g_post
        v_j_post[idx] = V_v - 0.5 * g_post

        return v_i_post, e_rot_i_post, v_j_post, e_rot_j_post

    def save_model(self, path):
        """
        Saves the model state dictionary to a .pt file.

        Args:
            path (str): Path to save the model, must end with .pt.
        """
        if (
            self.input_mean is None
            or self.input_std is None
            or self.output_mean is None
            or self.output_std is None
        ):
            raise ValueError(
                "Model has not been trained yet. Cannot save untrained model."
            )
        model_dict = {
            "state_dict": self.state_dict(),
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "output_mean": self.output_mean,
            "output_std": self.output_std,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
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
        self._cast_normalization_tensors()


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
        + torch.sum(torch.log(sigma**2), dim=2)
        + mu.size(2) * torch.log(torch.tensor(2 * torch.pi))
    )  # Shape (batch_size, K)

    # Weighted log probabilities
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    log_sum_exp = torch.logsumexp(weighted_log_prob, dim=1)

    return -torch.mean(log_sum_exp)
