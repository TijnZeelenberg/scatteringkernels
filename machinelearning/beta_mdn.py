import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
    random_split,
)
from tqdm import tqdm

# CTC datasets store energies as E/kB (Kelvin). DSMC passes energies in Joules
# at inference time, so we divide by kB before feeding the model.
_KB = 1.380649e-23


class BetaMixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network using Beta distributions for modeling post-collision
    energy fractions. Since Beta is naturally bounded to (0, 1), this is a
    natural fit for energy fractions and avoids the need for output normalization
    or hard clipping.

    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output (number of energy fractions).
        num_mixtures (int): Number of Beta mixture components.
        hidden_dim (int): Number of hidden units in the fully connected layers.
        randomseed (int): Random seed.

    Returns:
        pi (torch.Tensor): Mixture weights, shape (batch_size, num_mixtures).
        alpha (torch.Tensor): Alpha shape parameters, shape (batch_size, num_mixtures, output_dim).
        beta (torch.Tensor): Beta shape parameters, shape (batch_size, num_mixtures, output_dim).
    """

    def __init__(self, input_dim, output_dim, num_mixtures, hidden_dim, randomseed):
        super().__init__()
        self.rng = np.random.default_rng(randomseed)
        self.K = num_mixtures
        self.D = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.pi_layer = nn.Linear(hidden_dim, self.K)
        self.alpha_layer = nn.Linear(hidden_dim, self.K * self.D)
        self.beta_layer = nn.Linear(hidden_dim, self.K * self.D)

        self.input_mean: torch.Tensor = torch.empty(0)
        self.input_std: torch.Tensor = torch.empty(0)

    def forward(self, x: torch.Tensor):
        h = self.net(x)

        pi = F.softmax(self.pi_layer(h), dim=-1)

        # Both shape parameters must be strictly positive
        alpha = F.softplus(self.alpha_layer(h)) + 1e-6
        alpha = alpha.view(-1, self.K, self.D)

        beta = F.softplus(self.beta_layer(h)) + 1e-6
        beta = beta.view(-1, self.K, self.D)

        return pi, alpha, beta

    def _param_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        try:
            p = next(self.parameters())
        except StopIteration:
            return torch.device("cpu"), torch.float32
        return p.device, p.dtype

    def _cast_normalization_tensors(self) -> None:
        device, dtype = self._param_device_dtype()
        for attr in ("input_mean", "input_std"):
            t = getattr(self, attr)
            if t is not None:
                setattr(self, attr, t.to(device=device, dtype=dtype))

    def create_dataloaders(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size,
        shuffle,
        trainval_split,
        random_seed,
        weights=None,
    ):
        """
        Creates DataLoaders for training and validation.

        Inputs are normalized. Outputs (energy fractions) are NOT normalized
        since they are already in [0, 1] and the Beta distribution is defined
        on that domain.

        Args:
            X (torch.Tensor): Input features.
            y (torch.Tensor): Target energy fractions, values in [0, 1].
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle (ignored when weights are given).
            trainval_split (float): Fraction of data used for training.
            random_seed (int): Seed for the train/val split.
            weights (torch.Tensor | None): Per-sample importance weights.
        Returns:
            train_loader, val_loader (DataLoader): DataLoaders for training/validation.
        """
        if not (X.any() and y.any()):
            raise ValueError("X and y cannot be empty.")

        self.input_mean = X.mean(dim=0)
        self.input_std = X.std(dim=0) + 1e-6
        X = (X - self.input_mean) / self.input_std

        dataset = TensorDataset(X, y)
        train_size = int(trainval_split * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

        if weights is not None:
            train_weights = weights[train_dataset.indices]
            sampler = WeightedRandomSampler(
                train_weights, num_samples=len(train_dataset), replacement=True
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler
            )

            val_weights = weights[val_dataset.indices]
            val_weights = val_weights / val_weights.sum()
            val_weighted_dataset = TensorDataset(
                X[val_dataset.indices], y[val_dataset.indices], val_weights
            )
            val_loader = DataLoader(
                val_weighted_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs,
        patience: int | None = None,
    ):
        """
        Trains the Beta Mixture Density Network.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            num_epochs (int): Maximum number of epochs to train the model.
            patience (int | None): Early stopping patience.
        """
        self.train_loss_history = []
        self.val_loss_history = []
        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0
        best_epoch = 0

        device, _ = self._param_device_dtype()

        for epoch in tqdm(range(num_epochs), unit="epoch"):
            self.train()
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pi, alpha, beta = self.forward(x_batch)
                loss = beta_mdn_loss(pi, alpha, beta, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.train_loss_history.append(avg_loss)

            self.eval()
            val_loss = 0.0
            val_weight_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x_val, y_val, w_val = (t.to(device) for t in batch)
                        pi_val, alpha_val, beta_val = self.forward(x_val)
                        weighted_nll, w_sum = beta_mdn_loss_weighted(
                            pi_val, alpha_val, beta_val, y_val, w_val
                        )
                        val_loss += weighted_nll.item()
                        val_weight_total += w_sum.item()
                    else:
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        pi_val, alpha_val, beta_val = self.forward(x_val)
                        val_loss += beta_mdn_loss(
                            pi_val, alpha_val, beta_val, y_val
                        ).item()
                        val_weight_total += 1
            avg_val_loss = val_loss / val_weight_total
            self.val_loss_history.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = {k: v.clone() for k, v in self.state_dict().items()}
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    tqdm.write(
                        f"Early stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.4f})"
                    )
                    break

        if best_weights is not None:
            self.load_state_dict(best_weights)

        print(
            f"Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}"
        )
        return self.train_loss_history, self.val_loss_history

    def predict(self, x: torch.Tensor):
        """
        Returns the mixture parameters (pi, alpha, beta) for the given inputs.

        Args:
            x (torch.Tensor): Input data, shape (batch_size, input_dim).

        Returns:
            pi, alpha, beta: Mixture weights, alpha and beta shape parameters.
        """
        return self.forward(x)

    def sample(self, x: torch.Tensor):
        """
        Draws one sample per input from the predicted Beta mixture. Samples are
        naturally in (0, 1) — no output denormalization is needed.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, input_dim).
        Returns:
            samples (torch.Tensor): Sampled energy fractions, shape (batch_size, output_dim).
        """
        if self.input_mean.numel() == 0 or self.input_std.numel() == 0:
            raise ValueError(
                "Normalization parameters are not set. Ensure the model has been trained or loaded before sampling."
            )

        device, dtype = self._param_device_dtype()
        self._cast_normalization_tensors()
        x = x.to(device=device, dtype=dtype)

        self.eval()
        with torch.no_grad():
            x = (x - self.input_mean) / self.input_std
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x.clamp(min=-1e4, max=1e4)

            pi, alpha, beta = self.forward(x)

            # Ensure mixture weights are valid
            pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
            pi = torch.clamp(pi, min=0.0)
            pi_sum = pi.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(pi, 1.0 / pi.size(-1))
            pi = torch.where(pi_sum > 0, pi / pi_sum, uniform)

            # Select one component per sample
            component = torch.multinomial(pi, num_samples=1, replacement=True).squeeze(
                1
            )

            alpha_sel = alpha[torch.arange(alpha.size(0)), component]
            beta_sel = beta[torch.arange(beta.size(0)), component]

            # Draw samples from the selected Beta components
            dist = torch.distributions.Beta(alpha_sel, beta_sel)
            samples = dist.sample()

            # Beta samples are already in (0, 1) — no denormalization needed
            return samples

    def _sample_unit_direction(self, shape):
        while True:
            d = self.rng.normal(size=shape)
            n = np.linalg.norm(d)
            if n > 0.0:
                return d / n

    def collide(
        self,
        velocity_i: np.ndarray,
        e_rot_i: np.ndarray,
        velocity_j: np.ndarray,
        e_rot_j: np.ndarray,
        m: float,
        zrot: float = 1.0,
    ):
        """Performs a collision between two particles using the Beta MDN to predict post-collisional energy fractions."""

        if velocity_i.shape != velocity_j.shape:
            raise ValueError("Input velocity vectors must have the same shape.")

        g = velocity_i - velocity_j
        E_rel = 0.25 * m * np.sum(g**2)
        Etot = float(E_rel + e_rot_i + e_rot_j)
        Erot = float(e_rot_i + e_rot_j)

        if Etot <= 0 or Erot <= 0:
            return velocity_i, e_rot_i, velocity_j, e_rot_j

        direction = self._sample_unit_direction(velocity_i.shape)
        V = 0.5 * (velocity_i + velocity_j)

        if self.rng.random() > 1.0 / zrot:
            g_mag = np.sqrt(np.sum(g**2))
            g_post = direction * g_mag
            return V + 0.5 * g_post, e_rot_i, V - 0.5 * g_post, e_rot_j

        eta_tr = E_rel / Etot
        eta_rot_A = e_rot_i / Erot

        device, dtype = self._param_device_dtype()
        input_features = torch.tensor(
            [[Etot / _KB, eta_tr, eta_rot_A]], device=device, dtype=dtype
        )
        etap_tr, etap_rot_i = (
            self.sample(input_features).squeeze(0).detach().cpu().numpy()
        )

        etap_tr = float(np.clip(etap_tr, 0.0, 1.0))
        etap_rot_i = float(np.clip(etap_rot_i, 0.0, 1.0))

        E_rel_post = etap_tr * Etot
        E_rot_pool = Etot - E_rel_post
        E_rot_i_post = etap_rot_i * E_rot_pool
        E_rot_j_post = (1.0 - etap_rot_i) * E_rot_pool

        g_mag = np.sqrt(4.0 * E_rel_post / m)
        g_post = direction * g_mag

        v_i_post = V + 0.5 * g_post
        v_j_post = V - 0.5 * g_post
        return v_i_post, E_rot_i_post, v_j_post, E_rot_j_post

    def batch_collide(
        self,
        velocity_i: np.ndarray,
        e_rot_i: np.ndarray,
        velocity_j: np.ndarray,
        e_rot_j: np.ndarray,
        m: float,
        zrot: float = 1.0,
    ):
        """Performs a batch of collisions using the Beta MDN to predict post-collisional energy fractions."""

        g = velocity_i - velocity_j
        E_rel = 0.25 * m * np.sum(g**2, axis=1)
        Etot = E_rel + e_rot_i + e_rot_j
        Erot = e_rot_i + e_rot_j

        valid = (Etot > 0) & (Erot > 0)

        V = 0.5 * (velocity_i + velocity_j)
        g_speed = np.linalg.norm(g, axis=1)

        raw = self.rng.normal(size=(len(velocity_i), 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        directions = raw / np.where(norms > 0, norms, 1.0)

        g_post = directions * g_speed[:, None]
        v_i_post = V + 0.5 * g_post
        v_j_post = V - 0.5 * g_post
        e_rot_i_post = e_rot_i.copy()
        e_rot_j_post = e_rot_j.copy()

        if not np.any(valid):
            return v_i_post, e_rot_i_post, v_j_post, e_rot_j_post

        inelastic = valid & (self.rng.random(len(velocity_i)) < 1.0 / zrot)
        idx = np.where(inelastic)[0]

        if len(idx) == 0:
            return v_i_post, e_rot_i_post, v_j_post, e_rot_j_post

        eta_tr = E_rel[idx] / Etot[idx]
        eta_rot_A = e_rot_i[idx] / Erot[idx]

        device, dtype = self._param_device_dtype()
        input_tensor = torch.tensor(
            np.stack([Etot[idx] / _KB, eta_tr, eta_rot_A], axis=1),
            device=device,
            dtype=dtype,
        )
        samples = self.sample(input_tensor).detach().cpu().numpy()
        etap_tr = np.clip(samples[:, 0], 0.0, 1.0)
        etap_rot_i = np.clip(samples[:, 1], 0.0, 1.0)

        E_rel_post = etap_tr * Etot[idx]
        E_rot_pool = Etot[idx] - E_rel_post
        e_rot_i_post[idx] = etap_rot_i * E_rot_pool
        e_rot_j_post[idx] = (1.0 - etap_rot_i) * E_rot_pool

        g_mag = np.sqrt(4.0 * E_rel_post / m)
        g_post_inel = directions[idx] * g_mag[:, None]

        v_i_post[idx] = V[idx] + 0.5 * g_post_inel
        v_j_post[idx] = V[idx] - 0.5 * g_post_inel

        return v_i_post, e_rot_i_post, v_j_post, e_rot_j_post

    def save_model(self, path):
        """Saves the model state dict and input normalization parameters to a .pth file."""
        if self.input_mean.numel() == 0 or self.input_std.numel() == 0:
            raise ValueError(
                "Model has not been trained yet. Cannot save untrained model."
            )
        model_dict = {
            "state_dict": self.state_dict(),
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
        }
        torch.save(model_dict, path)

    def load_model(self, path):
        """Loads the model state dict and input normalization parameters from a .pth file."""
        model_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(model_dict["state_dict"])
        self.input_mean = model_dict["input_mean"]
        self.input_std = model_dict["input_std"]
        self._cast_normalization_tensors()


def beta_mdn_loss(pi, alpha, beta_param, y):
    """
    Negative log-likelihood loss for a Beta Mixture Density Network.

    Args:
        pi: Mixture weights, shape (batch_size, K)
        alpha: Alpha shape parameters, shape (batch_size, K, D)
        beta_param: Beta shape parameters, shape (batch_size, K, D)
        y: Target energy fractions in [0, 1], shape (batch_size, D)
    """
    # Clamp targets away from boundary to avoid log(0)
    y = y.clamp(1e-6, 1.0 - 1e-6)
    y = y.unsqueeze(1)  # (batch_size, 1, D)

    # Log Beta pdf: (α-1)*log(y) + (β-1)*log(1-y) - log B(α, β)
    log_prob = (
        (alpha - 1) * torch.log(y)
        + (beta_param - 1) * torch.log(1.0 - y)
        - torch.lgamma(alpha)
        - torch.lgamma(beta_param)
        + torch.lgamma(alpha + beta_param)
    )  # (batch_size, K, D)

    log_prob = log_prob.sum(dim=2)  # (batch_size, K)

    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    log_sum_exp = torch.logsumexp(weighted_log_prob, dim=1)

    return -torch.mean(log_sum_exp)


def beta_mdn_loss_weighted(pi, alpha, beta_param, y, w):
    """
    Weighted negative log-likelihood loss for a Beta Mixture Density Network.
    Returns (weighted_nll_sum, weight_sum) for accumulation across batches.
    """
    y = y.clamp(1e-6, 1.0 - 1e-6)
    y = y.unsqueeze(1)

    log_prob = (
        (alpha - 1) * torch.log(y)
        + (beta_param - 1) * torch.log(1.0 - y)
        - torch.lgamma(alpha)
        - torch.lgamma(beta_param)
        + torch.lgamma(alpha + beta_param)
    )

    log_prob = log_prob.sum(dim=2)

    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    log_sum_exp = torch.logsumexp(weighted_log_prob, dim=1)

    return -(w * log_sum_exp).sum(), w.sum()
