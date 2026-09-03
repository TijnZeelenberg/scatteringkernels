import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)
from tqdm import tqdm

# CTC datasets store energies as E/kB (Kelvin). DSMC passes energies in Joules
# at inference time, so we divide by kB before feeding the model.
_KB = 1.380649e-23


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

        self.pi_layer = nn.Linear(hidden_dim, self.K)  # Mixture weights
        self.mu_layer = nn.Linear(hidden_dim, self.K * self.D)  # Mixture means
        self.sigma_layer = nn.Linear(
            hidden_dim, self.K * self.D
        )  # Mixture standard deviations

        self.input_mean: torch.Tensor = torch.empty(0)
        self.input_std: torch.Tensor = torch.empty(0)
        self.output_mean: torch.Tensor = torch.empty(0)
        self.output_std: torch.Tensor = torch.empty(0)

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

        Args:
            X (torch.Tensor): Input features
            y (torch.Tensor): Target values
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle (ignored when weights are given).
            trainval_split (float): Fraction of data used for training.
            random_seed (int): Seed for the train/val split.
            weights (torch.Tensor | None): Per-sample importance weights for the
                training set. When provided, training samples are drawn with
                replacement proportional to these weights (WeightedRandomSampler),
                so the model is trained on the reweighted distribution. The
                validation loader is always unweighted.
        Returns:
            train_loader, val_loader (DataLoader): DataLoaders for training/validation.
        """
        # Normalize the data. When importance weights are given, compute the
        # mean/std over the *weighted* distribution so the model sees inputs
        # centered on the regime it actually trains on (NTC-reweighted), not
        # the regime CTC happened to sample uniformly from.
        if not (X.any() and y.any()):
            raise ValueError("X and y cannot be empty.")
        if weights is not None:
            w = (weights / weights.sum()).unsqueeze(1)
            self.input_mean = (X * w).sum(dim=0)
            self.input_std = (
                torch.sqrt(((X - self.input_mean) ** 2 * w).sum(dim=0)) + 1e-6
            )
            self.output_mean = (y * w).sum(dim=0)
            self.output_std = (
                torch.sqrt(((y - self.output_mean) ** 2 * w).sum(dim=0)) + 1e-6
            )
        else:
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

        # Create DataLoaders. When importance weights are present we no longer
        # use WeightedRandomSampler — at low ESS it collapses the effective
        # training set to a handful of repeated draws per epoch. Instead we
        # iterate uniformly over *every* unique sample and let the per-sample
        # weight enter the loss (importance-weighted SGD). Mathematically the
        # same expected objective, vastly more gradient signal per epoch.
        if weights is not None:
            train_weights = weights[train_dataset.indices]
            train_weighted_dataset = TensorDataset(
                X[train_dataset.indices], y[train_dataset.indices], train_weights
            )
            train_loader = DataLoader(
                train_weighted_dataset, batch_size=batch_size, shuffle=shuffle
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
        Trains the Mixture Density Network using the provided training data.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            num_epochs (int): Maximum number of epochs to train the model.
            patience (int | None): Early stopping patience — stop if val loss does not
                improve for this many consecutive epochs.
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
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                if len(batch) == 3:
                    x_batch, y_batch, w_batch = (t.to(device) for t in batch)
                    pi, mu, sigma = self.forward(x_batch)
                    nll_sum, w_sum = mdn_loss_weighted(pi, mu, sigma, y_batch, w_batch)
                    loss = nll_sum / w_sum.clamp(min=1e-8)
                else:
                    x_batch, y_batch = (t.to(device) for t in batch)
                    pi, mu, sigma = self.forward(x_batch)
                    loss = mdn_loss(pi, mu, sigma, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            n_batches = len(train_loader)
            self.train_loss_history.append(total_loss / n_batches)

            # Validation loop
            self.eval()
            val_loss = 0.0
            val_weight_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x_val, y_val, w_val = (t.to(device) for t in batch)
                        pi_val, mu_val, sigma_val = self.forward(x_val)
                        weighted_nll, w_sum = mdn_loss_weighted(
                            pi_val, mu_val, sigma_val, y_val, w_val
                        )
                        val_loss += weighted_nll.item()
                        val_weight_total += w_sum.item()
                    else:
                        x_val, y_val = (t.to(device) for t in batch)
                        pi_val, mu_val, sigma_val = self.forward(x_val)
                        val_loss += mdn_loss(pi_val, mu_val, sigma_val, y_val).item()
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

        # Restore weights from the best epoch
        if best_weights is not None:
            self.load_state_dict(best_weights)

        print(
            f"Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}"
        )
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
        Generates data samples from the predicted mixture of Gaussians. One sample is drawn for each input in the batch.

        Args:
            x (torch.Tensor): Input features
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

    def collide(
        self,
        velocity_i: np.ndarray,
        e_rot_i: np.ndarray,
        velocity_j: np.ndarray,
        e_rot_j: np.ndarray,
        m: float,
        zrot: float = 1.0,
    ):
        """Performs a collision between two particles using the MDN to predict post-collisional energy fractions."""

        if velocity_i.shape != velocity_j.shape:
            raise ValueError("Input velocity vectors must have the same shape.")

        # Compute precollisional energy fractions.
        # Etot is the redistributable energy (relative KE + rotational) — COM KE is
        # conserved implicitly through V and does not enter the redistribution pool.
        g = velocity_i - velocity_j
        E_rel = 0.25 * m * np.sum(g**2)
        Etot = float(E_rel + e_rot_i + e_rot_j)
        Erot = float(e_rot_i + e_rot_j)

        # Guard against zero energy cases
        if Etot <= 0 or Erot <= 0:
            return velocity_i, e_rot_i, velocity_j, e_rot_j

        # Sample isotropic random velocity direction (used by both branches)
        direction = self._sample_unit_direction(velocity_i.shape)
        V = 0.5 * (velocity_i + velocity_j)

        if self.rng.random() > 1.0 / zrot:
            # Elastic collision: randomize direction, preserve relative speed and rotational energies
            g_mag = np.sqrt(np.sum(g**2))
            g_post = direction * g_mag
            return V + 0.5 * g_post, e_rot_i, V - 0.5 * g_post, e_rot_j

        eta_tr = E_rel / Etot
        eta_rot_A = e_rot_i / Erot

        # Particle-swap symmetrization. Particles i and j are physically
        # identical, so the kernel ought to be invariant under
        # (eta_rot_A, eta_rot_A') -> (1 - eta_rot_A, 1 - eta_rot_A'). The
        # trained MDN doesn't get that for free, so we enforce it at sample
        # time: with probability 0.5, feed (1 - eta_rot_A) and invert the
        # sampled output. Marginally exact symmetrization, zero retrain cost.
        swap = self.rng.random() < 0.5
        eta_rot_A_in = 1.0 - eta_rot_A if swap else eta_rot_A

        # Sample new energy fractions from the predicted mixture of Gaussians
        device, dtype = self._param_device_dtype()
        input_features = torch.tensor(
            [[Etot / _KB, eta_tr, eta_rot_A_in]], device=device, dtype=dtype
        )
        etap_tr, etap_rot_i = (
            self.sample(input_features).squeeze(0).detach().cpu().numpy()
        )

        # Physical constraints: energy fractions must lie in [0, 1].
        etap_tr = float(np.clip(etap_tr, 0.0, 1.0))
        etap_rot_i = float(np.clip(etap_rot_i, 0.0, 1.0))
        if swap:
            etap_rot_i = 1.0 - etap_rot_i

        # Reconstruct post-collisional energies.
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
        """Performs a batch of collisions using the MDN to predict post-collisional energy fractions.
        Args:
            velocity_i (np.ndarray): Pre-collisional velocities of particle i
            e_rot_i (np.ndarray): Pre-collisional rotational energies of particle i
            velocity_j (np.ndarray): Pre-collisional velocities of particle j
            e_rot_j (np.ndarray): Pre-collisional rotational energies of particle j
            m (float): Mass of the particles
            zrot (float): Rotational degree of freedom parameter
        Returns:
            v_i_post (np.ndarray): Post-collisional velocities of particle i
            e_rot_i_post (np.ndarray): Post-collisional rotational energies of particle i
            v_j_post (np.ndarray): Post-collisional velocities of particle j
            e_rot_j_post (np.ndarray): Post-collisional rotational energies of particle j
        """

        # Compute precollisional energy fractions.
        # Etot is the redistributable energy (relative KE + rotational) — COM KE is
        # conserved implicitly through V and does not enter the redistribution pool.
        g = velocity_i - velocity_j  # (N, 3)
        E_rel = 0.25 * m * np.sum(g**2, axis=1)  # (N,)
        Etot = E_rel + e_rot_i + e_rot_j  # (N,)
        Erot = e_rot_i + e_rot_j  # (N,)

        # Guard against degenerate collisions; process only valid pairs.
        valid = (Etot > 0) & (Erot > 0)

        V = 0.5 * (velocity_i + velocity_j)  # (N, 3)
        g_speed = np.linalg.norm(g, axis=1)  # (N,)

        # Sample isotropic random velocity directions for all pairs (used by both branches)
        raw = self.rng.normal(size=(len(velocity_i), 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        directions = raw / np.where(norms > 0, norms, 1.0)

        # Default: elastic collision — randomize direction, preserve speed and rotational energies
        g_post = directions * g_speed[:, None]
        v_i_post = V + 0.5 * g_post
        v_j_post = V - 0.5 * g_post
        e_rot_i_post = e_rot_i.copy()
        e_rot_j_post = e_rot_j.copy()

        if not np.any(valid):
            return v_i_post, e_rot_i_post, v_j_post, e_rot_j_post

        # Inelastic collisions: valid pairs selected by zrot probability
        inelastic = valid & (self.rng.random(len(velocity_i)) < 1.0 / zrot)
        idx = np.where(inelastic)[0]

        if len(idx) == 0:
            return v_i_post, e_rot_i_post, v_j_post, e_rot_j_post

        eta_tr = E_rel[idx] / Etot[idx]
        eta_rot_A = e_rot_i[idx] / Erot[idx]

        # Particle-swap symmetrization (see `collide` for the motivation).
        # With probability 0.5 per pair, feed (1 - eta_rot_A) to the kernel
        # and invert the sampled output. Marginally enforces invariance under
        # i <-> j exchange without retraining.
        swap_mask = self.rng.random(len(idx)) < 0.5
        eta_rot_A_in = np.where(swap_mask, 1.0 - eta_rot_A, eta_rot_A)

        # Sample new energy fractions from the predicted mixture of Gaussians
        device, dtype = self._param_device_dtype()
        input_tensor = torch.tensor(
            np.stack([Etot[idx] / _KB, eta_tr, eta_rot_A_in], axis=1),
            device=device,
            dtype=dtype,
        )
        samples = self.sample(input_tensor).detach().cpu().numpy()
        etap_tr = np.clip(samples[:, 0], 0.0, 1.0)
        etap_rot_i = np.clip(samples[:, 1], 0.0, 1.0)
        etap_rot_i = np.where(swap_mask, 1.0 - etap_rot_i, etap_rot_i)

        # Reconstruct post-collisional energies.
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
        """
        Saves the model state dictionary to a .pth file.

        Args:
            path (str): Path to save the model, must end with .pth.
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
        Loads the model state dictionary from a .pth file.

        Args:
            path (str): Path to load the model from, must end with .pth.
        """
        model_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(model_dict["state_dict"])
        self.input_mean = model_dict["input_mean"]
        self.input_std = model_dict["input_std"]
        self.output_mean = model_dict["output_mean"]
        self.output_std = model_dict["output_std"]
        self._cast_normalization_tensors()


# Define loss function
def _mdn_log_prob(pi, mu, sigma, y):
    """Per-row log p(y | pi, mu, sigma) for a Gaussian MDN. Shape (batch_size,)."""
    y = y.unsqueeze(1)
    log_prob = -0.5 * (
        torch.sum(((y - mu) / sigma) ** 2, dim=2)
        + torch.sum(torch.log(sigma**2), dim=2)
        + mu.size(2) * torch.log(torch.tensor(2 * torch.pi))
    )
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    return torch.logsumexp(weighted_log_prob, dim=1)


def mdn_loss(pi, mu, sigma, y):
    """Negative log-likelihood loss for a Gaussian MDN (mean over batch)."""
    return -torch.mean(_mdn_log_prob(pi, mu, sigma, y))


def mdn_loss_weighted(pi, mu, sigma, y, w):
    """
    Weighted negative log-likelihood loss for a Mixture Density Network.
    Returns (weighted_nll_sum, weight_sum) so the caller can accumulate a
    properly normalized weighted mean across batches.
    """
    log_p = _mdn_log_prob(pi, mu, sigma, y)
    return -(w * log_p).sum(), w.sum()
