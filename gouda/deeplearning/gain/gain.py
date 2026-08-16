"""
GAIN: Generative Adversarial Imputation Networks
=================================================
Based on: Yoon, J., Jordon, J., & van der Schaar, M. (2018).
          GAIN: Missing Data Imputation using Generative Adversarial Nets.
          ICML 2018. https://arxiv.org/abs/1806.02920

Architecture
------------
- Generator  : observes corrupted data + mask, outputs imputations for missing entries.
- Discriminator: observes (possibly imputed) data + hint, classifies which components
                 were originally observed vs. imputed.
- Hint mechanism: leaks a random subset of the true mask to the discriminator so the
                  generator is forced to produce realistic imputations.

Stability features
------------------
* Gradient clipping on both networks.
* Label smoothing for the discriminator.
* Spectral normalisation on discriminator linear layers.
* Learning-rate schedulers (cosine annealing).
* Early stopping on discriminator + generator losses with patience.
* Input normalisation (min-max) inside the class — inverse-transformed on output.
* NaN / Inf guards after every forward pass.
* Deterministic seeding for reproducibility.
"""

from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
try:
    import torch
except ImportError as e:
    raise ImportError(
        "GAIN requires the 'deep' extra. "
        "Install it with: pip install gouda-cheese[deep]"
    ) from e

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute._base import _BaseImputer
from sklearn.base import TransformerMixin
from sklearn.utils.validation import validate_data
from sklearn.exceptions import NotFittedError
from gouda.gouda import raise_if_nan_col
from pyglue import Encoder


# ---------------------------------------------------------------------------
# Spectral-normalised linear layer helper
# ---------------------------------------------------------------------------

def _sn_linear(in_f: int, out_f: int) -> nn.Linear:
    """Linear layer with spectral normalisation (stabilises discriminator)."""
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Maps (X_corrupted ⊕ Mask) → imputed values for the missing positions.

    Input  : 2 * dim  (corrupted features concatenated with binary mask)
    Output : dim      (one value per original feature, passed through sigmoid
                       so it lives in [0, 1] after normalisation)

    Parameters
    ----------
    dim         : number of input features.
    hidden_dim  : width of hidden layers.
    num_layers  : total number of Linear→Norm→Act blocks (≥ 2).
    dropout     : dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        categorical_sizes: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be ≥ 2.")

        layers: list[nn.Module] = []

        # Input block
        layers += [nn.Linear(dim * 2, hidden_dim),
                   nn.LayerNorm(hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden blocks
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)
        self.numerical_head = nn.Linear(hidden_dim, dim)
        self.categorical_heads = nn.ModuleDict({
            str(column): nn.Linear(hidden_dim, size)
            for column, size in (categorical_sizes or {}).items()
        })
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """
        Parameters
        ----------
        x    : (B, dim) — observed values; missing positions can be 0 or random noise.
        mask : (B, dim) — 1 = observed, 0 = missing.

        Returns
        -------
        g_out : (B, dim) — generator output for ALL positions (mix with mask outside).
        """
        inp = torch.cat([x, mask], dim=1)
        hidden = self.trunk(inp)
        generated = torch.sigmoid(self.numerical_head(hidden))
        categorical_logits = {
            int(column): head(hidden)
            for column, head in self.categorical_heads.items()
        }

        # Keep the discriminator input compact. Each categorical distribution
        # is represented by its differentiable expected, normalised class ID.
        generated = generated.clone()
        for column, logits in categorical_logits.items():
            probabilities = torch.softmax(logits, dim=1)
            classes = torch.arange(
                logits.shape[1], device=logits.device, dtype=logits.dtype
            )
            denominator = max(logits.shape[1] - 1, 1)
            generated[:, column] = (probabilities * classes).sum(dim=1) / denominator

        return generated, categorical_logits


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Maps (X_imputed ⊕ Hint) → P(component was originally observed) per feature.

    Input  : 2 * dim  (imputed/completed data concatenated with hint vector)
    Output : dim      (probability each feature was observed, via sigmoid)

    Spectral normalisation on every linear layer keeps Lipschitz constant bounded,
    which is the primary stability mechanism for the discriminator.

    Parameters
    ----------
    dim         : number of input features.
    hidden_dim  : width of hidden layers.
    num_layers  : total number of SN-Linear→Act blocks (≥ 2).
    dropout     : dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be ≥ 2.")

        layers: list[nn.Module] = []

        # Input block (spectral norm — NO LayerNorm here, it interferes with SN)
        layers += [_sn_linear(dim * 2, hidden_dim), nn.LeakyReLU(0.2)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden blocks
        for _ in range(num_layers - 2):
            layers += [_sn_linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output block — sigmoid → probability per feature
        layers += [_sn_linear(hidden_dim, dim), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_hat: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_hat : (B, dim) — completed data (observed ∪ generated).
        hint  : (B, dim) — partial mask hint vector.

        Returns
        -------
        d_out : (B, dim) — per-feature probability of being originally observed.
        """
        inp = torch.cat([x_hat, hint], dim=1)
        return self.net(inp)


class GAIN(_BaseImputer, TransformerMixin):
    def __init__(self, *,
                 # Width of hidden layers in both G and D.
                 hidden_dim: int = 256,
                 num_layers: int = 3,           # Depth of both networks (≥ 2).
                 # Dropout rate in both networks (0 = disabled).
                 dropout: float = 0.0,
                 # Fraction of mask bits revealed to discriminator (0 < p ≤ 1).
                 hint_rate: float = 0.9,
                 # Weight on the MSE reconstruction term in G's loss.
                 alpha: float = 100.0,
                 batch_size: int = 256,         # Mini-batch size.
                 max_epochs: int = 300,         # Maximum training epochs.
                 # Initial learning rate for both G and D.
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,    # L2 regularisation.
                 # Maximum gradient norm (per network).
                 grad_clip: float = 5.0,
                 # Label smoothing applied to discriminator targets (e.g. 0.1).
                 label_smoothing: float = 0.1,
                 # Early-stopping patience (epochs without joint-loss improvement).
                 patience: int = 30,
                 # Minimum relative improvement to reset patience counter.
                 min_delta: float = 1e-4,
                 # 'cpu', 'cuda', 'mps', or None (auto-detect).
                 device: str | None = None,
                 # Integer seed for full reproducibility, or None.
                 random_state: int | None = 42,
                 # Print training progress every `verbose` epochs (0 = silent).
                 verbose: int = 0,
                 encoding: None | str = None,
                 # required by _BaseImputer
                 missing_values=np.nan,
                 add_indicator: bool = False,
                 keep_empty_features: bool = False,) -> None:

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing
        self.patience = patience
        self.min_delta = min_delta
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.encoding = encoding
        self.missing_values = missing_values
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0, 1], ignoring NaNs."""
        rng = self._data_max - self._data_min  # (dim,)
        rng = np.where(rng == 0, 1.0, rng)     # avoid /0 for constant columns
        return (X - self._data_min) / rng

    def _denormalise(self, X: np.ndarray) -> np.ndarray:
        rng = self._data_max - self._data_min
        rng = np.where(rng == 0, 1.0, rng)
        return X * rng + self._data_min

    @staticmethod
    def _make_mask(X: np.ndarray) -> np.ndarray:
        """1 = observed, 0 = missing."""
        return (~np.isnan(X)).astype(np.float32)

    def _build_hint(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Hint vector: with probability `hint_rate` reveal the true mask bit;
        otherwise set to 0.5 (uninformative).
        """
        hint_mask = (
            torch.bernoulli(torch.full_like(
                mask, self.hint_rate)).to(mask.device)
        )
        return mask * hint_mask + 0.5 * (1.0 - hint_mask)

    @staticmethod
    def _disc_loss(
        d_out: torch.Tensor,
        mask: torch.Tensor,
        smoothing: float,
    ) -> torch.Tensor:
        """
        Binary cross-entropy with optional label smoothing.
        Target: 1 (observed) / 0 (imputed), smoothed to (1-ε) / ε.
        """
        target_obs = 1.0 - smoothing
        target_imp = smoothing
        targets = mask * target_obs + (1.0 - mask) * target_imp
        return F.binary_cross_entropy(d_out, targets)

    @staticmethod
    def _gen_loss(
        d_out: torch.Tensor,
        mask: torch.Tensor,
        x_hat: torch.Tensor,
        x_norm: torch.Tensor,
        categorical_logits: dict[int, torch.Tensor],
        categorical_targets: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Generator loss = adversarial term (fool discriminator on MISSING entries)
                       + alpha * MSE reconstruction on OBSERVED entries.
        """
        # Adversarial: want discriminator to output 1 on missing positions
        missing_mask = 1.0 - mask
        n_missing = missing_mask.sum().clamp(min=1.0)
        adv_loss = -((torch.log(d_out + 1e-8)) *
                     missing_mask).sum() / n_missing

        # Reconstruction: penalise deviation from known values on observed entries
        numerical_mask = mask.clone()
        for column in categorical_logits:
            numerical_mask[:, column] = 0.0
        numerical_count = numerical_mask.sum().clamp(min=1.0)
        mse_loss = (
            (x_hat - x_norm) ** 2 * numerical_mask
        ).sum() / numerical_count

        categorical_loss = x_hat.new_tensor(0.0)
        categorical_count = 0
        for column, logits in categorical_logits.items():
            observed_rows = mask[:, column].bool()
            if observed_rows.any():
                categorical_loss = categorical_loss + F.cross_entropy(
                    logits[observed_rows],
                    categorical_targets[observed_rows, column].long(),
                )
                categorical_count += 1
        if categorical_count:
            categorical_loss = categorical_loss / categorical_count

        return adv_loss + alpha * (mse_loss + categorical_loss)

    def fit(self, X: np.ndarray, y=None) -> "GAIN":
        """
        Train GAIN on X (may contain NaNs).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored (sklearn compatibility)

        Returns
        -------
        self
        """
        self.check_parameters()
        self._encoder = Encoder(self.encoding)
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        X, cat_cols = self._encoder.encode(X)
        X = np.array(X, dtype=np.float64)
        raise_if_nan_col(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")

        n_samples, dim = X.shape
        if n_samples < 2:
            raise ValueError(
                f"n_samples = {n_samples}, GAIN requires at least 2 samples to fit.")

        # ---- reproducibility -----------------------------------------
        if self.random_state is not None:
            self._set_seed(self.random_state)

        # ---- device ---------------------------------------------------
        self._device = self._resolve_device()

        # ---- normalise ------------------------------------------------
        self._dim = dim
        self._cat_cols = sorted(cat_cols or [])
        self._categorical_sizes = {
            column: int(np.nanmax(X[:, column])) + 1
            for column in self._cat_cols
        }
        self._data_min = np.nanmin(X, axis=0)
        self._data_max = np.nanmax(X, axis=0)

        mask_np = self._make_mask(X)            # (N, dim) float32
        X_norm = self._normalise(X).astype(np.float32)
        # Replace NaNs with 0 in normalised array (masked out in losses)
        X_fill = np.where(mask_np == 1, X_norm, 0.0).astype(np.float32)

        X_t = torch.from_numpy(X_fill).to(self._device)
        mask_t = torch.from_numpy(mask_np).to(self._device)
        X_norm_t = torch.from_numpy(
            np.where(mask_np == 1, X_norm, 0.0).astype(np.float32)
        ).to(self._device)
        categorical_targets_t = torch.from_numpy(
            np.where(mask_np == 1, X, 0.0).astype(np.int64)
        ).to(self._device)

        dataset = TensorDataset(X_t, mask_t, X_norm_t, categorical_targets_t)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, n_samples),
            shuffle=True,
            drop_last=(n_samples > self.batch_size),
        )

        # ---- build networks ------------------------------------------
        self.generator_ = Generator(
            dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            self._categorical_sizes,
        ).to(self._device)
        self.discriminator_ = Discriminator(
            dim, self.hidden_dim, self.num_layers, self.dropout
        ).to(self._device)

        opt_G = torch.optim.Adam(
            self.generator_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        opt_D = torch.optim.Adam(
            self.discriminator_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_G, T_max=self.max_epochs, eta_min=self.lr * 1e-2
        )
        sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_D, T_max=self.max_epochs, eta_min=self.lr * 1e-2
        )

        hist_d: list[float] = []
        hist_g: list[float] = []
        best_joint = float("inf")
        patience_counter = 0

        # ---- training loop -------------------------------------------
        for epoch in range(1, self.max_epochs + 1):
            epoch_d, epoch_g = 0.0, 0.0
            n_batches = 0

            for x_batch, m_batch, xn_batch, cat_target_batch in loader:
                B = x_batch.size(0)

                # ---------- Discriminator step ----------------------
                self.discriminator_.train()
                self.generator_.eval()

                # Add small noise to observed inputs (instance noise → stability)
                noise = torch.randn_like(x_batch) * 0.01
                x_noisy = x_batch + noise * m_batch  # only on observed

                with torch.no_grad():
                    g_out, _ = self.generator_(x_noisy, m_batch)

                # Complete data: use observed values + generator for missing
                x_hat = m_batch * x_batch + (1.0 - m_batch) * g_out

                hint = self._build_hint(m_batch)
                d_out = self.discriminator_(x_hat.detach(), hint)

                loss_D = self._disc_loss(d_out, m_batch, self.label_smoothing)

                opt_D.zero_grad()
                loss_D.backward()
                nn.utils.clip_grad_norm_(
                    self.discriminator_.parameters(), self.grad_clip
                )
                opt_D.step()

                # ---------- Generator step --------------------------
                self.generator_.train()
                self.discriminator_.eval()

                g_out, categorical_logits = self.generator_(x_noisy, m_batch)
                x_hat = m_batch * x_batch + (1.0 - m_batch) * g_out

                hint = self._build_hint(m_batch)
                with torch.no_grad():
                    d_out = self.discriminator_(x_hat, hint)

                # Re-forward discriminator with grad only for G's adv loss
                # (we need d_out to have grad w.r.t. g_out)
                d_out_g = self.discriminator_(x_hat, hint)
                loss_G = self._gen_loss(
                    d_out_g,
                    m_batch,
                    g_out,
                    xn_batch,
                    categorical_logits,
                    cat_target_batch,
                    self.alpha,
                )

                opt_G.zero_grad()
                loss_G.backward()
                nn.utils.clip_grad_norm_(
                    self.generator_.parameters(), self.grad_clip
                )
                opt_G.step()

                # ---- NaN guard ------------------------------------
                if torch.isnan(loss_D) or torch.isnan(loss_G):
                    warnings.warn(
                        f"NaN loss detected at epoch {epoch}. "
                        "Stopping training early. Consider reducing lr or alpha.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break

                epoch_d += loss_D.item()
                epoch_g += loss_G.item()
                n_batches += 1
            else:
                # Only step schedulers if no NaN break occurred in inner loop
                sched_D.step()
                sched_G.step()

                epoch_d /= max(n_batches, 1)
                epoch_g /= max(n_batches, 1)
                hist_d.append(epoch_d)
                hist_g.append(epoch_g)

                # ---- logging -------------------------------------
                if self.verbose > 0 and epoch % self.verbose == 0:
                    print(
                        f"Epoch {epoch:4d}/{self.max_epochs}  "
                        f"D-loss: {epoch_d:.4f}  G-loss: {epoch_g:.4f}  "
                        f"LR: {sched_G.get_last_lr()[0]:.2e}"
                    )

                # ---- early stopping ------------------------------
                joint = epoch_d + epoch_g
                if joint < best_joint * (1.0 - self.min_delta):
                    best_joint = joint
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose > 0:
                            print(
                                f"Early stopping at epoch {epoch} "
                                f"(no improvement for {self.patience} epochs)."
                            )
                        break
                continue  # inner loop completed normally
            break           # inner loop had a NaN — exit outer loop too

        # Set both networks to eval mode after training
        self.generator_.eval()
        self.discriminator_.eval()
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values in X using the trained generator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            May contain NaNs in the same or different positions as training data.

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features), no NaNs.
        """
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError("This GAIN instance is not fitted yet. Call 'fit' before 'transform'.")

        if self.generator_ is None:
            raise RuntimeError("Call fit() before transform().")

        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None,
                          ensure_all_finite="allow-nan", reset=False)
        X, cat_cols = self._encoder.encode(X)
        cat_cols = sorted(cat_cols or [])
        if cat_cols != self._cat_cols:
            raise ValueError(
                "Categorical columns in transform data do not match the fitted data."
            )
        X = np.array(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        if X.shape[1] != self._dim:
            raise ValueError(
                f"Expected {self._dim} features, got {X.shape[1]}."
            )

        mask_np = self._make_mask(X)
        X_norm = self._normalise(X).astype(np.float32)
        X_fill = np.where(mask_np == 1, X_norm, 0.0).astype(np.float32)

        X_t = torch.from_numpy(X_fill).to(self._device)
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(self._device)

        self.generator_.eval()
        with torch.no_grad():
            g_out, categorical_logits = self.generator_(X_t, mask_t)

            # Training uses softmax expectations so gradients can reach the
            # generator. Inference uses an actual class for lossless decoding.
            for column, logits in categorical_logits.items():
                denominator = max(logits.shape[1] - 1, 1)
                g_out[:, column] = logits.argmax(dim=1).to(g_out.dtype) / denominator

            # Keep observed values; fill missing with generator output
            x_hat = mask_t * X_t + (1.0 - mask_t) * g_out

        x_hat_np = x_hat.cpu().numpy().astype(np.float64)
        # Inverse normalise back to original scale
        X_imputed = self._denormalise(x_hat_np)

        # Safety: restore any originally-observed values exactly
        obs_idx = mask_np == 1
        X_imputed[obs_idx] = X[obs_idx]

        X_imputed = self._encoder.decode(X_imputed)
        if is_df:
            X_imputed = pd.DataFrame(X_imputed, columns=columns)
        return X_imputed

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit the model and return the imputed training data."""
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "hidden_dim":      self.hidden_dim,
            "num_layers":      self.num_layers,
            "dropout":         self.dropout,
            "hint_rate":       self.hint_rate,
            "alpha":           self.alpha,
            "batch_size":      self.batch_size,
            "max_epochs":      self.max_epochs,
            "lr":              self.lr,
            "weight_decay":    self.weight_decay,
            "grad_clip":       self.grad_clip,
            "label_smoothing": self.label_smoothing,
            "patience":        self.patience,
            "min_delta":       self.min_delta,
            "device":          self.device,
            "random_state":    self.random_state,
            "verbose":         self.verbose,
            "encoding":        self.encoding,
            "missing_values":  self.missing_values,
            "add_indicator":   self.add_indicator,
            "keep_empty_features": self.keep_empty_features,
        }

    def set_params(self, **params) -> "GAIN":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter '{k}'.")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:  # pragma: no cover
        p = self.get_params()
        args = ", ".join(f"{k}={v!r}" for k, v in p.items())
        return f"GAIN({args})"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True   # declares intentional string/categorical support
        return tags

    def check_parameters(self) -> None:
        """Validate GAIN hyperparameters before fitting."""

        if not isinstance(self.hidden_dim, int) or isinstance(self.hidden_dim, bool):
            raise TypeError("hidden_dim must be an integer.")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")

        if not isinstance(self.num_layers, int) or isinstance(self.num_layers, bool):
            raise TypeError("num_layers must be an integer.")
        if self.num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise TypeError("dropout must be a number.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        if not isinstance(self.hint_rate, (int, float)) or isinstance(
            self.hint_rate, bool
        ):
            raise TypeError("hint_rate must be a number.")
        if not 0.0 < self.hint_rate <= 1.0:
            raise ValueError("hint_rate must be in the range (0.0, 1.0].")

        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool):
            raise TypeError("alpha must be a number.")
        if self.alpha < 0.0:
            raise ValueError("alpha must be greater than or equal to 0.")

        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise TypeError("batch_size must be an integer.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if not isinstance(self.max_epochs, int) or isinstance(self.max_epochs, bool):
            raise TypeError("max_epochs must be an integer.")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")

        if not isinstance(self.lr, (int, float)) or isinstance(self.lr, bool):
            raise TypeError("lr must be a number.")
        if self.lr <= 0.0:
            raise ValueError("lr must be greater than 0.")

        if not isinstance(self.weight_decay, (int, float)) or isinstance(
            self.weight_decay, bool
        ):
            raise TypeError("weight_decay must be a number.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be greater than or equal to 0.")

        if not isinstance(self.grad_clip, (int, float)) or isinstance(
            self.grad_clip, bool
        ):
            raise TypeError("grad_clip must be a number.")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be greater than 0.")

        if not isinstance(self.label_smoothing, (int, float)) or isinstance(
            self.label_smoothing, bool
        ):
            raise TypeError("label_smoothing must be a number.")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(
                "label_smoothing must be in the range [0.0, 0.5)."
            )

        if not isinstance(self.patience, int) or isinstance(self.patience, bool):
            raise TypeError("patience must be an integer.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")

        if not isinstance(self.min_delta, (int, float)) or isinstance(
            self.min_delta, bool
        ):
            raise TypeError("min_delta must be a number.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be greater than or equal to 0.")

        if self.device is not None:
            if not isinstance(self.device, str):
                raise TypeError("device must be a string or None.")

            try:
                device = torch.device(self.device)
            except (TypeError, RuntimeError) as error:
                raise ValueError(
                    f"Invalid device {self.device!r}."
                ) from error

            if device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError(
                    "device='cuda' was requested, but CUDA is not available."
                )

            if device.type == "mps" and not torch.backends.mps.is_available():
                raise ValueError(
                    "device='mps' was requested, but MPS is not available."
                )

        if self.random_state is not None:
            if not isinstance(self.random_state, int) or isinstance(
                self.random_state, bool
            ):
                raise TypeError("random_state must be an integer or None.")
            if self.random_state < 0:
                raise ValueError(
                    "random_state must be greater than or equal to 0."
                )

        if not isinstance(self.verbose, int) or isinstance(self.verbose, bool):
            raise TypeError("verbose must be an integer.")
        if self.verbose < 0:
            raise ValueError("verbose must be greater than or equal to 0.")

        if self.encoding not in (None, "label"):
            raise ValueError(
                "encoding must be 'label' or None."
            )

        try:
            missing_values_is_nan = bool(np.isnan(self.missing_values))
        except TypeError:
            missing_values_is_nan = False

        if not missing_values_is_nan:
            raise ValueError(
                "GAIN currently supports only missing_values=np.nan."
            )

        if not isinstance(self.add_indicator, bool):
            raise TypeError("add_indicator must be a boolean.")
        if self.add_indicator:
            raise ValueError(
                "add_indicator=True is not currently supported by GAIN."
            )

        if not isinstance(self.keep_empty_features, bool):
            raise TypeError("keep_empty_features must be a boolean.")
        if self.keep_empty_features:
            raise ValueError(
                "keep_empty_features=True is not currently supported by GAIN."
            )
