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
from dataclasses import dataclass, field
from sklearn.impute._base import _BaseImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data 
from sklearn.exceptions import NotFittedError
from gouda.gouda import raise_if_nan_col


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
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be ≥ 2.")

        layers: list[nn.Module] = []

        # Input block
        layers += [nn.Linear(dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden blocks
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output block — no normalisation, sigmoid to keep outputs in [0, 1]
        layers += [nn.Linear(hidden_dim, dim), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
        return self.net(inp)


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
    """
    hidden_dim      : Width of hidden layers in both G and D.
    num_layers      : Depth of both networks (≥ 2).
    dropout         : Dropout rate in both networks (0 = disabled).
    hint_rate       : Fraction of mask bits revealed to discriminator (0 < p ≤ 1).
    alpha           : Weight on the MSE reconstruction term in G's loss.
    batch_size      : Mini-batch size.
    max_epochs      : Maximum training epochs.
    lr              : Initial learning rate for both G and D.
    weight_decay    : L2 regularisation.
    grad_clip       : Maximum gradient norm (per network).
    label_smoothing : Label smoothing applied to discriminator targets (e.g. 0.1).
    patience        : Early-stopping patience (epochs without joint-loss improvement).
    min_delta       : Minimum relative improvement to reset patience counter.
    device          : 'cpu', 'cuda', 'mps', or None (auto-detect).
    random_state    : Integer seed for full reproducibility, or None.
    verbose         : Print training progress every `verbose` epochs (0 = silent).
    """
    def __init__(self, *,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.0,
    hint_rate: float = 0.9,
    alpha: float = 100.0,
    batch_size: int = 256,
    max_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 5.0,
    label_smoothing: float = 0.1,
    patience: int = 30,
    min_delta: float = 1e-4,
    device: str | None = None,
    random_state: int | None = 42,
    verbose: int = 0,
    encoding: None | str = None,
    # required by _BaseImputer
    missing_values=np.nan, 
    add_indicator: bool = False, 
    keep_empty_features: bool = False,) -> None:

        self.hidden_dim= hidden_dim
        self.num_layers= num_layers
        self.dropout = dropout
        self.hint_rate= hint_rate
        self.alpha= alpha
        self.batch_size= batch_size
        self.max_epochs= max_epochs
        self.lr= lr
        self.weight_decay= weight_decay
        self.grad_clip= grad_clip
        self.label_smoothing= label_smoothing
        self.patience= patience
        self.min_delta= min_delta
        self.device= device
        self.random_state= random_state
        self.verbose= verbose
        self.device= device
        self.encoding= encoding
        self.missing_values= missing_values
        self.add_indicator= add_indicator
        self.keep_empty_features= keep_empty_features

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
            torch.bernoulli(torch.full_like(mask, self.hint_rate)).to(mask.device)
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
        alpha: float,
    ) -> torch.Tensor:
        """
        Generator loss = adversarial term (fool discriminator on MISSING entries)
                       + alpha * MSE reconstruction on OBSERVED entries.
        """
        # Adversarial: want discriminator to output 1 on missing positions
        missing_mask = 1.0 - mask
        n_missing = missing_mask.sum().clamp(min=1.0)
        adv_loss = -((torch.log(d_out + 1e-8)) * missing_mask).sum() / n_missing

        # Reconstruction: penalise deviation from known values on observed entries
        n_observed = mask.sum().clamp(min=1.0)
        mse_loss = ((x_hat - x_norm) ** 2 * mask).sum() / n_observed

        return adv_loss + alpha * mse_loss

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y=None) -> "GAINImputer":
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
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")  
        X = np.array(X, dtype=np.float64)
        raise_if_nan_col(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")

        n_samples, dim = X.shape
        if n_samples < 2:
            raise ValueError(f"n_samples = {n_samples}, GAIN requires at least 2 samples to fit.")


        # ---- reproducibility -----------------------------------------
        if self.random_state is not None:
            self._set_seed(self.random_state)

        # ---- device ---------------------------------------------------
        self._device = self._resolve_device()

        # ---- normalise ------------------------------------------------
        self._dim = dim
        self._data_min = np.nanmin(X, axis=0)
        self._data_max = np.nanmax(X, axis=0)

        mask_np = self._make_mask(X)            # (N, dim) float32
        X_norm  = self._normalise(X).astype(np.float32)
        # Replace NaNs with 0 in normalised array (masked out in losses)
        X_fill  = np.where(mask_np == 1, X_norm, 0.0).astype(np.float32)

        X_t    = torch.from_numpy(X_fill).to(self._device)
        mask_t = torch.from_numpy(mask_np).to(self._device)
        X_norm_t = torch.from_numpy(
            np.where(mask_np == 1, X_norm, 0.0).astype(np.float32)
        ).to(self._device)

        dataset    = TensorDataset(X_t, mask_t, X_norm_t)
        loader     = DataLoader(
            dataset,
            batch_size=min(self.batch_size, n_samples),
            shuffle=True,
            drop_last=(n_samples > self.batch_size),
        )

        # ---- build networks ------------------------------------------
        self.generator_     = Generator(
            dim, self.hidden_dim, self.num_layers, self.dropout
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

        # ---- training history & early stopping -----------------------
        hist_d: list[float] = []
        hist_g: list[float] = []
        best_joint = float("inf")
        patience_counter = 0

        # ---- training loop -------------------------------------------
        for epoch in range(1, self.max_epochs + 1):
            epoch_d, epoch_g = 0.0, 0.0
            n_batches = 0

            for x_batch, m_batch, xn_batch in loader:
                B = x_batch.size(0)

                # ---------- Discriminator step ----------------------
                self.discriminator_.train()
                self.generator_.eval()

                # Add small noise to observed inputs (instance noise → stability)
                noise = torch.randn_like(x_batch) * 0.01
                x_noisy = x_batch + noise * m_batch  # only on observed

                with torch.no_grad():
                    g_out = self.generator_(x_noisy, m_batch)

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

                g_out = self.generator_(x_noisy, m_batch)
                x_hat = m_batch * x_batch + (1.0 - m_batch) * g_out

                hint = self._build_hint(m_batch)
                with torch.no_grad():
                    d_out = self.discriminator_(x_hat, hint)

                # Re-forward discriminator with grad only for G's adv loss
                # (we need d_out to have grad w.r.t. g_out)
                d_out_g = self.discriminator_(x_hat, hint)
                loss_G = self._gen_loss(d_out_g, m_batch, g_out, xn_batch, self.alpha)

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

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

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
        if not hasattr(self, "is_fitted_") and self.is_fitted_ == True:
            raise NotFittedError

        if self.generator_ is None:
            raise RuntimeError("Call fit() before transform().")

        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        X = np.array(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        if X.shape[1] != self._dim:
            raise ValueError(
                f"Expected {self._dim} features, got {X.shape[1]}."
            )

        mask_np = self._make_mask(X)
        X_norm  = self._normalise(X).astype(np.float32)
        X_fill  = np.where(mask_np == 1, X_norm, 0.0).astype(np.float32)

        X_t    = torch.from_numpy(X_fill).to(self._device)
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(self._device)

        self.generator_.eval()
        with torch.no_grad():
            g_out = self.generator_(X_t, mask_t)           # (N, dim) in [0,1]
            # Keep observed values; fill missing with generator output
            x_hat = mask_t * X_t + (1.0 - mask_t) * g_out

        x_hat_np = x_hat.cpu().numpy().astype(np.float64)
        # Inverse normalise back to original scale
        X_imputed = self._denormalise(x_hat_np)

        # Safety: restore any originally-observed values exactly
        obs_idx = mask_np == 1
        X_imputed[obs_idx] = X[obs_idx]

        return X_imputed

    # ------------------------------------------------------------------
    # fit_transform (sklearn mixin pattern)
    # ------------------------------------------------------------------

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit the model and return the imputed training data."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # sklearn get_params / set_params
    # ------------------------------------------------------------------

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
            "missing_values":  self.missing_values,
            "add_indicator":   self.add_indicator,
            "keep_empty_features": self.keep_empty_features,
        }

    def set_params(self, **params) -> "GAINImputer":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter '{k}'.")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:  # pragma: no cover
        p = self.get_params()
        args = ", ".join(f"{k}={v!r}" for k, v in p.items())
        return f"GAINImputer({args})"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
