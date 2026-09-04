"""Online accumulation of Jacobian statistics.

Uses Welford-style online updates to compute:
  J̄_ℓ = mean of per-prompt Jacobians
  Σ_ℓ  = E[J⊤J] - J̄⊤J̄  (fluctuation covariance)
  E[tr(J⊤J)] = mean of squared Frobenius norms
  κ_ℓ  = coherence (coherent/total energy ratio)
  FCR_ℓ = tr(Σ_ℓ) / ‖J̄_ℓ‖²_F  (fluctuation-to-coherent ratio)
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class JacobianAccumulator:
    """Online accumulator for Jacobian mean and covariance per layer."""

    def __init__(self, n_layers: int, d_model: int, dtype: torch.dtype = torch.float32):
        self.n_layers = n_layers
        self.d_model = d_model
        self.dtype = dtype
        self.device: Optional[torch.device] = None

        # Running sums
        self.n = 0  # prompt count

        # Per-layer accumulators (on CPU to save GPU memory)
        self.sum_j = [None] * n_layers  # running sum of J
        self.sum_jtj = [None] * n_layers  # running sum of J⊤J
        self.sum_tr_jtj = [0.0] * n_layers  # running sum of tr(J⊤J)

    def update(self, jacobians: dict[int, torch.Tensor]) -> None:
        """Update accumulators with per-prompt Jacobians.

        Args:
            jacobians: Dict mapping layer_idx → D×D Jacobian matrix
        """
        self.n += 1

        for layer_idx, jac in jacobians.items():
            # Move to CPU for accumulation (GPU memory is precious)
            j_cpu = jac.detach().cpu().to(self.dtype)

            if self.device is None:
                self.device = jac.device

            # Welford-style mean update
            if self.sum_j[layer_idx] is None:
                self.sum_j[layer_idx] = j_cpu.clone()
                self.sum_jtj[layer_idx] = (j_cpu.T @ j_cpu).to(self.dtype)
            else:
                self.sum_j[layer_idx] += j_cpu
                self.sum_jtj[layer_idx] += (j_cpu.T @ j_cpu).to(self.dtype)

            # Frobenius norm squared: tr(J⊤J)
            self.sum_tr_jtj[layer_idx] += torch.sum(j_cpu**2).item()

    def get_mean(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get J̄_ℓ = mean Jacobian for a layer, or None if no data."""
        if self.sum_j[layer_idx] is None:
            return None
        return self.sum_j[layer_idx] / self.n

    def get_gram(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get J̄⊤J̄ for a layer, or None if no data."""
        jbar = self.get_mean(layer_idx)
        if jbar is None:
            return None
        return (jbar.T @ jbar).to(self.dtype)

    def get_coherence(self, layer_idx: int) -> Optional[float]:
        """Get κ_ℓ = tr(J̄⊤J̄) / E[tr(J⊤J)], or None if no data."""
        jbar = self.get_mean(layer_idx)
        if jbar is None:
            return None
        tr_jbart_jbar = torch.sum(jbar**2).item()
        tr_mean = self.sum_tr_jtj[layer_idx] / self.n
        if tr_mean == 0:
            return 0.0
        return tr_jbart_jbar / tr_mean

    def get_a_coeff(self, layer_idx: int) -> Optional[float]:
        """Get a_ℓ = tr(J̄)/d, or None if no data."""
        jbar = self.get_mean(layer_idx)
        if jbar is None:
            return None
        return torch.trace(jbar).item() / self.d_model

    def get_r_norm(self, layer_idx: int) -> Optional[float]:
        """Get ‖R‖_F / √d where R = J̄ - a_ℓ·I, or None if no data."""
        jbar = self.get_mean(layer_idx)
        if jbar is None:
            return None
        a = torch.trace(jbar).item() / self.d_model
        eye = torch.eye(self.d_model, device=jbar.device, dtype=jbar.dtype)
        r = jbar - a * eye
        return torch.sqrt(torch.sum(r**2)).item() / (self.d_model**0.5)

    def get_tr_jtj_mean(self, layer_idx: int) -> Optional[float]:
        """Get E[tr(J⊤J)], or None if no data."""
        if self.sum_j[layer_idx] is None:
            return None
        return self.sum_tr_jtj[layer_idx] / self.n

    def get_sigma(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get Σ_ℓ = E[J⊤J] - J̄⊤J̄, the fluctuation covariance.

        Σ_ℓ is the second central moment of the per-prompt Jacobian
        distribution. Its spectral properties (eigenvalues, eigenvectors,
        effective rank) reveal the structure of prompt-dependent transport.

        Returns None if no data.
        """
        jbar = self.get_mean(layer_idx)
        if jbar is None:
            return None
        e_jtj = self.sum_jtj[layer_idx] / self.n
        jbar_gram = jbar.T @ jbar
        sigma = e_jtj - jbar_gram
        # Clamp tiny negative eigenvalues from numerical error
        sigma = (sigma + sigma.T) / 2  # symmetrize
        return sigma.to(self.dtype)

    def get_sigma_gram(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get Σ_ℓ as a numpy array (for storage). Returns None if no data."""
        sigma = self.get_sigma(layer_idx)
        if sigma is None:
            return None
        return sigma.numpy()

    def get_fcr(self, layer_idx: int) -> Optional[float]:
        """Get fluctuation-to-coherent ratio: tr(Σ_ℓ) / ‖J̄_ℓ‖²_F.

        FCR = (1 - κ_ℓ) / κ_ℓ measures how much more fluctuation energy
        exists relative to coherent energy.  Unlike κ_ℓ, FCR does not
        saturate at 1 for near-identity transport, making it a cleaner
        order parameter across initialization and convergence.

        Returns None if no data, inf if ‖J̄‖²_F = 0.
        """
        jbar = self.get_mean(layer_idx)
        if jbar is None:
            return None
        sigma = self.get_sigma(layer_idx)
        if sigma is None:
            return None
        tr_sigma = torch.trace(sigma).item()
        tr_jbar_gram = torch.sum(jbar ** 2).item()
        if tr_jbar_gram == 0:
            return float('inf')
        return tr_sigma / tr_jbar_gram

    def get_all_stats(self) -> dict:
        """Get all accumulated statistics for layers that have data."""
        jbar_list = []
        jtj_gram_list = []
        sigma_gram_list = []
        for i in range(self.n_layers):
            m = self.get_mean(i)
            g = self.get_gram(i)
            s = self.get_sigma_gram(i)
            jbar_list.append(m.numpy() if m is not None else None)
            jtj_gram_list.append(g.numpy() if g is not None else None)
            sigma_gram_list.append(s if s is not None else None)
        return {
            "n": self.n,
            "jbar": jbar_list,
            "jtj_gram": jtj_gram_list,
            "sigma_gram": sigma_gram_list,
            "coherence": [self.get_coherence(i) for i in range(self.n_layers)],
            "a_coeff": [self.get_a_coeff(i) for i in range(self.n_layers)],
            "r_norm": [self.get_r_norm(i) for i in range(self.n_layers)],
            "tr_jtj_mean": [self.get_tr_jtj_mean(i) for i in range(self.n_layers)],
            "fcr": [self.get_fcr(i) for i in range(self.n_layers)],
        }
