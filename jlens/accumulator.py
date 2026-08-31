"""Online accumulation of Jacobian statistics.

Uses Welford-style online updates to compute:
  J̄_ℓ = mean of per-prompt Jacobians
  Σ_ℓ  = sum of squared deviations (for coherence)
  E[tr(J⊤J)] = mean of squared Frobenius norms
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

    def get_mean(self, layer_idx: int) -> torch.Tensor:
        """Get J̄_ℓ = mean Jacobian for a layer."""
        if self.sum_j[layer_idx] is None:
            raise ValueError(f"No data for layer {layer_idx}")
        return self.sum_j[layer_idx] / self.n

    def get_gram(self, layer_idx: int) -> torch.Tensor:
        """Get J̄⊤J̄ for a layer."""
        jbar = self.get_mean(layer_idx)
        return (jbar.T @ jbar).to(self.dtype)

    def get_coherence(self, layer_idx: int) -> float:
        """Get κ_ℓ = tr(J̄⊤J̄) / E[tr(J⊤J)]."""
        jbar = self.get_mean(layer_idx)
        tr_jbart_jbar = torch.sum(jbar**2).item()
        tr_mean = self.sum_tr_jtj[layer_idx] / self.n
        if tr_mean == 0:
            return 0.0
        return tr_jbart_jbar / tr_mean

    def get_a_coeff(self, layer_idx: int) -> float:
        """Get a_ℓ = tr(J̄)/d."""
        jbar = self.get_mean(layer_idx)
        return torch.trace(jbar).item() / self.d_model

    def get_r_norm(self, layer_idx: int) -> float:
        """Get ‖R‖_F / √d where R = J̄ - a_ℓ·I."""
        jbar = self.get_mean(layer_idx)
        a = torch.trace(jbar).item() / self.d_model
        eye = torch.eye(self.d_model, device=jbar.device, dtype=jbar.dtype)
        r = jbar - a * eye
        return torch.sqrt(torch.sum(r**2)).item() / (self.d_model**0.5)

    def get_tr_jtj_mean(self, layer_idx: int) -> float:
        """Get E[tr(J⊤J)] for a layer."""
        return self.sum_tr_jtj[layer_idx] / self.n

    def get_all_stats(self) -> dict:
        """Get all accumulated statistics for all layers."""
        return {
            "n": self.n,
            "jbar": [self.get_mean(i).numpy() for i in range(self.n_layers)],
            "jtj_gram": [self.get_gram(i).numpy() for i in range(self.n_layers)],
            "coherence": [self.get_coherence(i) for i in range(self.n_layers)],
            "a_coeff": [self.get_a_coeff(i) for i in range(self.n_layers)],
            "r_norm": [self.get_r_norm(i) for i in range(self.n_layers)],
            "tr_jtj_mean": [self.get_tr_jtj_mean(i) for i in range(self.n_layers)],
        }
