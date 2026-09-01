"""Tests for jlens.accumulator — the module that had the `.numpy() on None` bug."""

import numpy as np
import torch
import pytest

from jlens.accumulator import JacobianAccumulator


class TestJacobianAccumulator:
    """Test the JacobianAccumulator, especially None-tolerant behavior."""

    def make_jac(self, d=8, val=0.5):
        """Make a d×d Jacobian with known trace = val * d."""
        j = torch.eye(d) * val + torch.randn(d, d) * 0.01
        return j

    def test_basic_accumulation(self):
        """Single update, single layer → stats are consistent."""
        d = 8
        acc = JacobianAccumulator(n_layers=3, d_model=d)
        j0 = self.make_jac(d, val=0.5)
        acc.update({0: j0})

        assert acc.n == 1
        jbar = acc.get_mean(0)
        assert jbar is not None
        np.testing.assert_allclose(jbar.numpy(), j0.numpy(), rtol=1e-4)

        coh = acc.get_coherence(0)
        # single sample → J̄ = J → coherence = tr(J̄⊤J̄)/tr(J⊤J) = 1.0
        assert coh is not None
        assert abs(coh - 1.0) < 0.05

    def test_multiple_updates(self):
        """Multiple updates compute correct mean."""
        d = 4
        acc = JacobianAccumulator(n_layers=1, d_model=d)
        j1 = torch.ones(d, d)
        j2 = torch.ones(d, d) * 3
        acc.update({0: j1})
        acc.update({0: j2})

        jbar = acc.get_mean(0)
        assert jbar is not None
        expected = (j1 + j2) / 2.0
        np.testing.assert_allclose(jbar.numpy(), expected.numpy(), rtol=1e-4)
        assert acc.n == 2

    def test_missing_layer_returns_none(self):
        """Layer without data → all getters return None."""
        acc = JacobianAccumulator(n_layers=5, d_model=16)
        acc.update({0: torch.eye(16), 2: torch.eye(16)})

        # Layer 0 has data
        assert acc.get_mean(0) is not None
        # Layer 1 has no data
        assert acc.get_mean(1) is None
        assert acc.get_gram(1) is None
        assert acc.get_coherence(1) is None
        assert acc.get_a_coeff(1) is None
        assert acc.get_r_norm(1) is None
        assert acc.get_tr_jtj_mean(1) is None

    def test_get_all_stats_mixed_layers(self):
        """get_all_stats() works with mix of data and None layers (THE BUG)."""
        d = 8
        acc = JacobianAccumulator(n_layers=12, d_model=d)  # 12 like Pythia
        # Only layers 0-10 have data (like real pipeline)
        for i in range(11):
            acc.update({i: self.make_jac(d)})
        # Layer 11 never gets data

        stats = acc.get_all_stats()

        assert stats["n"] == 11  # one update per layer
        assert len(stats["jbar"]) == 12
        assert len(stats["jtj_gram"]) == 12
        assert len(stats["coherence"]) == 12

        # Layers 0-10: real arrays
        for i in range(11):
            assert isinstance(stats["jbar"][i], np.ndarray), f"layer {i} jbar should be array"
            assert isinstance(stats["jtj_gram"][i], np.ndarray), f"layer {i} jtj_gram should be array"
            assert isinstance(stats["coherence"][i], float), f"layer {i} coherence should be float"
            assert not np.isnan(stats["coherence"][i]), f"layer {i} coherence should not be NaN"

        # Layer 11: None (no data)
        assert stats["jbar"][11] is None
        assert stats["jtj_gram"][11] is None
        assert stats["coherence"][11] is None
        assert stats["a_coeff"][11] is None
        assert stats["r_norm"][11] is None
        assert stats["tr_jtj_mean"][11] is None

    def test_coherence_range(self):
        """Coherence should be in [0, 1]."""
        d = 16
        acc = JacobianAccumulator(n_layers=1, d_model=d)
        for _ in range(10):
            acc.update({0: torch.randn(d, d) * 0.1 + torch.eye(d) * 0.7})

        coh = acc.get_coherence(0)
        assert coh is not None
        assert 0.0 <= coh <= 1.0, f"coherence {coh} out of [0,1]"

    def test_tr_jtj_mean_positive(self):
        """E[tr(J⊤J)] should be positive for non-zero Jacobians."""
        d = 8
        acc = JacobianAccumulator(n_layers=1, d_model=d)
        acc.update({0: torch.eye(d) * 2.0})

        tr_m = acc.get_tr_jtj_mean(0)
        assert tr_m is not None
        assert tr_m > 0

    def test_a_coeff_identity(self):
        """For identity Jacobian, a_coeff = 1.0."""
        d = 16
        acc = JacobianAccumulator(n_layers=1, d_model=d)
        acc.update({0: torch.eye(d)})

        a = acc.get_a_coeff(0)
        assert a is not None
        assert abs(a - 1.0) < 1e-4

    def test_r_norm_identity(self):
        """For identity Jacobian, R = 0 → r_norm = 0."""
        d = 16
        acc = JacobianAccumulator(n_layers=1, d_model=d)
        acc.update({0: torch.eye(d)})

        r = acc.get_r_norm(0)
        assert r is not None
        assert abs(r) < 1e-4

    def test_multiple_prompts_mean_convergence(self):
        """Mean of many Jacobians should be close to the matrix mean."""
        d = 8
        acc = JacobianAccumulator(n_layers=1, d_model=d)
        n = 100
        matrices = []
        for _ in range(n):
            m = torch.randn(d, d) * 0.1 + torch.eye(d)
            matrices.append(m)
            acc.update({0: m})

        expected = torch.stack(matrices).mean(dim=0)
        jbar = acc.get_mean(0)
        assert jbar is not None
        np.testing.assert_allclose(jbar.numpy(), expected.numpy(), rtol=1e-3)
