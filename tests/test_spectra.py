"""Tests for jlens.spectra — eigenvalue, MP fit, power-law tail, None-tolerance."""

import numpy as np
from scipy.integrate import trapezoid
import pytest

from jlens.spectra import (
    analyze_checkpoint,
    effective_rank,
    eigen_decompose,
    fit_mp_bulk,
    fit_power_law_tail,
    marchenko_pastur_pdf,
)


class TestEigenDecompose:
    def test_identity(self):
        """Eigenvalues of I are all 1."""
        gram = np.eye(16)
        ev = eigen_decompose(gram)
        np.testing.assert_allclose(ev, np.ones(16), rtol=1e-5)

    def test_diagonal(self):
        """Eigenvalues of diag matrix are the diagonal entries."""
        diag_vals = np.array([4.0, 3.0, 2.0, 1.0])
        gram = np.diag(diag_vals)
        ev = eigen_decompose(gram)
        np.testing.assert_allclose(ev, diag_vals, rtol=1e-5)

    def test_random_psd(self):
        """Eigenvalues all ≥ 0 for PSD matrix."""
        a = np.random.randn(32, 32) * 0.1
        gram = a.T @ a  # PSD
        ev = eigen_decompose(gram)
        assert np.all(ev >= -1e-10)

    def test_sorted_descending(self):
        """Eigenvalues are sorted descending."""
        a = np.random.randn(64, 64)
        gram = a.T @ a
        ev = eigen_decompose(gram)
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i + 1] - 1e-10


class TestEffectiveRank:
    def test_identity(self):
        """Effective rank of I_d is d."""
        d = 32
        ev = np.ones(d)
        assert effective_rank(ev) == pytest.approx(d, rel=1e-3)

    def test_rank_one(self):
        """Effective rank of rank-1 matrix is 1."""
        ev = np.array([10.0] + [0.0] * 15)
        assert effective_rank(ev) == pytest.approx(1.0, rel=1e-2)

    def test_all_zeros(self):
        """Effective rank of all zeros is 0."""
        assert effective_rank(np.zeros(8)) == 0.0


class TestPowerLawTail:
    def test_pure_power_law(self):
        """Synthetic power-law eigenvalues → α ≈ true α."""
        rng = np.random.default_rng(42)
        alpha_true = 0.45
        # Generate from power-law distribution
        ev = rng.pareto(alpha_true, size=1000)
        ev = ev[ev < 100]  # truncate extreme values
        result = fit_power_law_tail(ev, tail_fraction=0.5)
        # Not super precise but should be in ballpark
        assert 0.1 < result["alpha"] < 1.5

    def test_few_points(self):
        """Works even with few eigenvalues."""
        ev = np.linspace(0.01, 1.0, 20)
        result = fit_power_law_tail(ev)
        assert "alpha" in result

    def test_zeros_handled(self):
        """Zeros in eigenvalues don't crash."""
        ev = np.array([0.0] * 5 + [0.1, 0.2, 0.3, 0.4, 0.5])
        result = fit_power_law_tail(ev)
        assert "alpha" in result  # no crash


class TestMPBulk:
    def test_wishart_eigenvalues(self):
        """MP fit on Wishart matrix gives q ≈ n/d."""
        # Generate Wishart: X ∈ R^{n×d}, Gram = X⊤X / n
        n, d = 500, 100
        X = np.random.randn(n, d)
        gram = X.T @ X / n
        ev = eigen_decompose(gram)
        result = fit_mp_bulk(ev)

        # q ≈ d/n = 0.2
        assert 0.05 < result["q_eff"] < 0.5
        # sigma2 ≈ 1 (since X is N(0,1))
        assert 0.5 < result["sigma2"] < 2.0

    def test_no_spikes_in_pure_noise(self):
        """Pure Wishart → no spikes."""
        n, d = 500, 100
        X = np.random.randn(n, d)
        gram = X.T @ X / n
        ev = eigen_decompose(gram)
        result = fit_mp_bulk(ev)
        # Without signal, spike count should be small
        assert result["n_spikes"] < 20


class TestMPPDF:
    def test_normalization_approximate(self):
        """Marchenko-Pastur PDF integrates approximately to 1."""
        sigma2, q = 1.0, 0.5
        x = np.linspace(0.01, 5, 1000)
        pdf = marchenko_pastur_pdf(x, sigma2, q)
        integral = trapezoid(pdf, x)
        assert 0.8 < integral < 1.2  # approximate

    def test_edge_cases(self):
        """Edge cases don't crash."""
        # q=0, sigma2=0, negative values
        marchenko_pastur_pdf(np.array([1.0]), 0.0, 1.0)
        marchenko_pastur_pdf(np.array([1.0]), 1.0, 0.0)
        marchenko_pastur_pdf(np.array([1.0]), 1.0, -1.0)


class TestAnalyzeCheckpoint:
    """End-to-end: stats → spectral analysis, including None tolerance."""

    def make_stats(self, n_layers, d_model, population_mask=None):
        """Minimal stats dict for analyze_checkpoint."""
        if population_mask is None:
            population_mask = [True] * n_layers

        jbar = []
        jtj_gram = []
        coherence = []
        a_coeff = []
        r_norm = []
        tr_jtj_mean = []

        rng = np.random.default_rng(123)
        for i in range(n_layers):
            if population_mask[i]:
                a = rng.normal(0, 0.1, (d_model, d_model)) + np.eye(d_model) * 0.7
                jbar.append(a)
                jtj_gram.append(a.T @ a)
                coherence.append(float(np.trace(a.T @ a) / np.sum(a**2)))
                a_coeff.append(float(np.trace(a) / d_model))
                r_norm.append(0.1)
                tr_jtj_mean.append(float(np.sum(a**2)))
            else:
                jbar.append(None)
                jtj_gram.append(None)
                coherence.append(None)
                a_coeff.append(None)
                r_norm.append(None)
                tr_jtj_mean.append(None)

        return {
            "n": 50,
            "jbar": jbar,
            "jtj_gram": jtj_gram,
            "coherence": coherence,
            "a_coeff": a_coeff,
            "r_norm": r_norm,
            "tr_jtj_mean": tr_jtj_mean,
        }

    def test_all_layers_populated(self):
        """Full analysis with all layers populated."""
        stats = self.make_stats(5, 64)
        result = analyze_checkpoint(stats)

        assert result["n_layers"] == 5
        assert result["d_model"] == 64
        assert len(result["eigenvalues"]) == 5
        assert len(result["q_eff"]) == 5
        assert len(result["n_spikes"]) == 5
        assert len(result["power_law_alpha"]) == 5
        assert len(result["effective_rank"]) == 5

        # All values should be finite
        for i in range(5):
            assert not np.isnan(result["q_eff"][i])
            assert not np.isnan(result["effective_rank"][i])

    def test_missing_last_layer(self):
        """Last layer is None — analyze_checkpoint should not crash (THE BUG)."""
        n_layers = 12
        mask = [True] * 11 + [False]
        stats = self.make_stats(n_layers, 768, population_mask=mask)

        # THIS MUST NOT CRASH
        result = analyze_checkpoint(stats)

        assert result["n_layers"] == 12

        # Layers 0-10: valid
        for i in range(11):
            assert len(result["eigenvalues"][i]) == 768, f"layer {i} eigenvalues wrong shape"
            assert not np.isnan(result["effective_rank"][i])

        # Layer 11: NaN placeholders
        assert result["eigenvalues"][11] is not None
        assert np.isnan(result["effective_rank"][11])

    def test_sparse_layers(self):
        """Only some layers populated — all get valid or NaN."""
        n_layers = 12
        mask = [i in (0, 3, 5, 7, 9, 10) for i in range(n_layers)]
        stats = self.make_stats(n_layers, 256, population_mask=mask)

        result = analyze_checkpoint(stats)

        for i in range(n_layers):
            if mask[i]:
                assert len(result["eigenvalues"][i]) == 256
                assert not np.isnan(result["effective_rank"][i])
            else:
                assert np.isnan(result["effective_rank"][i])

    def test_effective_rank_monotonic_with_coh(self):
        """Higher coherence (more aligned) → lower effective rank."""
        d = 128
        # Low coherence: random-ish J
        stats_low = self.make_stats(1, d)
        stats_low["jtj_gram"][0] = np.eye(d)  # perfectly isotropic
        stats_low["coherence"][0] = 0.05
        result_low = analyze_checkpoint(stats_low)

        # High coherence: strongly aligned J
        a = np.ones((d, d)) * 0.1 + np.eye(d) * 0.9
        stats_high = self.make_stats(1, d)
        stats_high["jtj_gram"][0] = a.T @ a
        stats_high["coherence"][0] = 0.95
        result_high = analyze_checkpoint(stats_high)

        # Isotropic J should have higher effective rank
        assert result_low["effective_rank"][0] > result_high["effective_rank"][0]
