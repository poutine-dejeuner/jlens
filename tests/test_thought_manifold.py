"""Tests for jlens.thought_manifold — token representation computation,
cosine similarity eigendecomposition, and save/load roundtrip.

All tests use small synthetic matrices, no GPU required.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from jlens.thought_manifold import (
    compute_similarity_eigenvectors,
)


# ---------------------------------------------------------------------------
# Equivalence: Gram+eigh produces same results as explicit similarity SVD
# ---------------------------------------------------------------------------

class TestSVDEquivalence:
    """The Gram+eigh approach (compute_similarity_eigenvectors) must produce
    results consistent with the explicit SVD of the similarity matrix S.
    """

    def test_eigenvalues_are_squared_singular_values(self):
        """The eigenvalues of S = X̂X̂ᵀ equal σ² where σ are singular values
        of X̂.  These also equal the eigenvalues of G = X̂ᵀX̂."""
        rng = np.random.default_rng(42)
        V, d = 100, 20
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))
        X_hat = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)

        # Direct SVD of X_hat
        U_svd, S_svd, Vt_svd = torch.linalg.svd(X_hat, full_matrices=False)
        eigvals_from_svd = (S_svd ** 2).numpy()  # σ²

        # Gram+eigh
        result = compute_similarity_eigenvectors(X, center=False)
        eigvals_from_gram = result["eigenvalues"]  # top-k descending

        # All eigenvalues (ascending) should match σ² in descending order
        G = X_hat.T @ X_hat
        all_eigvals_g = torch.linalg.eigvalsh(G).numpy()[::-1]  # descending
        np.testing.assert_allclose(all_eigvals_g, eigvals_from_svd,
                                   rtol=1e-4, atol=1e-6)

        # Top-k from our function should match top-k from direct SVD
        k = result["eigenvalues"].shape[0]
        np.testing.assert_allclose(
            result["eigenvalues"],
            eigvals_from_svd[:k],
            rtol=1e-4, atol=1e-6,
        )

    def test_eigenvector_directions_match_svd(self):
        """Top-k eigenvectors from Gram+eigh should span the same subspace
        as the top-k left singular vectors from SVD."""
        rng = np.random.default_rng(123)
        V, d = 50, 10
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))
        X_hat = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)

        # Direct SVD
        U_svd, S_svd, _ = torch.linalg.svd(X_hat, full_matrices=False)
        U_svd_top = U_svd[:, :5].cpu().numpy()  # top-5 (as numpy)

        # Gram+eigh
        result = compute_similarity_eigenvectors(X, k=5, center=False)
        U_gram = result["eigenvectors"]  # (V, 5)

        # Check that each Gram eigenvector is in span of SVD singular vectors
        # by computing projection residual
        proj = U_svd_top @ (U_svd_top.T @ U_gram)  # projection onto SVD span
        residual = np.linalg.norm(U_gram - proj) / np.linalg.norm(U_gram)
        assert residual < 1e-5, f"Residual too large: {residual}"

    def test_eigenvectors_are_orthonormal(self):
        """Top-k eigenvectors should be orthonormal: UᵀU = I_k."""
        rng = np.random.default_rng(456)
        V, d = 80, 15
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=10, center=False)
        U = result["eigenvectors"]  # (V, k)
        gram_u = U.T @ U

        np.testing.assert_allclose(gram_u, np.eye(10), atol=1e-5, rtol=1e-4)

    def test_random_matrix_eigenvalues_positive(self):
        """For a random X_hat, all nonzero eigenvalues should be positive."""
        rng = np.random.default_rng(789)
        V, d = 200, 30
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=20, center=False)
        assert (result["eigenvalues"] >= 0).all()
        assert result["eigenvalues"][0] > 0  # top eigenvalue > 0


# ---------------------------------------------------------------------------
# Centering
# ---------------------------------------------------------------------------

class TestCentering:
    """Centering changes the similarity structure by removing the mean
    direction before L2-normalization.
    """

    def test_centered_vs_uncentered_different(self):
        """Centered and uncentered results should differ for non-zero-mean data."""
        rng = np.random.default_rng(42)
        V, d = 100, 10

        # Create data with a strong mean
        base = rng.normal(0, 0.01, (V, d)).astype(np.float32)
        offset = rng.normal(10, 1, (1, d)).astype(np.float32)
        X = torch.tensor(base + offset)  # strong common-mode

        result_c = compute_similarity_eigenvectors(X, k=5, center=True)
        result_nc = compute_similarity_eigenvectors(X, k=5, center=False)

        # Eigenvalues should differ
        assert not np.allclose(result_c["eigenvalues"],
                               result_nc["eigenvalues"], rtol=0.1)

    def test_centered_zero_mean_data_equivalent(self):
        """If data is already zero-mean, centering should barely change results."""
        rng = np.random.default_rng(43)
        V, d = 100, 10
        X_raw = rng.normal(0, 1, (V, d)).astype(np.float32)
        X = torch.tensor(X_raw - X_raw.mean(axis=0, keepdims=True))  # already zero-mean

        result_c = compute_similarity_eigenvectors(X, k=10, center=True)
        result_nc = compute_similarity_eigenvectors(X, k=10, center=False)

        np.testing.assert_allclose(result_c["eigenvalues"],
                                   result_nc["eigenvalues"], rtol=1e-4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling: small V, small d, k larger than rank, etc."""

    def test_k_larger_than_d(self):
        """When k > d, should return all available eigenvectors (min(k, d))."""
        rng = np.random.default_rng(42)
        V, d = 100, 5
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))

        # Request k > d; implementation caps at d implicitly
        result = compute_similarity_eigenvectors(X, k=20, center=False)
        assert result["eigenvalues"].shape[0] == d
        assert result["eigenvectors"].shape[1] == d
        assert result["singular_values"].shape[0] == d

    def test_k_larger_than_V(self):
        """When V < d, rank is limited by V."""
        d = 10
        V = 3  # fewer tokens than hidden dim
        rng = np.random.default_rng(42)
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=10, center=False)
        # Rank at most V-1 (centered) or V (uncentered).
        # But with float32 and very small V, numerical noise can create
        # spurious eigenvalues. Check that the leading V eigenvalues
        # capture >99.9% of variance.
        ev = result["singular_values"]
        leading_var = (ev[:V] ** 2 / (ev ** 2).sum()).sum()
        assert leading_var > 0.999, \
            f"Top {V} singular values only explain {100*leading_var:.2f}% variance"

    def test_all_identical_rows(self):
        """If all tokens produce identical representations, there's one
        eigenvector with eigenvalue = V."""
        V, d = 10, 4
        # All identical rows
        row = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        X = torch.tensor(np.tile(row, (V, 1)))

        result = compute_similarity_eigenvectors(X, k=5, center=True)
        # After centering, all rows are zero, so eigenvalues should be zero
        assert result["eigenvalues"][0] < 1e-10

        result_nc = compute_similarity_eigenvectors(X, k=5, center=False)
        # L2-normalized identical rows → all pairwise cosine = 1.
        # The similarity matrix is all-ones; rank 1.  Top eigenvalue ≈ V.
        assert abs(result_nc["eigenvalues"][0] - V) < 1e-4
        # Remaining eigenvalues should be ~0
        assert (result_nc["eigenvalues"][1:] < 1e-5).all(), \
            f"Non-leading eigenvalues: {result_nc['eigenvalues'][1:]}"

    def test_small_vocab(self):
        """Works for small vocabulary sizes."""
        rng = np.random.default_rng(42)
        V, d = 5, 3
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=3, center=False)
        assert len(result["eigenvalues"]) == min(3, d)
        assert result["V"] == V
        assert result["d"] == d

    def test_explained_variance_sums_to_one(self):
        """Sum of explained_variance_ratio for all components = 1.0."""
        rng = np.random.default_rng(54321)
        V, d = 200, 15
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=d, center=False)
        # Should sum to very close to 1.0
        total = result["explained_variance_ratio"].sum()
        assert abs(total - 1.0) < 1e-6, f"Sum = {total}"


# ---------------------------------------------------------------------------
# Output shapes and types
# ---------------------------------------------------------------------------

class TestOutputFormat:
    """Verify output dictionary structure and shapes."""

    def test_output_keys(self):
        """All expected keys are present."""
        rng = np.random.default_rng(42)
        X = torch.tensor(rng.normal(0, 1, (50, 10)).astype(np.float32))
        result = compute_similarity_eigenvectors(X, k=8, center=False)

        expected_keys = {
            "eigenvalues", "eigenvectors", "explained_variance_ratio",
            "singular_values", "centered", "V", "d",
        }
        assert set(result.keys()) == expected_keys

    def test_shapes_consistent(self):
        """Output array shapes are consistent with V, d, and k."""
        rng = np.random.default_rng(42)
        V, d, k = 60, 12, 8
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=k, center=False)

        assert result["eigenvalues"].shape == (k,)
        assert result["eigenvectors"].shape == (V, k)
        assert result["explained_variance_ratio"].shape == (k,)
        assert result["singular_values"].shape == (d,)
        assert result["V"] == V
        assert result["d"] == d

    def test_float32_dtype(self):
        """All array outputs should be float32 for storage efficiency."""
        rng = np.random.default_rng(42)
        X = torch.tensor(rng.normal(0, 1, (50, 10)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=5, center=False)

        assert result["eigenvalues"].dtype == np.float32
        assert result["eigenvectors"].dtype == np.float32
        assert result["explained_variance_ratio"].dtype == np.float32
        assert result["singular_values"].dtype == np.float32

    def test_eigenvalues_descending(self):
        """Top-k eigenvalues must be strictly non-increasing."""
        rng = np.random.default_rng(42)
        X = torch.tensor(rng.normal(0, 1, (100, 20)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=20, center=True)
        ev = result["eigenvalues"]
        assert (np.diff(ev) <= 1e-6).all(), f"Not descending: {ev}"


# ---------------------------------------------------------------------------
# Save/load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    """Save and load thought manifold results via HDF5."""

    def test_save_then_load_preserves_data(self):
        """Saving and loading produces identical results."""
        from jlens.storage import save_thought_manifold, load_thought_manifold

        rng = np.random.default_rng(42)
        V, d = 50, 10
        X = torch.tensor(rng.normal(0, 1, (V, d)).astype(np.float32))

        result = compute_similarity_eigenvectors(X, k=8, center=True)
        result["representations"] = X.numpy()

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            step = 0

            save_thought_manifold(results_dir, step, result)

            # Verify files exist
            step_dir = results_dir / "step0000"
            assert step_dir.exists()
            assert (step_dir / "manifold.h5").exists()

            # Load and compare
            loaded = load_thought_manifold(results_dir, step)

            np.testing.assert_allclose(
                loaded["eigenvalues"], result["eigenvalues"], rtol=1e-5)
            np.testing.assert_allclose(
                loaded["eigenvectors"], result["eigenvectors"], rtol=1e-5)
            np.testing.assert_allclose(
                loaded["explained_variance_ratio"],
                result["explained_variance_ratio"], rtol=1e-5)
            np.testing.assert_allclose(
                loaded["singular_values"], result["singular_values"], rtol=1e-5)
            assert loaded["centered"] == result["centered"]
            assert loaded["V"] == result["V"]
            assert loaded["d"] == result["d"]

    def test_load_nonexistent_returns_none(self):
        """Loading a non-existent checkpoint returns None."""
        from jlens.storage import load_thought_manifold

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            result = load_thought_manifold(results_dir, step=5)
            assert result is None


# ---------------------------------------------------------------------------
# Contextual pipeline (Option B) — unit tests with mocked Wikipedia
# ---------------------------------------------------------------------------

class TestContextualPipeline:
    """Tests for compute_contextual_representations and
    thought_manifold_contextual_pipeline."""

    def test_empty_token_ids_returns_zero_tensor(self):
        """With empty token_ids, returns zero-shaped tensor without
        touching Wikipedia."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2", dtype=torch.float32)

        from jlens.thought_manifold import compute_contextual_representations
        X = compute_contextual_representations(
            model, token_ids=[], n_contexts=1)

        assert X.shape == (0, model.config.hidden_size)
