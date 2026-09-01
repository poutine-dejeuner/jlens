"""Tests for jlens.storage — save/load roundtrip with None-tolerant layers."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from jlens.storage import (
    is_checkpoint_done,
    load_checkpoint_result,
    save_checkpoint_result,
)


def make_stats(n_layers, d_model, population_mask=None):
    """Build a stats dict like JacobianAccumulator.get_all_stats().

    population_mask: list of bools, True = has data, False = None (no data).
                     Default: all layers have data.
    """
    if population_mask is None:
        population_mask = [True] * n_layers

    jbar = []
    jtj_gram = []
    coherence = []
    a_coeff = []
    r_norm = []
    tr_jtj_mean = []

    for i in range(n_layers):
        if population_mask[i]:
            m = np.random.randn(d_model, d_model) * 0.1 + np.eye(d_model)
            jbar.append(m)
            jtj_gram.append(m.T @ m)
            coherence.append(float(np.trace(m.T @ m) / np.sum(m**2)))
            a_coeff.append(float(np.trace(m) / d_model))
            r_norm.append(0.1)
            tr_jtj_mean.append(float(np.sum(m**2)))
        else:
            jbar.append(None)
            jtj_gram.append(None)
            coherence.append(None)
            a_coeff.append(None)
            r_norm.append(None)
            tr_jtj_mean.append(None)

    return {
        "n": 10,
        "jbar": jbar,
        "jtj_gram": jtj_gram,
        "coherence": coherence,
        "a_coeff": a_coeff,
        "r_norm": r_norm,
        "tr_jtj_mean": tr_jtj_mean,
    }


def make_spectral(n_layers, d_model, population_mask=None):
    """Build a spectral dict like analyze_checkpoint() output."""
    if population_mask is None:
        population_mask = [True] * n_layers

    eigenvalues = []
    eff_rank = []
    q_eff = []
    n_spikes = []
    power_law_alpha = []
    mp_sigma2 = []

    for i in range(n_layers):
        if population_mask[i]:
            eigenvalues.append(np.sort(np.random.rand(d_model))[::-1])
            eff_rank.append(float(d_model * 0.5))
            q_eff.append(0.8)
            n_spikes.append(3)
            power_law_alpha.append(0.45)
            mp_sigma2.append(1.2)
        else:
            eigenvalues.append(np.array([np.nan]))
            eff_rank.append(np.nan)
            q_eff.append(np.nan)
            n_spikes.append(np.nan)
            power_law_alpha.append(np.nan)
            mp_sigma2.append(np.nan)

    return {
        "n_layers": n_layers,
        "d_model": d_model,
        "n_prompts": 10,
        "eigenvalues": eigenvalues,
        "coherence": [0.7 + i * 0.02 for i in range(n_layers)],
        "a_coeff": [0.5 for _ in range(n_layers)],
        "r_norm": [0.1 for _ in range(n_layers)],
        "tr_jtj_mean": [100.0 for _ in range(n_layers)],
        "effective_rank": eff_rank,
        "q_eff": q_eff,
        "n_spikes": n_spikes,
        "power_law_alpha": power_law_alpha,
        "mp_sigma2": mp_sigma2,
    }


class TestSaveLoadRoundtrip:
    """Save → load → compare: the full roundtrip."""

    def test_roundtrip_all_layers_populated(self):
        """All layers have data — save and load should match."""
        n_layers, d_model = 5, 16
        stats = make_stats(n_layers, d_model)
        spectral = make_spectral(n_layers, d_model)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(results_dir, step=42, stats=stats, spectral=spectral)

            assert is_checkpoint_done(results_dir, 42)
            assert is_checkpoint_done(str(results_dir), 42)  # string Path test

            loaded = load_checkpoint_result(results_dir, 42)

            # Check stats
            ls = loaded["stats"]
            assert ls["n"] == stats["n"]
            for i in range(n_layers):
                np.testing.assert_allclose(ls["jbar"][i], stats["jbar"][i], rtol=1e-5)
                np.testing.assert_allclose(ls["jtj_gram"][i], stats["jtj_gram"][i], rtol=1e-5)
                assert ls["coherence"][i] == pytest.approx(stats["coherence"][i])
                assert ls["a_coeff"][i] == pytest.approx(stats["a_coeff"][i])
                assert ls["r_norm"][i] == pytest.approx(stats["r_norm"][i])
                assert ls["tr_jtj_mean"][i] == pytest.approx(stats["tr_jtj_mean"][i])

            # Check spectral
            lsp = loaded["spectral"]
            assert lsp["n_layers"] == n_layers
            assert lsp["d_model"] == d_model
            for i in range(n_layers):
                np.testing.assert_allclose(lsp["eigenvalues"][i], spectral["eigenvalues"][i], rtol=1e-5)

            # Check metadata file
            with open(results_dir / "step0042" / "metadata.json") as f:
                meta = json.load(f)
            assert meta["step"] == 42
            assert meta["n_layers"] == n_layers
            assert meta["d_model"] == d_model

    def test_roundtrip_missing_last_layer(self):
        """Last layer has no data (like Pythia layer 11) — save and load should work."""
        n_layers, d_model = 12, 768
        mask = [True] * 11 + [False]  # layers 0-10: data, layer 11: None
        stats = make_stats(n_layers, d_model, population_mask=mask)
        spectral = make_spectral(n_layers, d_model, population_mask=mask)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            # THIS MUST NOT CRASH (was the bug)
            save_checkpoint_result(results_dir, step=1, stats=stats, spectral=spectral)

            assert is_checkpoint_done(results_dir, 1)

            loaded = load_checkpoint_result(results_dir, 1)

            ls = loaded["stats"]
            lsp = loaded["spectral"]

            # Layers 0-10: data
            for i in range(11):
                assert ls["jbar"][i] is not None
                assert ls["jtj_gram"][i] is not None
                assert ls["coherence"][i] is not None
                assert lsp["eigenvalues"][i] is not None

            # Layer 11: None / NaN (stored as NaN in HDF5, loaded back as np.nan)
            assert ls["jbar"][11] is None
            assert ls["jtj_gram"][11] is None
            assert np.isnan(ls["coherence"][11])  # stored as np.nan
            assert lsp["eigenvalues"][11] is not None  # array([nan]) loaded from HDF5
            assert np.isnan(lsp["eigenvalues"][11]).all()

    def test_roundtrip_sparse_layers(self):
        """Only specific layers have data (like layers 0, 2, 5, 9 of Pythia)."""
        n_layers, d_model = 12, 768
        mask = [i in (0, 2, 5, 7, 9, 10) for i in range(n_layers)]
        stats = make_stats(n_layers, d_model, population_mask=mask)
        spectral = make_spectral(n_layers, d_model, population_mask=mask)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(results_dir, step=99, stats=stats, spectral=spectral)
            loaded = load_checkpoint_result(results_dir, 99)

            for i in range(n_layers):
                if mask[i]:
                    assert loaded["stats"]["jbar"][i] is not None
                    assert loaded["spectral"]["eigenvalues"][i] is not None
                else:
                    assert loaded["stats"]["jbar"][i] is None
                    # eigenvalues stored as array([nan]), loaded back as array([nan])
                    assert loaded["spectral"]["eigenvalues"][i] is not None
                    assert np.isnan(loaded["spectral"]["eigenvalues"][i]).all()

    def test_is_checkpoint_done_false(self):
        """is_checkpoint_done returns False for non-existent checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            assert not is_checkpoint_done(Path(tmp), 999)

    def test_is_checkpoint_done_string_path(self):
        """is_checkpoint_done works with string results_dir (THE BUG)."""
        with tempfile.TemporaryDirectory() as tmp:
            stats = make_stats(3, 8)
            spectral = make_spectral(3, 8)
            save_checkpoint_result(Path(tmp), step=7, stats=stats, spectral=spectral)

            # This used to crash with 'str / str' TypeError
            assert is_checkpoint_done(tmp, 7)
            assert not is_checkpoint_done(tmp, 42)

    def test_load_nonexistent_raises(self):
        """Loading a non-existent checkpoint should raise."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(Exception):
                load_checkpoint_result(Path(tmp), 999)
