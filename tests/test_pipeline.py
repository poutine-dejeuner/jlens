"""Integration test: simulate the full pipeline from accumulator → storage → spectra.

This catches the exact bug pattern that wasted tens of hours:
  - accumulator.get_all_stats() returns None for layer with no data
  - save_checkpoint_result writes None layers without crashing
  - analyze_checkpoint handles None gram matrices
  - load_checkpoint_result roundtrips correctly
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from jlens.accumulator import JacobianAccumulator
from jlens.spectra import analyze_checkpoint
from jlens.storage import (
    is_checkpoint_done,
    load_checkpoint_result,
    save_checkpoint_result,
)


class TestFullPipeline:
    """Simulate the exact pattern that caused the crash."""

    def test_pythia_12_layer_pattern(self):
        """12 layers (like Pythia-160M), only 0-10 get Jacobian data, layer 11 is None.

        This is EXACTLY what the real pipeline does.
        """
        d_model = 768
        n_layers = 12
        acc = JacobianAccumulator(n_layers=n_layers, d_model=d_model)

        # Simulate 100 prompts: only layers 0-10 get Jacobians
        rng = np.random.default_rng(42)
        for _ in range(100):
            jacs = {}
            for i in range(11):
                a = torch.from_numpy(rng.normal(0, 0.1, (d_model, d_model)).astype(np.float32))
                jacs[i] = a + torch.eye(d_model) * 0.7
            acc.update(jacs)

        # Step 1: get_all_stats — MUST NOT CRASH
        stats = acc.get_all_stats()

        # Verify: layers 0-10 have data, layer 11 is None
        assert stats["jbar"][0] is not None
        assert stats["jtj_gram"][0] is not None
        assert stats["coherence"][0] is not None
        assert stats["jbar"][11] is None

        # Step 2: analyze_checkpoint — MUST NOT CRASH
        spectral = analyze_checkpoint(stats)

        assert len(spectral["eigenvalues"]) == 12
        assert spectral["eigenvalues"][0] is not None
        assert len(spectral["eigenvalues"][0]) == d_model
        assert spectral["eigenvalues"][11] is not None  # array of NaN (placeholder)

        # Step 3: save — MUST NOT CRASH
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(results_dir, step=1000, stats=stats, spectral=spectral)

            assert is_checkpoint_done(results_dir, 1000)

            # Step 4: load — MUST NOT CRASH
            loaded = load_checkpoint_result(results_dir, 1000)

            # Verify roundtrip
            ls = loaded["stats"]
            lsp = loaded["spectral"]

            # Layer 0: full data
            np.testing.assert_allclose(ls["jbar"][0], stats["jbar"][0], rtol=1e-4)
            assert ls["coherence"][0] == pytest.approx(stats["coherence"][0])

            # Layer 11: None / NaN (stored as NaN in HDF5, loaded as float NaN)
            assert ls["jbar"][11] is None
            assert ls["jtj_gram"][11] is None
            assert np.isnan(ls["coherence"][11])
            assert lsp["eigenvalues"][11] is not None  # array([nan]) from HDF5
            assert np.isnan(lsp["eigenvalues"][11]).all()

    def test_all_layers_populated(self):
        """All layers populated — normal case."""
        d_model = 128
        n_layers = 6
        acc = JacobianAccumulator(n_layers=n_layers, d_model=d_model)

        rng = np.random.default_rng(7)
        for _ in range(50):
            jacs = {}
            for i in range(n_layers):
                a = torch.from_numpy(rng.normal(0, 0.05, (d_model, d_model)).astype(np.float32))
                jacs[i] = a + torch.eye(d_model)
            acc.update(jacs)

        stats = acc.get_all_stats()
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(results_dir, step=5000, stats=stats, spectral=spectral)
            loaded = load_checkpoint_result(results_dir, 5000)

            for i in range(n_layers):
                assert loaded["stats"]["jbar"][i] is not None
                assert loaded["spectral"]["eigenvalues"][i] is not None
                assert not np.isnan(loaded["spectral"]["eigenvalues"][i]).all()

    def test_multiple_checkpoints(self):
        """Simulate sequential processing of multiple checkpoints."""
        d_model = 64
        n_layers = 4
        mask = [True, True, True, False]  # last layer empty

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)

            for step in [1000, 2000, 5000, 10000, 20000, 50000, 143000]:
                acc = JacobianAccumulator(n_layers=n_layers, d_model=d_model)
                rng = np.random.default_rng(step)

                for _ in range(20):
                    jacs = {}
                    for i in range(n_layers - 1):
                        a = torch.from_numpy(rng.normal(0, 0.1, (d_model, d_model)).astype(np.float32))
                        jacs[i] = a + torch.eye(d_model) * 0.6
                    acc.update(jacs)

                stats = acc.get_all_stats()
                spectral = analyze_checkpoint(stats)
                save_checkpoint_result(results_dir, step=step, stats=stats, spectral=spectral)

                assert is_checkpoint_done(results_dir, step)

            # Load them all back
            for step in [1000, 2000, 5000, 10000, 20000, 50000, 143000]:
                loaded = load_checkpoint_result(results_dir, step)
                assert loaded["stats"]["n"] == 20
                assert loaded["spectral"]["n_layers"] == n_layers
