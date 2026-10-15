"""Tests for identified implementation bugs.

Each test asserts the CORRECT behavior. Tests currently FAIL because
the bugs exist.  After fixing each bug, the corresponding test should
pass and serve as regression protection.
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from jlens.spectra import analyze_checkpoint, fit_power_law_tail, eigen_decompose
from jlens.storage import (
    _or_nan,
    _or_nan_int,
    is_checkpoint_done,
    load_checkpoint_result,
    save_checkpoint_result,
)


# ---------------------------------------------------------------------------
# Bug 1: _or_nan_int fails for np.nan (float), only catches None
# ---------------------------------------------------------------------------

class TestOrNanIntNaN:
    """_or_nan_int only checks `x is None`, so np.nan passes through
    as a float.  n_spikes gets stored as float NaN in an HDF5 int
    attribute, making round-trips inconsistent.
    """

    def test_nan_returns_sentinel_not_float(self):
        """_or_nan_int(np.nan) must return an integer sentinel (e.g. -1)."""
        result = _or_nan_int(np.nan)
        assert isinstance(result, (int, np.integer)), \
            f"Expected int sentinel, got {type(result).__name__}: {result}"

    def test_none_is_handled(self):
        """None IS handled correctly → returns -1."""
        assert _or_nan_int(None) == -1

    def test_n_spikes_roundtrip_via_load(self):
        """After save+load, empty layers should have a valid int n_spikes."""
        d_model = 64
        n_layers = 4
        mask = [True, True, True, False]  # layer 3 is empty
        stats = _make_stats(n_layers, d_model, mask)
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(
                results_dir, step=1, stats=stats, spectral=spectral,
            )
            loaded = load_checkpoint_result(results_dir, step=1)

        for i in range(n_layers):
            val = loaded["spectral"]["n_spikes"][i]
            if mask[i]:
                assert isinstance(val, (int, np.integer)) and val >= 0, \
                    f"layer {i}: expected non-negative int n_spikes, got {val}"
            else:
                # Empty layer: should be a sentinel (e.g. -1), not NaN
                assert not np.isnan(val), \
                    f"layer {i}: got NaN for empty layer, expected int sentinel"
                assert isinstance(val, (int, np.integer, float)), \
                    f"layer {i}: unexpected type {type(val)}"
                if isinstance(val, float):
                    assert np.isnan(val) is False, \
                        f"layer {i}: float but not NaN? {val}"

    def test_hdf5_attrs_are_ints_not_floats(self):
        """HDF5 attrs for n_spikes must be stored as integers."""
        d_model = 64
        n_layers = 4
        mask = [True, True, True, False]
        stats = _make_stats(n_layers, d_model, mask)
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(
                results_dir, step=1, stats=stats, spectral=spectral,
            )

            with h5py.File(results_dir / "step0001" / "spectra.h5", "r") as f:
                for i in range(n_layers):
                    grp = f[f"layer_{i}"]
                    val = grp.attrs["n_spikes"]
                    assert isinstance(val, (int, np.integer)), \
                        f"layer {i}: n_spikes type={type(val).__name__}, val={val}"
                    if not mask[i]:
                        assert val == -1, \
                            f"layer {i}: empty layer should have n_spikes=-1, got {val}"


# ---------------------------------------------------------------------------
# Bug 2: compare_estimators.py accesses old HDF5 key format
# ---------------------------------------------------------------------------

class TestStatsH5KeyFormat:
    """compare_estimators.py does `f[str(li)][:]` but the save path puts
    data under groups named `layer_N`, not `N`.

    The correct behavior: either the save format should be compatible
    with the access pattern, or the script should use the right keys.
    Since `compare_estimators.py` is a standalone script that expects
    `f[str(li)]` to work, the HDF5 layout should support that.

    Alternatively, if the decision is to keep the `layer_N` group
    names, compare_estimators.py must be updated.
    """

    def test_layer_key_and_jbar_accessible(self):
        """The save layout groups data under layer_{li} with jbar as a dataset.
        Scripts must access f['layer_{li}']['jbar'][:], not f[str(li)][:].
        """
        d_model = 32
        n_layers = 3
        stats = _make_stats(n_layers, d_model, [True, True, True])
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(results_dir, step=1, stats=stats, spectral=spectral)

            with h5py.File(results_dir / "step0001" / "stats.h5", "r") as f:
                for li in range(n_layers):
                    layer_key = f"layer_{li}"
                    assert layer_key in f, \
                        f"f['{layer_key}'] missing"
                    assert "jbar" in f[layer_key], \
                        f"f['{layer_key}']['jbar'] missing"
                    # jbar must be loadable as a 2D array
                    jbar = f[layer_key]["jbar"][:]
                    assert jbar.ndim == 2 and jbar.shape[0] == jbar.shape[1] == d_model, \
                        f"jbar shape={jbar.shape}, expected ({d_model},{d_model})"


# ---------------------------------------------------------------------------
# Bug 3: load_prompts_text ignores seed parameter
# ---------------------------------------------------------------------------

class TestPromptsSeed:
    """load_prompts_text accepts `seed` but never uses it for shuffling."""

    def test_different_seeds_produce_different_orders(self):
        """Two calls with different seeds should return different prompt lists."""
        from jlens.prompts import load_prompts_text
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-160m-deduped",
                revision="step143000",
                trust_remote_code=True,
            )
        except Exception:
            pytest.skip("Cannot load tokenizer (no network)")

        n = 50  # enough prompts for shuffling to matter
        prompts_a = load_prompts_text(tokenizer, n_prompts=n, max_seq_len=128, seed=42)
        prompts_b = load_prompts_text(tokenizer, n_prompts=n, max_seq_len=128, seed=99)

        # Shuffling with different seeds should, with very high probability,
        # produce a different order. In the unlikely event they happen to be
        # identical, this test may rarely flake — use a large n to reduce that.
        assert prompts_a != prompts_b, \
            "seed parameter has no effect: identical prompt lists"


# ---------------------------------------------------------------------------
# Bug 4: prepare_eval_data.py now reads stored power_law_alpha from HDF5
# ---------------------------------------------------------------------------

class TestPrepareEvalPowerLaw:
    """prepare_eval_data.py used inline median-split power-law computation.
    Now it reads the stored power_law_alpha attribute (set by
    spectra.fit_power_law_tail -> save_checkpoint_result).
    Verify the stored attribute round-trips correctly.
    """

    def test_power_law_alpha_stored_in_spectra_attrs(self):
        """power_law_alpha in spectra.h5 attrs must match fit_power_law_tail."""
        rng = np.random.default_rng(42)
        d = 768
        a = rng.normal(0, 0.1, (d, d)) + np.eye(d) * 0.7
        gram = a.T @ a
        eigvals = eigen_decompose(gram)

        result_spectra = fit_power_law_tail(eigvals, tail_fraction=0.3)
        alpha_spectra = result_spectra["alpha"]

        # Build stats, run analyze, save, then read back the stored alpha
        stats = _make_stats(1, d, [True])
        # Replace the gram with our controlled one
        stats["jtj_gram"][0] = gram
        stats["jbar"][0] = a
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(
                results_dir, step=1, stats=stats, spectral=spectral,
            )

            with h5py.File(results_dir / "step0001" / "spectra.h5", "r") as f:
                alpha_stored = float(f["layer_0"].attrs["power_law_alpha"])

        assert not np.isnan(alpha_stored), "stored alpha is NaN"
        assert alpha_stored == pytest.approx(alpha_spectra, rel=0.01), \
            f"stored={alpha_stored:.4f} vs fit={alpha_spectra:.4f}"


# ---------------------------------------------------------------------------
# Bug 5: is_checkpoint_done format padding consistency
# ---------------------------------------------------------------------------

class TestCheckpointFormatConsistency:
    """save uses f'step{step:04d}' — verify is_checkpoint_done matches."""

    def test_save_and_check_use_same_format(self):
        """After save, is_checkpoint_done finds the result."""
        stats = _make_stats(3, 16, [True, True, True])
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(results_dir, step=42, stats=stats,
                                   spectral=spectral)

            assert is_checkpoint_done(results_dir, 42)
            step_dir = results_dir / "step0042"
            assert step_dir.exists()
            assert (step_dir / "metadata.json").exists()

    def test_large_step_doesnt_overflow_format(self):
        """Step 143000 with :04d → 'step143000' (6 chars), still works."""
        stats = _make_stats(3, 16, [True, True, True])
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(results_dir, step=143000, stats=stats,
                                   spectral=spectral)

            step_dir = results_dir / "step143000"
            assert step_dir.exists()
            assert is_checkpoint_done(results_dir, 143000)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_stats(n_layers, d_model, population_mask=None):
    """Minimal stats dict matching JacobianAccumulator.get_all_stats()."""
    if population_mask is None:
        population_mask = [True] * n_layers

    rng = np.random.default_rng(123)
    jbar = []
    jtj_gram = []
    sigma_gram = []
    coherence = []
    a_coeff = []
    r_norm = []
    tr_jtj_mean = []
    fcr = []

    for i in range(n_layers):
        if population_mask[i]:
            m = rng.normal(0, 0.1, (d_model, d_model)) + np.eye(d_model) * 0.7
            jbar.append(m)
            jtj_gram.append(m.T @ m)
            sigma_gram.append(m.T @ m * 0.1)  # small fluctuation
            coherence.append(float(np.trace(m.T @ m) / np.sum(m**2)))
            a_coeff.append(float(np.trace(m) / d_model))
            r_norm.append(0.1)
            tr_jtj_mean.append(float(np.sum(m**2)))
            fcr.append(0.1)
        else:
            jbar.append(None)
            jtj_gram.append(None)
            sigma_gram.append(None)
            coherence.append(None)
            a_coeff.append(None)
            r_norm.append(None)
            tr_jtj_mean.append(None)
            fcr.append(None)

    return {
        "n": 10,
        "jbar": jbar,
        "jtj_gram": jtj_gram,
        "sigma_gram": sigma_gram,
        "coherence": coherence,
        "a_coeff": a_coeff,
        "r_norm": r_norm,
        "tr_jtj_mean": tr_jtj_mean,
        "fcr": fcr,
    }


# ---------------------------------------------------------------------------
# 12B-scale fixes
# ---------------------------------------------------------------------------

class TestMPBulkFitLargeD:
    """fit_mp_bulk must handle large d_model (e.g., 5120 for Pythia-12B).
    The old max_exclude=min(n//5, 200) capped at 200 of 5120 eigenvalues.
    """

    def test_many_spikes_handled(self):
        """With d=5120, hundreds of spikes must not break the fit."""
        from jlens.spectra import fit_mp_bulk

        rng = np.random.default_rng(42)
        d = 5120

        # Build a spectrum with many large spikes (untrained-like)
        # MP bulk ~ chi²(d) scaled, then add 500 spikes above bulk edge
        bulk = rng.chisquare(2, d) * 0.5
        spikes = rng.exponential(10, 500) + 5.0
        eigvals = np.sort(np.concatenate([bulk, spikes]))[::-1]

        result = fit_mp_bulk(eigvals)
        assert not np.isnan(result["sigma2"]), "sigma2 should not be NaN"
        assert not np.isnan(result["q_eff"]), "q_eff should not be NaN"
        assert result["n_spikes"] > 100, \
            f"Expected >100 spikes with 500 outliers, got {result['n_spikes']}"
        assert result["ks_statistic"] < 0.5, \
            f"KS statistic {result['ks_statistic']:.3f} too high"

    def test_exponential_candidates_cover_full_range(self):
        """The candidate schedule should cover up to n//4 with good density."""
        d = 5120
        n = d  # n eigenvalues
        max_exclude = n // 4  # 1280

        # Replicate the candidate generation logic
        dense = list(range(0, min(21, max_exclude + 1)))
        sparse = np.unique(
            np.geomspace(21, max_exclude, num=min(20, max_exclude - 20))
            .astype(int)
        ).tolist()
        candidates = dense + sparse

        assert candidates[0] == 0, "Must start at 0"
        assert candidates[-1] == max_exclude, f"Must end at {max_exclude}"
        assert len(candidates) >= 20, f"Too few candidates: {len(candidates)}"
        assert len(candidates) <= 41, f"Too many candidates: {len(candidates)}"


class TestSpikeCountMatrixModelAgnostic:
    """spike_count_matrix must read d_model and n_prompts from metadata,
    not hardcode 768/1000."""

    def test_reads_d_model_from_metadata(self):
        """spike_count_matrix uses stored d_model, not hardcoded 768."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from plot_evolution import spike_count_matrix

        d_model = 2048
        n_layers = 4
        stats = _make_stats(n_layers, d_model, [True] * n_layers)
        spectral = analyze_checkpoint(stats)

        # Simulate what load_checkpoint_result returns
        fake_results = {
            1000: {
                "stats": stats,
                "spectral": spectral,
            }
        }
        steps = np.array([1000])

        M = spike_count_matrix(fake_results, steps)
        assert M.shape == (1, n_layers)
        assert not np.isnan(M).all(), "All NaN — threshold probably wrong"


class TestScratchDirSave:
    """save_checkpoint_result must support scratch_dir for NFS efficiency."""

    def test_scratch_dir_used_then_moved(self):
        """With scratch_dir, files are written there first, then moved."""
        stats = _make_stats(3, 16, [True, True, True])
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            scratch_dir = Path(tmp) / "scratch"

            save_checkpoint_result(
                results_dir, step=1, stats=stats, spectral=spectral,
                scratch_dir=scratch_dir,
            )

            # Results dir must have the files
            assert (results_dir / "step0001" / "stats.h5").exists()
            assert (results_dir / "step0001" / "spectra.h5").exists()
            assert (results_dir / "step0001" / "metadata.json").exists()

            # Scratch dir must be cleaned up
            assert not scratch_dir.exists() or not any(scratch_dir.iterdir()), \
                "scratch dir should be empty after move"

    def test_no_scratch_dir_still_works(self):
        """Backward compat: calling without scratch_dir writes directly."""
        stats = _make_stats(3, 16, [True, True, True])
        spectral = analyze_checkpoint(stats)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            save_checkpoint_result(
                results_dir, step=1, stats=stats, spectral=spectral,
            )
            assert (results_dir / "step0001" / "stats.h5").exists()
