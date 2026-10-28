"""Tests for the Olmo-3 λ → λ/d_model normalization in olmo3_analysis/.

The analysis is meant to be a pure change of units: the fit and histogram
windows are the raw [1e-6, 1e3] windows divided by d_model, so normalized
results equal raw results with log-locations shifted by -log(d_model).
"""

import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

OLMO3_DIR = Path(__file__).resolve().parents[1] / "olmo3_analysis"
sys.path.insert(0, str(OLMO3_DIR))

import fit_pythia_three_component_mixture as fit3  # noqa: E402
import olmo3_driver  # noqa: E402
import plot_logskewnormal_fits as lsn  # noqa: E402
import plot_spectra_histogram_dashboard as dash  # noqa: E402
import plot_spectrum_histogram as hist  # noqa: E402

D = 5120
RAW_BINS = np.geomspace(1e-6, 1e3, 811)


def write_spectra(path: Path, layers: list[np.ndarray], d_model: int = D) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["d_model"] = d_model
        handle.attrs["n_layers"] = len(layers)
        for layer, values in enumerate(layers):
            handle.create_dataset(f"layer_{layer}/eigenvalues", data=values)


def raw_spectrum(seed: int, n: int = 2000) -> np.ndarray:
    """Log-normal raw eigenvalues centred near 1e-2, like a mid-training layer."""
    return np.exp(np.random.default_rng(seed).normal(np.log(1e-2), 1.5, n))


class TestFitWindow:
    def test_bins_are_raw_window_divided_by_d_model(self):
        assert lsn.D_MODEL == D
        np.testing.assert_allclose(lsn.BINS, RAW_BINS / D)

    def test_histogram_masses_unit_invariant(self):
        raw = raw_spectrum(0)
        expected = np.histogram(raw, bins=RAW_BINS)[0] / raw.size
        np.testing.assert_allclose(lsn.histogram_masses(raw / D), expected)

    def test_normalized_eigenvalues_divides_and_filters(self, tmp_path):
        values = np.array([D * 1.0, 0.0, -1.0, np.nan, D * 2.0])
        write_spectra(tmp_path / "spectra.h5", [values])
        with h5py.File(tmp_path / "spectra.h5", "r") as handle:
            np.testing.assert_allclose(lsn.normalized_eigenvalues(handle, 0), [1.0, 2.0])

    def test_normalized_eigenvalues_rejects_other_d_model(self, tmp_path):
        write_spectra(tmp_path / "spectra.h5", [np.ones(4)], d_model=768)
        with h5py.File(tmp_path / "spectra.h5", "r") as handle:
            with pytest.raises(ValueError, match="d_model"):
                lsn.normalized_eigenvalues(handle, 0)


class TestThreeComponentFit:
    def test_fit_sees_whole_normalized_spectrum(self):
        """A raw spectrum inside [1e-6, 1e3] must stay fully inside the fit window."""
        values = raw_spectrum(1, n=200_000) / D
        assert lsn.histogram_masses(values).sum() == pytest.approx(1.0, abs=1e-3)
        r2, parameters = fit3.fit_three_component_mixture(values)
        assert r2 > 0.98

    def test_fit_checkpoint_reports_normalized_parameters(self, tmp_path):
        raw = raw_spectrum(2)
        write_spectra(tmp_path / "spectra.h5", [raw, raw])
        (row,) = fit3.fit_checkpoint(0, tmp_path / "spectra.h5")
        weights = np.array([row["mp_weight"], row["semicircle_weight"], row["lognormal_weight"]])
        # Every log-domain component that carries weight sits at the normalized scale.
        centres = np.array([
            np.log(row["mp_sigma2"]), row["semicircle_center"], row["lognormal_location"]
        ])
        expected = np.log(1e-2) - np.log(D)
        assert np.average(centres, weights=weights) == pytest.approx(expected, abs=1.0)


class TestHistogramDashboard:
    def test_window_constants_are_normalized(self):
        assert dash.HISTOGRAM_X_MIN == pytest.approx(1e-6 / D)
        assert dash.HISTOGRAM_X_MAX == pytest.approx(1e3 / D)
        bins = dash.histogram_bins(90)
        assert bins[0] == pytest.approx(dash.HISTOGRAM_X_MIN)
        assert bins[-1] == pytest.approx(dash.HISTOGRAM_X_MAX)

    def test_read_eigenvalues_uses_file_d_model(self, tmp_path):
        write_spectra(tmp_path / "spectra.h5", [np.array([D * 3.0, 0.0])])
        (values,) = dash.read_eigenvalues(tmp_path, 1)
        np.testing.assert_allclose(values, [3.0])

    def test_dashboard_axis_matches_bins(self, tmp_path, monkeypatch):
        for step in (0, 10):
            write_spectra(tmp_path / f"step{step}" / "spectra.h5", [raw_spectrum(step), raw_spectrum(step + 1)])
        captured = {}
        monkeypatch.setattr(hist, "write_dashboard", lambda figure, output, log_space: captured.update(figure=figure))
        hist.make_dashboard(tmp_path, "wikitext", "Olmo-3-1125-32B", tmp_path / "out.html", 90)
        figure = captured["figure"]
        np.testing.assert_allclose(figure.layout.xaxis.range, np.log10([1e-6 / D, 1e3 / D]))
        x = np.asarray(figure.data[0].x)
        assert x.min() > 1e-6 / D and x.max() < 1e3 / D
        # The whole synthetic spectrum lands inside the plotted window.
        y = np.asarray(figure.data[0].y)
        assert np.sum(y * np.diff(np.log10(dash.histogram_bins(90)))) == pytest.approx(1.0, abs=1e-3)
        annotation = figure.layout.annotations[0].text
        assert "Raw (unnormalized)" not in annotation


class TestDriver:
    def test_main_honours_directory_flags(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        write_spectra(results / "step0" / "spectra.h5", [raw_spectrum(3), raw_spectrum(4), raw_spectrum(5)])
        parts, figures = tmp_path / "parts", tmp_path / "figures"
        monkeypatch.setattr(sys, "argv", [
            "olmo3_driver.py", "--results-dir", str(results), "--parts-dir", str(parts),
            "--figures-dir", str(figures), "--overlay-steps", "0",
        ])
        olmo3_driver.main()
        with (parts / "step_0.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert (tmp_path / "olmo3_3mix_all.csv").exists()
        assert (figures / "olmo3_three_component_overlay_step0.png").exists()
        assert (figures / "olmo3_three_component_mixture_composition_heatmap.png").exists()

    def test_parallel_fit_pickles(self, tmp_path, monkeypatch):
        """SLURM_CPUS_PER_TASK > 1 fits steps in a process pool."""
        results = tmp_path / "results"
        for step in (0, 10):
            write_spectra(results / f"step{step}" / "spectra.h5", [raw_spectrum(step), raw_spectrum(step + 1)])
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
        olmo3_driver.fit_all_steps(results, [0, 10], tmp_path / "parts")
        assert sorted(p.name for p in (tmp_path / "parts").iterdir()) == ["step_0.csv", "step_10.csv"]


L62_SPECTRA = Path("/data/jlens/olmo3-32b/step370000/spectra.h5")


def mixture_raw_spectrum(seed: int, n: int = 20_000) -> np.ndarray:
    """Narrow, peaked spectrum near λ = 1 (raw), like the last layers."""
    rng = np.random.default_rng(seed)
    core = np.exp(rng.normal(0.0, 0.03, n // 2))
    shoulder = np.exp(rng.normal(-0.1, 0.25, n - n // 2))
    return np.concatenate([core, shoulder])


class TestRobustMixtureFit:
    def test_params_from_row_roundtrip(self, tmp_path):
        raw = raw_spectrum(6, n=20_000)
        write_spectra(tmp_path / "spectra.h5", [raw, raw])
        (row,) = fit3.fit_checkpoint(0, tmp_path / "spectra.h5")
        target = lsn.histogram_masses(raw / D)
        r2 = lsn.r2_score(target, fit3.mixture_masses(fit3.params_from_row(row)))
        assert r2 == pytest.approx(row["r2"], abs=1e-6)

    def test_params_from_row_clips_degenerate_weights(self):
        row = {"mp_sigma2": 1e-4, "mp_q": 1e-3, "semicircle_center": -8.5, "semicircle_radius": 0.05,
               "lognormal_location": -8.5, "lognormal_log_scale": -2.0,
               "mp_weight": 0.0, "semicircle_weight": 0.5, "lognormal_weight": 0.5}
        parameters = fit3.params_from_row(row)
        assert np.all(parameters >= fit3.LOWER) and np.all(parameters <= fit3.UPPER)

    def test_multistart_varies_mp_shape(self):
        assert len(fit3.MP_Q_STARTS) >= 3
        assert min(fit3.MP_Q_STARTS) > fit3.LOWER[1]

    def test_refine_never_worse_than_seed(self):
        target = lsn.histogram_masses(mixture_raw_spectrum(7) / D)
        r2_fit, x_fit = fit3.fit_three_component_mixture(mixture_raw_spectrum(7) / D)
        r2_refined, _ = fit3.refine_from_seeds(target, [x_fit])
        assert r2_refined >= r2_fit - 1e-9

    def test_fit_checkpoint_with_seeds_never_worse(self, tmp_path):
        write_spectra(tmp_path / "spectra.h5", [mixture_raw_spectrum(8), mixture_raw_spectrum(9), np.ones(4)])
        first = fit3.fit_checkpoint(0, tmp_path / "spectra.h5")
        seeds = [{row["layer"]: row for row in first}]
        second = fit3.fit_checkpoint(0, tmp_path / "spectra.h5", seed_rows=seeds)
        assert [r["layer"] for r in second] == [0, 1]
        for a, b in zip(first, second):
            assert b["r2"] >= a["r2"] - 1e-9

    def test_main_seed_dir_uses_neighbouring_steps(self, tmp_path, monkeypatch):
        results, seed_dir = tmp_path / "results", tmp_path / "pass1"
        for step in (0, 10, 20):
            write_spectra(results / f"step{step}" / "spectra.h5", [mixture_raw_spectrum(step), np.ones(4)])
        for step in (0, 10, 20):
            monkeypatch.setattr(sys, "argv", ["fit", "--results-dir", str(results), "--steps", str(step),
                                              "--csv", str(seed_dir / f"step_{step}.csv")])
            fit3.main()
        assert fit3.neighbour_steps(10, [0, 10, 20]) == [0, 10, 20]
        assert fit3.neighbour_steps(0, [0, 10, 20]) == [0, 10]
        monkeypatch.setattr(sys, "argv", ["fit", "--results-dir", str(results), "--steps", "10",
                                          "--csv", str(tmp_path / "out.csv"), "--seed-dir", str(seed_dir)])
        fit3.main()
        with (tmp_path / "out.csv").open(newline="") as handle:
            (refined,) = list(csv.DictReader(handle))
        with (seed_dir / "step_10.csv").open(newline="") as handle:
            (seed,) = list(csv.DictReader(handle))
        assert float(refined["r2"]) >= float(seed["r2"]) - 1e-9

    @pytest.mark.skipif(not L62_SPECTRA.exists(), reason="needs OLMo-3 spectra")
    def test_layer62_step370k_regression(self):
        """Cell where the 4-start fit got trapped at the q bound (R² 0.935)."""
        with h5py.File(L62_SPECTRA, "r") as handle:
            values = lsn.normalized_eigenvalues(handle, 62)
        r2, parameters = fit3.fit_three_component_mixture(values)
        assert r2 > 0.99
