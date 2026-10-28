"""Tests for jlens.gaussianity on synthetic matrices with known structure."""

import numpy as np

from jlens.gaussianity import (
    gaussian_null,
    looks_gaussian,
    mp_cdf_square,
    remove_identity,
    summarize,
)

D = 512


def test_mp_cdf_is_valid_cdf():
    x = np.linspace(0.0, 4.0, 50)
    cdf = mp_cdf_square(x)
    assert np.all(np.diff(cdf) >= -1e-9)
    assert cdf[0] < 1e-3 and abs(cdf[-1] - 1.0) < 1e-6


def test_gaussian_matrix_is_accepted():
    summary = summarize(2.5 * gaussian_null(D, seed=1) + 0.3)
    assert looks_gaussian(summary, D), summary


def test_heavy_tailed_entries_are_rejected():
    rng = np.random.default_rng(2)
    summary = summarize(rng.standard_t(df=3, size=(D, D)))
    assert summary["excess_kurtosis"] > 1.0
    assert not looks_gaussian(summary, D)


def test_row_scaled_gaussian_is_rejected():
    rng = np.random.default_rng(3)
    scales = np.exp(rng.normal(scale=0.5, size=(D, 1)))
    summary = summarize(scales * rng.standard_normal((D, D)))
    assert summary["row_std_cv"] > 3 * summary["cv_gaussian_null"]
    assert not looks_gaussian(summary, D)


def test_low_rank_plus_noise_has_spectral_outliers():
    rng = np.random.default_rng(4)
    u, v = rng.standard_normal((D, 8)), rng.standard_normal((D, 8))
    summary = summarize(rng.standard_normal((D, D)) + 0.8 * u @ v.T / np.sqrt(8))
    assert summary["n_above_edge"] >= 3
    assert not looks_gaussian(summary, D)


def test_remove_identity_recovers_noise_part():
    noise = gaussian_null(D, seed=5)
    assert looks_gaussian(summarize(remove_identity(np.eye(D) + noise)), D)
    assert not looks_gaussian(summarize(np.eye(D) + 0.1 * noise), D)
