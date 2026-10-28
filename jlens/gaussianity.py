"""Does a square matrix look like i.i.d. Gaussian entries?

Entry-level, row/column-scale and spectral diagnostics. Effect sizes are
reported instead of p-values: with ~2.6e7 entries any test rejects tiny
deviations, so each statistic is compared with its value for a true Gaussian
matrix of the same size.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mp_cdf_square(x: np.ndarray) -> np.ndarray:
    """CDF of the Marchenko-Pastur law with q = 1 (support [0, 4], unit variance)."""
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 4.0)
    t = np.linspace(0.0, np.pi, 20001)
    lam = 4.0 * np.sin(t / 2.0) ** 2
    pdf = np.sqrt(np.maximum(4.0 * lam - lam**2, 0.0)) / (2.0 * np.pi * np.maximum(lam, 1e-12))
    # Integrate in lambda: pdf(lam) dlam, with dlam = 2 sin(t) dt
    integrand = pdf * 2.0 * np.sin(t)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    return np.interp(x, lam, cdf)


def remove_identity(matrix: np.ndarray) -> np.ndarray:
    """Return `matrix - I` (residual-connection part removed)."""
    return matrix - np.eye(matrix.shape[0], dtype=matrix.dtype)


def entry_statistics(matrix: np.ndarray, max_entries: int = 2_000_000, seed: int = 0) -> dict[str, float]:
    """Moments and tail statistics of the standardized entries."""
    entries = np.asarray(matrix, dtype=np.float64).ravel()
    mean, std = float(entries.mean()), float(entries.std())
    z = (entries - mean) / std
    rng = np.random.default_rng(seed)
    sample = z if z.size <= max_entries else rng.choice(z, max_entries, replace=False)
    return {
        "mean": mean,
        "std": std,
        "skew": float(stats.skew(z)),
        "excess_kurtosis": float(stats.kurtosis(z)),
        "ks_vs_normal": float(stats.kstest(sample, "norm").statistic),
        "frac_beyond_4sigma": float(np.mean(np.abs(z) > 4.0)),
        "frac_beyond_4sigma_gaussian": float(2 * stats.norm.sf(4.0)),
    }


def scale_dispersion(matrix: np.ndarray) -> dict[str, float]:
    """Coefficient of variation of row and column standard deviations.

    For i.i.d. entries it is ~ 1/sqrt(2 d) (gaussian null returned alongside).
    """
    m = np.asarray(matrix, dtype=np.float64)
    d = m.shape[0]
    row = m.std(axis=1)
    col = m.std(axis=0)
    return {
        "row_std_cv": float(row.std() / row.mean()),
        "col_std_cv": float(col.std() / col.mean()),
        "cv_gaussian_null": float(1.0 / np.sqrt(2.0 * d)),
    }


def spectral_statistics(matrix: np.ndarray) -> dict[str, float]:
    """Compare eigenvalues of M^T M / (d sigma^2) with Marchenko-Pastur (q = 1)."""
    m = np.asarray(matrix, dtype=np.float64)
    d = m.shape[0]
    sigma2 = float(m.var())
    singular = np.linalg.svd(m, compute_uv=False)
    eig = singular**2 / (d * sigma2)
    n = eig.size
    sorted_eig = np.sort(eig)
    ks = float(np.max(np.abs(mp_cdf_square(sorted_eig) - (np.arange(1, n + 1) / n))))
    return {
        "mp_ks": ks,
        "top_eig_over_edge": float(sorted_eig[-1] / 4.0),
        "n_above_edge": int(np.sum(sorted_eig > 4.0 * (1.0 + 3.0 * d ** (-2.0 / 3.0)))),
    }


def gaussian_null(d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((d, d))


def summarize(matrix: np.ndarray) -> dict[str, float]:
    return {**entry_statistics(matrix), **scale_dispersion(matrix), **spectral_statistics(matrix)}


def looks_gaussian(summary: dict[str, float], d: int) -> bool:
    """Heuristic verdict: every diagnostic within a few null-sized tolerances."""
    return (
        abs(summary["skew"]) < 0.1
        and abs(summary["excess_kurtosis"]) < 0.2
        and summary["ks_vs_normal"] < 0.01
        and summary["row_std_cv"] < 3.0 * summary["cv_gaussian_null"]
        and summary["col_std_cv"] < 3.0 * summary["cv_gaussian_null"]
        and summary["mp_ks"] < 0.03
        and summary["n_above_edge"] <= 2
    )
