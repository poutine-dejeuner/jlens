"""Spectral analysis of accumulated Jacobian Gram matrices.

Computes per-layer:
  - Eigenvalue spectrum of J̄⊤J̄
  - MP bulk fit → q_eff
  - Power-law tail exponent α
  - Spike count above MP threshold
  - Effective rank (entropy-based)
"""

import logging

import numpy as np
from scipy import optimize, stats
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


def eigen_decompose(gram: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of J̄⊤J̄, sorted descending."""
    # gram should be symmetric PSD
    eigvals = np.linalg.eigvalsh(gram)
    # Sort descending
    eigvals = eigvals[::-1]
    # Clip tiny negatives from numerical error
    eigvals = np.clip(eigvals, 0, None)
    return eigvals


def effective_rank(eigvals: np.ndarray) -> float:
    """Entropy-based effective rank: exp(-Σ p_k log p_k) where p_k = λ_k / Σλ."""
    total = eigvals.sum()
    if total == 0:
        return 0.0
    p = eigvals / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def fit_power_law_tail(
    eigvals: np.ndarray,
    tail_fraction: float = 0.3,
) -> dict:
    """Fit power-law CDF(λ) ∝ λ^α to the low-λ tail.

    Uses the smallest tail_fraction of eigenvalues.
    For a product of N free Ginibre matrices, α = 1/(N+1) ≈ 0.08 for N=12.
    A trained aligned chain gives α ≈ 0.45.

    Returns:
        Dict with 'alpha', 'lambda_min', 'lambda_max', 'r_squared', 'n_points'
    """
    n = len(eigvals)
    n_tail = max(int(n * tail_fraction), 10)
    tail = eigvals[-n_tail:]  # smallest eigenvalues

    if tail[0] <= 0:
        # Skip zero eigenvalues
        tail = tail[tail > 0]
        if len(tail) < 10:
            return {"alpha": np.nan, "n_points": len(tail)}

    # Fit: log CDF(λ) = α log λ + const
    # CDF is empirical: for sorted ascending tail, CDF(λ_i) ≈ i/n_tail
    lambda_sorted = np.sort(tail)
    cdf = np.arange(1, len(lambda_sorted) + 1) / len(lambda_sorted)

    # Fit in log-log space
    log_lambda = np.log(lambda_sorted)
    log_cdf = np.log(cdf)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_lambda, log_cdf
    )

    return {
        "alpha": float(slope),
        "lambda_min": float(lambda_sorted[0]),
        "lambda_max": float(lambda_sorted[-1]),
        "r_squared": float(r_value**2),
        "n_points": len(lambda_sorted),
    }


def marchenko_pastur_pdf(x, sigma2, q):
    """Marchenko-Pastur density at point x.

    ρ(x) = (1/(2πσ²qx)) √((λ_+ - x)(x - λ_-))
    for λ_- ≤ x ≤ λ_+, where λ_± = σ²(1 ± √q)².
    """
    if q <= 0 or sigma2 <= 0:
        return np.zeros_like(x)
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(q)) ** 2 if q <= 1 else 0.0

    density = np.zeros_like(x)
    mask = (x > 1e-15) & (x >= lambda_minus) & (x <= lambda_plus)
    x_masked = x[mask]

    if len(x_masked) == 0:
        return density

    density[mask] = np.sqrt(
        np.clip((lambda_plus - x_masked) * (x_masked - lambda_minus), 0, None)
    ) / (2 * np.pi * sigma2 * q * x_masked)

    return density


def find_spike_threshold(eigvals: np.ndarray, sigma2: float, q: float) -> float:
    """BBP threshold: λ > σ²(1 + √q) is a spike."""
    return sigma2 * (1 + np.sqrt(q)) ** 2


def fit_mp_bulk(eigvals: np.ndarray) -> dict:
    """Fit Marchenko-Pastur to the bulk of the eigenvalue distribution.

    Uses the method from the companion notebook:
    1. Fit σ² and q to minimize KS distance between empirical CDF and MP CDF.
    2. The MP CDF is the integral of the MP PDF.
    3. Exclude top eigenvalues (potential spikes) iteratively.

    Returns:
        Dict with 'sigma2', 'q', 'q_eff', 'n_spikes', 'spike_threshold',
        'ks_statistic', 'bulk_eigenvalues'
    """
    n = len(eigvals)
    eigvals_pos = eigvals[eigvals > 1e-10]

    if len(eigvals_pos) < 20:
        return {
            "sigma2": np.nan,
            "q": np.nan,
            "q_eff": np.nan,
            "n_spikes": 0,
            "spike_threshold": np.nan,
            "ks_statistic": np.nan,
        }

    # Start with all eigenvalues, iteratively remove spikes until fit converges
    best_result = None
    best_ks = np.inf
    working = eigvals_pos.copy()

    # Try progressively excluding more top eigenvalues
    max_exclude = min(n // 5, 200)  # at most 20% or 200 eigenvalues

    for n_exclude in range(0, max_exclude + 1, max(1, max_exclude // 20)):
        if n_exclude > 0:
            working = eigvals_pos[n_exclude:]

        if len(working) < 20:
            break

        # Estimate σ² and q from the working set
        # Mean of MP = σ²
        sigma2_est = float(np.mean(working))

        # For q: match the variance of MP = σ⁴ q
        var_est = float(np.var(working))
        q_est = max(var_est / (sigma2_est**2), 1e-6) if sigma2_est > 0 else 1.0

        # Compute KS statistic
        working_sorted = np.sort(working)
        empirical_cdf = np.arange(1, len(working_sorted) + 1) / len(working_sorted)

        # MP CDF at the empirical points
        mp_cdf = _mp_cdf(working_sorted, sigma2_est, q_est)

        ks = float(np.max(np.abs(empirical_cdf - mp_cdf)))

        if ks < best_ks:
            best_ks = ks
            spike_threshold = find_spike_threshold(
                eigvals_pos, sigma2_est, q_est
            )
            n_spikes = int(np.sum(eigvals_pos > spike_threshold))
            q_eff = float(n / (n - n_spikes) * q_est if n_spikes < n else q_est)
            best_result = {
                "sigma2": sigma2_est,
                "q": q_est,
                "q_eff": q_eff,
                "n_spikes": n_spikes,
                "spike_threshold": float(spike_threshold),
                "ks_statistic": float(ks),
            }

    if best_result is None:
        return {
            "sigma2": float(np.mean(eigvals_pos)),
            "q": 1.0,
            "q_eff": 1.0,
            "n_spikes": 0,
            "spike_threshold": 0.0,
            "ks_statistic": 1.0,
        }

    return best_result


def _mp_cdf(x: np.ndarray, sigma2: float, q: float) -> np.ndarray:
    """Empirical approximation of Marchenko-Pastur CDF.

    Computes by integrating the MP PDF numerically.
    """
    # Use numerical integration of the PDF
    # For efficiency, sample the PDF densely and do trapezoidal integration
    if q <= 0 or sigma2 <= 0:
        return np.zeros_like(x)

    x_sorted = np.sort(x)
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2

    # Generate dense grid for integration
    grid = np.linspace(0, lambda_plus * 1.2, 2000)
    pdf = marchenko_pastur_pdf(grid, sigma2, q)
    cdf_grid = np.cumsum(pdf) * (grid[1] - grid[0])
    cdf_grid = cdf_grid / (cdf_grid[-1] + 1e-10)  # normalize

    # Interpolate to x
    cdf_at_x = np.interp(x_sorted, grid, cdf_grid, left=0, right=1)
    return cdf_at_x


def analyze_checkpoint(stats: dict) -> dict:
    """Run full spectral analysis on accumulated stats for one checkpoint.

    Args:
        stats: Output of JacobianAccumulator.get_all_stats()

    Returns:
        Dict with per-layer spectral quantities.
    """
    n_layers = len(stats["jbar"])
    d_model = stats["jbar"][0].shape[0]

    results = {
        "n_layers": n_layers,
        "d_model": d_model,
        "n_prompts": stats["n"],
        "coherence": stats["coherence"],
        "a_coeff": stats["a_coeff"],
        "r_norm": stats["r_norm"],
        "tr_jtj_mean": stats["tr_jtj_mean"],
    }

    eigenvalues = []
    q_eff = []
    power_law_alpha = []
    n_spikes = []
    effective_ranks = []
    mp_sigma2 = []

    for layer_idx in range(n_layers):
        gram = stats["jtj_gram"][layer_idx]
        eigvals = eigen_decompose(gram)

        eigenvalues.append(eigvals)
        effective_ranks.append(effective_rank(eigvals))

        # MP fit
        mp_fit = fit_mp_bulk(eigvals)
        q_eff.append(mp_fit["q_eff"])
        n_spikes.append(mp_fit["n_spikes"])
        mp_sigma2.append(mp_fit["sigma2"])

        # Power-law tail
        pl_fit = fit_power_law_tail(eigvals)
        power_law_alpha.append(pl_fit["alpha"])

    results["eigenvalues"] = eigenvalues
    results["effective_rank"] = effective_ranks
    results["q_eff"] = q_eff
    results["n_spikes"] = n_spikes
    results["power_law_alpha"] = power_law_alpha
    results["mp_sigma2"] = mp_sigma2

    return results
