#!/usr/bin/env python3
"""Fit an MP plus log-normal mixture to Pythia Jacobian Gram spectra."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from plot_logskewnormal_fits import BINS, histogram_masses, r2_score
from plot_pythia_skewnormal_heatmap import checkpoint_paths, fit_mp, mp_bin_masses


def lognormal_bin_masses(location: float, log_scale: float) -> np.ndarray:
    """Return log-normal probability masses in the fixed eigenvalue bins."""
    return np.diff(
        norm.cdf(np.log(BINS), loc=location, scale=np.exp(log_scale))
    )


def mixture_bin_masses(parameters: np.ndarray) -> np.ndarray:
    """Return MP plus log-normal mixture masses for fitted parameters."""
    log_sigma2, q, location, log_scale, mp_weight = parameters
    mp = mp_bin_masses(float(np.exp(log_sigma2)), float(q))
    lognormal = lognormal_bin_masses(float(location), float(log_scale))
    return mp_weight * mp + (1.0 - mp_weight) * lognormal


def fit_mp_lognormal_mixture(
    eigenvalues: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Fit MP scale/aspect, log-normal location/scale, and mixture weight."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("Spectrum contains no finite positive eigenvalues.")
    target = histogram_masses(values)
    mp_r2, mp_sigma2, mp_q = fit_mp(values)
    del mp_r2
    logs = np.log(values)
    normal_location = float(logs.mean())
    normal_log_scale = float(np.log(logs.std()))
    starts = [
        np.array(
            [
                np.log(mp_sigma2),
                mp_q,
                normal_location,
                normal_log_scale,
                weight,
            ]
        )
        for weight in (0.1, 0.5, 0.9)
    ]
    best = None
    for start in starts:
        result = least_squares(
            lambda parameters: mixture_bin_masses(parameters) - target,
            start,
            bounds=(
                np.array([-30.0, 1e-3, -30.0, -8.0, 0.0]),
                np.array([10.0, 1.0, 10.0, 8.0, 1.0]),
            ),
            max_nfev=3_000,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    log_sigma2, q, location, log_scale, mp_weight = best.x
    r2 = r2_score(target, mixture_bin_masses(best.x))
    return (
        r2,
        float(np.exp(log_sigma2)),
        float(q),
        float(location),
        float(log_scale),
        float(mp_weight),
    )


def fit_checkpoint(step: int, spectra_path: Path) -> list[dict[str, float | int]]:
    """Fit the mixture to every nontrivial transport spectrum in one checkpoint."""
    rows = []
    with h5py.File(spectra_path, "r") as handle:
        for layer in range(int(handle.attrs["n_layers"]) - 1):
            values = np.asarray(handle[f"layer_{layer}/eigenvalues"], dtype=float)
            r2, sigma2, q, location, log_scale, mp_weight = (
                fit_mp_lognormal_mixture(values)
            )
            rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "r2_mp_lognormal_mixture": r2,
                    "mixture_mp_sigma2": sigma2,
                    "mixture_mp_q": q,
                    "mixture_lognormal_location": location,
                    "mixture_lognormal_log_scale": log_scale,
                    "mixture_mp_weight": mp_weight,
                }
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/pythia_mp_lognormal_mixture_grid.csv"),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Fit selected Pythia checkpoints and write their mixture-fit rows."""
    args = parse_args()
    paths = checkpoint_paths(args.results_dir)
    if args.steps is not None:
        requested = set(args.steps)
        missing = requested - paths.keys()
        if missing:
            raise ValueError(f"Missing checkpoint spectra for steps: {sorted(missing)}")
        paths = {step: paths[step] for step in sorted(requested)}
    if not paths:
        raise ValueError(f"No checkpoint spectra found in {args.results_dir}.")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fit_checkpoint, step, path): step
            for step, path in paths.items()
        }
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f"Fitted step {futures[future]}", flush=True)
    rows.sort(key=lambda row: (int(row["step"]), int(row["layer"])))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
