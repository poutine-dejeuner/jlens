#!/usr/bin/env python3
"""Fit MP, log-semicircle, and log-normal mixtures to Pythia spectra."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares

from fit_pythia_mp_lognormal_mixture import lognormal_bin_masses
from plot_logsemicircle_fits import fit_log_semicircle, semicircle_cdf
from plot_logskewnormal_fits import (
    BINS,
    fit_log_normal,
    histogram_masses,
    normalized_eigenvalues,
    r2_score,
)
from plot_pythia_skewnormal_heatmap import checkpoint_paths, fit_mp, mp_bin_masses

LOWER = np.array([-30, 1e-3, -30, -8, -30, -8, -10, -10], dtype=float)
UPPER = np.array([10, 1, 10, 8, 10, 8, 10, 10], dtype=float)
WEIGHT_STARTS = ((1 / 3, 1 / 3, 1 / 3), (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))
# Extra MP shape starts: narrow, peaked spectra trap the fit at the q lower
# bound when every start reuses the single-component MP fit's q.
MP_Q_STARTS = (0.01, 0.1, 0.5)


def weights_from_logits(logits: np.ndarray) -> np.ndarray:
    """Map two unconstrained logits to three simplex weights."""
    values = np.array([logits[0], logits[1], 0.0])
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def mixture_masses(parameters: np.ndarray) -> np.ndarray:
    """Return MP, log-semicircle, and log-normal mixture bin masses."""
    log_sigma2, q, center, log_radius, location, log_scale, mp_logit, semi_logit = (
        parameters
    )
    weights = weights_from_logits(np.array([mp_logit, semi_logit]))
    mp = mp_bin_masses(float(np.exp(log_sigma2)), float(q))
    semicircle = np.diff(
        semicircle_cdf(np.log(BINS), float(center), float(np.exp(log_radius)))
    )
    lognormal = lognormal_bin_masses(float(location), float(log_scale))
    return weights[0] * mp + weights[1] * semicircle + weights[2] * lognormal


def fit_three_component_mixture(eigenvalues: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit the three component parameters and simplex mixture weights."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    target = histogram_masses(values)
    _, mp_sigma2, mp_q = fit_mp(values)
    _, semi_center, semi_radius = fit_log_semicircle(values)
    normal = fit_log_normal(values)
    base = np.array(
        [
            np.log(mp_sigma2),
            mp_q,
            semi_center,
            np.log(semi_radius),
            normal.parameters[0],
            normal.parameters[1],
            0.0,
            0.0,
        ]
    )
    starts = []
    for q in (mp_q, *MP_Q_STARTS):
        for weights in WEIGHT_STARTS:
            start = base.copy()
            start[1] = q
            start[-2:] = np.log(np.array(weights[:2]) / weights[2])
            starts.append(start)
    return refine_from_seeds(target, starts)


def refine_from_seeds(target: np.ndarray, starts: list[np.ndarray]) -> tuple[float, np.ndarray]:
    """Run bounded least squares from every start and keep the lowest cost."""
    best = None
    for start in starts:
        result = least_squares(
            lambda parameters: mixture_masses(parameters) - target,
            np.clip(start, LOWER, UPPER),
            bounds=(LOWER, UPPER),
            max_nfev=3000,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    return r2_score(target, mixture_masses(best.x)), best.x


def params_from_row(row: dict[str, float]) -> np.ndarray:
    """Invert a CSV row back to the optimizer's parameter vector."""
    with np.errstate(divide="ignore"):
        parameters = np.array([
            np.log(row["mp_sigma2"]), row["mp_q"], row["semicircle_center"],
            np.log(row["semicircle_radius"]), row["lognormal_location"],
            row["lognormal_log_scale"],
            np.log(row["mp_weight"] / row["lognormal_weight"]),
            np.log(row["semicircle_weight"] / row["lognormal_weight"]),
        ])
    return np.clip(parameters, LOWER, UPPER)


def neighbour_steps(step: int, steps: list[int]) -> list[int]:
    """Return step and its immediate neighbours in the sorted step list."""
    ordered = sorted(steps)
    index = ordered.index(step)
    return ordered[max(index - 1, 0): index + 2]


def load_seed_rows(seed_dir: Path, step: int) -> list[dict[int, dict[str, float]]]:
    """Load per-layer rows of step and its neighbours from step_<N>.csv files."""
    available = [int(path.stem.removeprefix("step_")) for path in seed_dir.glob("step_*.csv")]
    if step not in available:
        raise ValueError(f"No seed CSV for step {step} in {seed_dir}")
    seeds = []
    for neighbour in neighbour_steps(step, available):
        with (seed_dir / f"step_{neighbour}.csv").open(newline="") as handle:
            seeds.append({
                int(row["layer"]): {key: float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            })
    return seeds


def fit_checkpoint(
    step: int,
    spectra_path: Path,
    seed_rows: list[dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, float | int]]:
    """Fit all nontrivial layers in one checkpoint.

    Without seeds, runs the multistart fit. With seed_rows (per-layer rows from
    a previous pass, e.g. this step and its neighbours), restarts only from
    those optima, so the result is never worse than the step's own seed.
    """
    rows = []
    with h5py.File(spectra_path, "r") as handle:
        for layer in range(int(handle.attrs["n_layers"]) - 1):
            values = normalized_eigenvalues(handle, layer)
            if seed_rows is None:
                r2, parameters = fit_three_component_mixture(values)
            else:
                starts = [params_from_row(seed[layer]) for seed in seed_rows if layer in seed]
                r2, parameters = refine_from_seeds(histogram_masses(values), starts)
            weights = weights_from_logits(parameters[-2:])
            rows.append(
                {
                    "step": step, "layer": layer, "r2": r2,
                    "mp_weight": weights[0], "semicircle_weight": weights[1],
                    "lognormal_weight": weights[2],
                    "mp_sigma2": np.exp(parameters[0]), "mp_q": parameters[1],
                    "semicircle_center": parameters[2],
                    "semicircle_radius": np.exp(parameters[3]),
                    "lognormal_location": parameters[4],
                    "lognormal_log_scale": parameters[5],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--seed-dir", type=Path, default=None,
        help="Refine from a previous pass's step_<N>.csv optima (this step and its neighbours).",
    )
    args = parser.parse_args()
    paths = checkpoint_paths(args.results_dir)
    missing = set(args.steps) - paths.keys()
    if missing:
        raise ValueError(f"Missing checkpoint spectra: {sorted(missing)}")
    rows = []
    for step in args.steps:
        seeds = None if args.seed_dir is None else load_seed_rows(args.seed_dir, step)
        rows.extend(fit_checkpoint(step, paths[step], seeds))
        print(f"Fitted step {step}", flush=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
