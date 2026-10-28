#!/usr/bin/env python3
"""Fit 3-component mixture (MP + log-semicircle + log-normal) to Olmo3-32B spectra
and render overlay + heatmap figures. Intended to run on a Slurm compute node (CPU).

Pipeline (mirrors the Pythia-12B reference scripts):
  1. fit_checkpoint() per step -> results/olmo3_3mix_parts/step_<step>.csv  (one row / layer)
  2. plot_step() per selected step -> figures/olmo3_three_component_overlay_step<step>.png
  3. load_grid() over all CSVs -> figures/olmo3_three_component_mixture_composition_heatmap.png
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from fit_pythia_three_component_mixture import fit_checkpoint
from plot_logskewnormal_fits import D_MODEL
from plot_three_component_mixture_overlays import plot_step
from plot_three_component_mixture_heatmap import load_grid
from plot_pythia_skewnormal_heatmap import checkpoint_paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path("/data/jlens/olmo3-32b")
PARTS_DIR = Path("results/olmo3_3mix_parts")
FIGURES_DIR = Path("figures")
OVERLAY_STEPS = [0, 100000, 300000, 650000]


def _fit_one(step: int, spectra_path: Path, parts_dir: Path) -> str:
    # Module-level so ProcessPoolExecutor can pickle it.
    out_csv = parts_dir / f"step_{step}.csv"
    if out_csv.exists():
        return f"[fit] step {step}: CSV already exists, skipping"
    rows = fit_checkpoint(step, spectra_path)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return f"[fit] step {step}: wrote {out_csv} ({len(rows)} layers)"


def fit_all_steps(results_dir: Path, steps: list[int], parts_dir: Path) -> None:
    paths = checkpoint_paths(results_dir)
    missing = [s for s in steps if s not in paths]
    if missing:
        raise ValueError(f"Missing checkpoint spectra for steps: {missing}")
    parts_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, min(len(steps), int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_fit_one, step, paths[step], parts_dir): step for step in steps}
            for future in as_completed(futures):
                print(future.result(), flush=True)
    else:
        for step in steps:
            print(_fit_one(step, paths[step], parts_dir), flush=True)
def plot_overlays(results_dir: Path, steps: list[int], parts_dir: Path, output_dir: Path) -> None:
    paths = checkpoint_paths(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        csv_path = parts_dir / f"step_{step}.csv"
        if not csv_path.exists():
            print(f"[overlay] step {step}: CSV missing, skipping")
            continue
        rows: dict[int, dict[str, float]] = {}
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows[int(row["layer"])] = {k: float(v) for k, v in row.items()}
        out_png = output_dir / f"olmo3_three_component_overlay_step{step}.png"
        plot_step(step, paths[step], rows, out_png)
        print(f"[overlay] step {step}: wrote {out_png}", flush=True)


def plot_heatmap(csv_path: Path, output_png: Path) -> None:
    steps, layers, composition, r2 = load_grid(csv_path)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].imshow(composition, aspect="auto", origin="lower")
    axes[0].set_title("Mixture composition (red: log-semicircle; green: log-normal; blue: MP)")
    image = axes[1].imshow(r2, aspect="auto", origin="lower", cmap="viridis", vmin=r2.min(), vmax=1)
    axes[1].set_title(f"Three-component mixture fit quality (R²; λ → λ/d_model, d={D_MODEL})")
    figure.colorbar(image, ax=axes[1], label="R^2")
    ticks = np.linspace(0, len(steps) - 1, min(10, len(steps)), dtype=int)
    for axis in axes:
        axis.set_xlabel("training step")
        axis.set_ylabel("layer")
        axis.set_xticks(ticks, [f"{steps[i]:,}" for i in ticks], rotation=45)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=200)
    plt.close(figure)
    print(f"[heatmap] wrote {output_png}", flush=True)


def combine_csvs(parts_dir: Path, combined_csv: Path) -> None:
    part_files = sorted(parts_dir.glob("step_*.csv"))
    if not part_files:
        raise ValueError(f"No per-step CSVs found in {parts_dir}")
    all_rows = []
    fieldnames = None
    for pf in part_files:
        with pf.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or reader.fieldnames
            for row in reader:
                all_rows.append(row)
    all_rows.sort(key=lambda r: (int(r["step"]), int(r["layer"])))
    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    with combined_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[combine] wrote {combined_csv} ({len(all_rows)} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--parts-dir", type=Path, default=PARTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="Steps to fit (default: every available checkpoint).")
    parser.add_argument("--overlay-steps", type=int, nargs="*", default=OVERLAY_STEPS,
                        help="Steps to render overlay grids for.")
    parser.add_argument("--skip-fit", action="store_true", help="Skip fitting; use existing CSVs.")
    args = parser.parse_args()

    if args.steps is None:
        paths = checkpoint_paths(args.results_dir)
        steps = sorted(paths.keys())
    else:
        steps = args.steps

    if not args.skip_fit:
        fit_all_steps(args.results_dir, steps, args.parts_dir)

    combined_csv = args.parts_dir.parent / "olmo3_3mix_all.csv"
    combine_csvs(args.parts_dir, combined_csv)

    plot_overlays(args.results_dir, args.overlay_steps, args.parts_dir, args.figures_dir)
    plot_heatmap(combined_csv, args.figures_dir / "olmo3_three_component_mixture_composition_heatmap.png")
    print("Done.")


if __name__ == "__main__":
    main()

