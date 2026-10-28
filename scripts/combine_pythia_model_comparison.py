#!/usr/bin/env python3
"""Merge separately fitted Pythia model CSVs and render the best-model heatmap."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from plot_pythia_skewnormal_heatmap import plot_heatmap

MODEL_SCORES = {
    "mp": "r2_mp",
    "logskewnormal": "r2_logskewnormal",
    "semicircle": "r2_semicircle",
}


def read_rows(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    """Read one model's rows indexed by (step, layer)."""
    with path.open(newline="") as handle:
        return {
            (int(row["step"]), int(row["layer"])): row
            for row in csv.DictReader(handle)
        }


def combine_rows(paths: dict[str, Path]) -> list[dict[str, float | int | str]]:
    """Merge partial model-fit tables after verifying matching grid coverage."""
    tables = {model: read_rows(path) for model, path in paths.items()}
    keys = next(iter(tables.values())).keys()
    if any(table.keys() != keys for table in tables.values()):
        raise ValueError("Model CSVs do not cover the same (step, layer) cells.")
    rows = []
    for step, layer in sorted(keys):
        row: dict[str, float | int | str] = {"step": step, "layer": layer}
        for model, score_name in MODEL_SCORES.items():
            row[score_name] = float(tables[model][(step, layer)][score_name])
        row["best_model"] = max(MODEL_SCORES, key=lambda model: row[MODEL_SCORES[model]])
        rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODEL_SCORES:
        parser.add_argument(f"--{model}-csv", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("docs/paper/model_comparison_grid_pythia12b_logskewnormal.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/pythia_skewnormal_heatmap.png"),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Merge partial fits and write the full comparison artifacts."""
    args = parse_args()
    rows = combine_rows(
        {
            "mp": args.mp_csv,
            "logskewnormal": args.logskewnormal_csv,
            "semicircle": args.semicircle_csv,
        }
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_heatmap(rows, args.figure)
    print(f"Wrote {args.output_csv} and {args.figure}")


if __name__ == "__main__":
    main()
