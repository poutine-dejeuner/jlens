#!/usr/bin/env python3
"""Add MP plus log-normal mixture fits to an existing Pythia model grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from plot_pythia_skewnormal_heatmap import plot_heatmap


def read_rows(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    """Read fit rows indexed by checkpoint step and layer."""
    with path.open(newline="") as handle:
        return {
            (int(row["step"]), int(row["layer"])): row
            for row in csv.DictReader(handle)
        }


def add_mixture_rows(
    base_rows: dict[tuple[int, int], dict[str, str]],
    mixture_rows: dict[tuple[int, int], dict[str, str]],
) -> list[dict[str, float | int | str]]:
    """Join mixture rows to an existing three-model comparison grid."""
    if base_rows.keys() != mixture_rows.keys():
        raise ValueError("Mixture and base grids do not cover the same cells.")
    rows = []
    for step, layer in sorted(base_rows):
        row = {
            key: value if key == "best_model" else float(value)
            for key, value in base_rows[(step, layer)].items()
        }
        row["step"] = step
        row["layer"] = layer
        for key, value in mixture_rows[(step, layer)].items():
            if key not in {"step", "layer"}:
                row[key] = float(value)
        scores = {
            "mp": float(row["r2_mp"]),
            "logskewnormal": float(row["r2_logskewnormal"]),
            "semicircle": float(row["r2_semicircle"]),
            "mp_lognormal_mixture": float(row["r2_mp_lognormal_mixture"]),
        }
        row["best_model"] = max(scores, key=scores.__getitem__)
        rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--mixture-csv", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "docs/paper/model_comparison_grid_pythia12b_with_mp_lognormal_mixture.csv"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/pythia_mp_lognormal_mixture_heatmap.png"),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Add mixture scores and render the four-model best-fit heatmap."""
    args = parse_args()
    rows = add_mixture_rows(read_rows(args.base_csv), read_rows(args.mixture_csv))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_heatmap(rows, args.figure)
    print(f"Wrote {args.output_csv} and {args.figure}")


if __name__ == "__main__":
    main()
