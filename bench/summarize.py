#!/usr/bin/env python3
"""
Read all results/<model>/summary.json files and print a combined comparison
table across models and batch sizes.

Usage:
  python bench/summarize.py                     # reads ./results/
  python bench/summarize.py --results-dir path  # custom dir
  python bench/summarize.py --metric output_tps # default; or total_tps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Returns {model_name: [row_dict, ...]} sorted by batch_size."""
    data: dict[str, list[dict]] = {}
    for summary_file in sorted(results_dir.glob("*/summary.json")):
        model = summary_file.parent.name
        with open(summary_file) as f:
            summary = json.load(f)
        # Support both InferenceX format and minimal script format
        rows = summary.get("results", [])
        data[model] = sorted(rows, key=lambda r: r.get("batch_size", 0))
    return data


def print_table(data: dict[str, list[dict]], metric: str) -> None:
    if not data:
        print("No results found.")
        return

    # Collect all batch sizes across all models
    all_bs: list[int] = sorted({
        r["batch_size"]
        for rows in data.values()
        for r in rows
    })

    col_w = 10
    model_w = max(len(m) for m in data) + 2

    header = f"{'Model':<{model_w}}" + "".join(f"{'bs='+str(bs):>{col_w}}" for bs in all_bs)
    print(f"\n--- {metric} ---")
    print(header)
    print("-" * len(header))

    for model, rows in sorted(data.items()):
        bs_map = {r["batch_size"]: r.get(metric, 0) for r in rows}
        row = f"{model:<{model_w}}"
        for bs in all_bs:
            val = bs_map.get(bs)
            cell = f"{val:.1f}" if val is not None else "—"
            row += f"{cell:>{col_w}}"
        print(row)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TPS sweep results")
    parser.add_argument("--results-dir", default="results",
                        help="Root directory containing per-model result subdirs")
    parser.add_argument("--metric", default="output_tps",
                        choices=["output_tps", "total_tps", "request_tps",
                                 "latency_p99_s", "ttft_p99_s"],
                        help="Which metric to display in the table")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Results dir not found: {results_dir}")
        return

    data = load_results(results_dir)
    print(f"Models found: {', '.join(sorted(data))}")

    for metric in ["output_tps", "total_tps", "latency_p99_s", "ttft_p99_s"]:
        print_table(data, metric)


if __name__ == "__main__":
    main()
