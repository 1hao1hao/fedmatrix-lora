#!/usr/bin/env python3
"""Plot quick paper-facing summaries from evidence/result tables."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN = ROOT / "paper_assets/02_experiments/main_results_table.csv"
DEFAULT_ABLATION = ROOT / "paper_assets/02_experiments/ablation_results_table.csv"
DEFAULT_OUT = ROOT / "paper_assets/05_figures"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evidence summary figures.")
    parser.add_argument("--main-table", default=str(DEFAULT_MAIN))
    parser.add_argument("--ablation-table", default=str(DEFAULT_ABLATION))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def short_method(name: str) -> str:
    replacements = {
        "FedAvg-LoRA homogeneous": "FedAvg",
        "Uniform allocation": "Uniform",
        "Physical-layer FIM": "Physical",
        "random/no-importance scores": "Random scores",
        "Ours-FIM w/o depth prior": "No depth",
        "Ours-FIM additive-memory DP": "Additive mem",
        "Ours-FIM lambda_prime=0": "No latency",
        "greedy value/cost scheduler": "Greedy",
        "Ours-FIM MCKP": "Ours-FIM",
    }
    return replacements.get(name, name)


def plot_main(rows: list[dict], out_dir: Path, fmt: str, metric_col: str) -> Path:
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["distribution"])].append(row)

    keys = sorted(groups.keys(), key=lambda item: (item[0], item[1]))
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        group = sorted(groups[key], key=lambda row: int(row["manifest_row"]))
        labels = [short_method(row["method"]) for row in group]
        values = [f(row[metric_col]) for row in group]
        colors = ["#4c78a8" if not label.startswith("Ours") else "#f58518" for label in labels]
        ax.bar(range(len(values)), values, color=colors, width=0.72)
        ax.set_title(f"{key[0]} / {key[1]} ({group[0]['metric']})", fontsize=10)
        ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    out_path = out_dir / f"main_grid_{metric_col}.{fmt}"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ablation(rows: list[dict], out_dir: Path, fmt: str) -> Path:
    import matplotlib.pyplot as plt

    labels = [f"C{row['manifest_row']} {short_method(row['method'])}" for row in rows]
    values = [f(row["best_metric"]) for row in rows]
    colors = ["#f58518" if "Ours" in row["method"] else "#54a24b" for row in rows]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(values)), values, color=colors, width=0.72)
    ax.set_ylabel("Best metric")
    ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out_path = out_dir / f"ablation_best_metric.{fmt}"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    main_table = Path(args.main_table)
    ablation_table = Path(args.ablation_table)
    out_dir = Path(args.out_dir)
    if not main_table.is_absolute():
        main_table = ROOT / main_table
    if not ablation_table.is_absolute():
        ablation_table = ROOT / ablation_table
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    main_rows = read_rows(main_table)
    ablation_rows = read_rows(ablation_table)
    outputs = [
        plot_main(main_rows, out_dir, args.format, "best_metric"),
        plot_main(main_rows, out_dir, args.format, "final_metric"),
        plot_ablation(ablation_rows, out_dir, args.format),
    ]
    for path in outputs:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
