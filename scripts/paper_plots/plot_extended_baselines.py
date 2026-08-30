#!/usr/bin/env python3
"""Plot extended baseline evidence, including legacy backfilled QQP/MNLI/CIFAR100."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "paper_assets/02_experiments/extended_main_results_table.csv"
GAPS = ROOT / "paper_assets/02_experiments/extended_baseline_gap_report.csv"
OUT = ROOT / "paper_assets/05_figures/extended_baselines"
MANIFEST = ROOT / "paper_assets/05_figures/extended_baselines_manifest.md"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

METHOD_ORDER = [
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Ours-FIM",
    "Ours-Taylor",
    "FedRA",
    "MH",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot extended baseline figures.")
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def method_rank(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def short_method(method: str) -> str:
    return {
        "FedAvg-LoRA homogeneous": "FedAvg",
        "Uniform allocation": "Uniform",
    }.get(method, method)


def plot_text_full(rows: list[dict], fmt: str) -> Path:
    import matplotlib.pyplot as plt

    subset = [r for r in rows if r["dataset"] in {"qqp", "mnli"}]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in subset:
        grouped[(row["dataset"], row["distribution"])].append(row)
    groups = sorted(grouped)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), squeeze=False)
    for ax, group in zip(axes.ravel(), groups):
        data = sorted(grouped[group], key=lambda r: method_rank(r["method"]))
        labels = [short_method(r["method"]) for r in data]
        values = [f(r["best_metric"]) for r in data]
        colors = ["#f58518" if label.startswith("Ours") else "#4c78a8" for label in labels]
        ax.bar(range(len(values)), values, color=colors, width=0.72)
        ax.set_title(f"{group[0]} / {group[1]}")
        ax.set_ylim(0, max(1.0, max(values) * 1.12 if values else 1.0))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.suptitle("Extended QQP/MNLI baseline evidence: best Accuracy", y=1.02)
    fig.tight_layout()
    path = OUT / f"extended_qqp_mnli_full_baselines.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cifar_available(rows: list[dict], fmt: str) -> Path:
    import matplotlib.pyplot as plt

    subset = [r for r in rows if r["dataset"] == "cifar100"]
    data = sorted(subset, key=lambda r: method_rank(r["method"]))
    labels = [short_method(r["method"]) for r in data]
    values = [f(r["best_metric"]) for r in data]
    metric_labels = [r["metric"] for r in data]
    colors = ["#f58518" if "Ours" in r["method"] else "#8c8c8c" for r in data]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(range(len(values)), values, color=colors, width=0.72)
    ax.set_title("CIFAR100 legacy available evidence")
    ax.set_ylabel("Best metric (Accuracy where available; Loss for loss-only CSVs)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{label}\n{metric}" for label, metric in zip(labels, metric_labels)], fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    path = OUT / f"extended_cifar100_available_legacy.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_manifest(paths: list[Path]) -> None:
    with MANIFEST.open("w", encoding="utf-8") as handle:
        handle.write("# Extended Baseline Figure Manifest\n\n")
        handle.write("These figures include legacy backfilled evidence. Current-protocol and legacy rows are distinguished in `extended_main_results_table.csv`.\n\n")
        handle.write("| Asset | Path | Meaning |\n|---|---|---|\n")
        for path in paths:
            handle.write(f"| Extended baseline figure | `{rel(path)}` | Baseline coverage after legacy backfill. |\n")
        handle.write(f"| Extended result table | `paper_assets/02_experiments/extended_main_results_table.csv` | Current + legacy baseline evidence. |\n")
        handle.write(f"| Gap report | `paper_assets/02_experiments/extended_baseline_gap_report.csv` | Missing CIFAR100 baselines requiring rerun. |\n")


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(TABLE)
    paths = [
        plot_text_full(rows, args.format),
        plot_cifar_available(rows, args.format),
    ]
    write_manifest(paths)
    for path in paths:
        print(rel(path))
    print(rel(MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
