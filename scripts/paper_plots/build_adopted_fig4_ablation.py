#!/usr/bin/env python3
"""Build final Fig.4 from the adopted table2 ablation pairs."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAIR_TABLE = ROOT / "paper_assets" / "06_tables" / "final" / "table2_ablation_pairs.csv"
OUT_DIR = ROOT / "paper_assets" / "05_figures" / "final"

LABELS = {
    "C1": "Matrix\nGranularity",
    "C2": "Local\nProfiling",
    "C3": "Depth\nScaling",
    "C4": "MCKP\nScheduler",
    "C5": "Latency\nPenalty",
}


def read_rows() -> list[dict[str, str]]:
    with PAIR_TABLE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = read_rows()
    metric_rows = [row for row in rows if row["component"] in {"C1", "C2", "C3", "C4"}]
    latency_rows = [row for row in rows if row["component"] == "C5"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

    metric_labels = [LABELS.get(row["component"], row["component"]) for row in metric_rows]
    metric_values = [f(row["metric_delta"]) for row in metric_rows]
    axes[0].bar(range(len(metric_rows)), metric_values, color="#f58518", width=0.68)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("Full minus ablation (best metric)")
    axes[0].set_xticks(range(len(metric_rows)))
    axes[0].set_xticklabels(metric_labels, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    for idx, val in enumerate(metric_values):
        axes[0].text(idx, val + (0.004 if val >= 0 else -0.004), f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    latency_labels = [LABELS.get(row["component"], row["component"]) for row in latency_rows]
    latency_values = [f(row["time_delta"]) for row in latency_rows]
    axes[1].bar(range(len(latency_rows)), latency_values, color="#4c78a8", width=0.55)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Ablated minus full cumulative time (s)")
    axes[1].set_xticks(range(len(latency_rows)))
    axes[1].set_xticklabels(latency_labels, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    for idx, val in enumerate(latency_values):
        axes[1].text(idx, val + (20 if val >= 0 else -20), f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    fig.suptitle("Component ablations: performance gains and latency-efficiency effect")
    fig.tight_layout()
    save(fig, "fig4_ablation_delta")
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}/fig4_ablation_delta.(png|pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
