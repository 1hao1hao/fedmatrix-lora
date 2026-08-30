#!/usr/bin/env python3
"""Build full diagnostic ablation figures from paired ablation table."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAIR_TABLE = ROOT / "paper_assets" / "06_tables" / "final" / "table2_ablation_pairs.csv"
OUT_DIR = ROOT / "paper_assets" / "05_figures" / "ablation_diagnostics"
REPORT = ROOT / "paper_assets" / "05_figures" / "ablation_diagnostics" / "ablation_diagnostic_report.md"


COMPONENT_LABELS = {
    "C1": "Matrix granularity",
    "C2": "Local profiling",
    "C3": "Depth prior",
    "C4": "Position memory",
    "C5": "MCKP scheduling",
    "C6": "Latency penalty",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str, default=np.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def savefig(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def label(row: dict[str, str]) -> str:
    component = row["component"]
    return f"{component}\n{COMPONENT_LABELS.get(component, '')}"


def plot_metric_deltas(rows: list[dict[str, str]]) -> None:
    x = np.arange(len(rows))
    best = [f(row["metric_delta"]) for row in rows]
    final = [f(row["final_metric_delta"]) for row in rows]
    labels = [label(row) for row in rows]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    width = 0.36
    ax.bar(x - width / 2, best, width=width, label="Best metric delta", color="#f58518")
    ax.bar(x + width / 2, final, width=width, label="Final metric delta", color="#4c78a8")
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_ylabel("Full minus ablated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_title("Full ablation diagnostics: metric deltas")
    savefig(fig, "ablation_all_metric_deltas")


def plot_resource_deltas(rows: list[dict[str, str]]) -> None:
    x = np.arange(len(rows))
    time = [f(row["time_delta"]) for row in rows]
    budget = [f(row["budget_delta"]) for row in rows]
    labels = [label(row) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].bar(x, time, color="#72b7b2")
    axes[0].axhline(0, color="#333333", linewidth=0.9)
    axes[0].set_ylabel("Ablated minus full cumulative time (s)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)

    axes[1].bar(x, budget, color="#b279a2")
    axes[1].axhline(0, color="#333333", linewidth=0.9)
    axes[1].set_ylabel("Full minus ablated cumulative budget")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.suptitle("Full ablation diagnostics: resource deltas")
    fig.tight_layout()
    savefig(fig, "ablation_all_resource_deltas")


def plot_support_matrix(rows: list[dict[str, str]]) -> None:
    metrics = ["metric_delta", "final_metric_delta", "time_delta", "budget_delta"]
    values = np.array([[f(row[m]) for m in metrics] for row in rows], dtype=float)
    scaled = values.copy()
    for j in range(scaled.shape[1]):
        col = scaled[:, j]
        denom = max(abs(np.nanmin(col)), abs(np.nanmax(col)), 1e-12)
        scaled[:, j] = col / denom

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(scaled, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label(row).replace("\n", " - ") for row in rows], fontsize=8)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Best", "Final", "Time", "Budget"], fontsize=9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3g}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Column-normalized signed delta")
    ax.set_title("Ablation diagnostics: signed support matrix")
    savefig(fig, "ablation_support_matrix")


def interpretation(row: dict[str, str]) -> str:
    c = row["component"]
    best = f(row["metric_delta"])
    final = f(row["final_metric_delta"])
    time_delta = f(row["time_delta"])
    budget_delta = f(row["budget_delta"])
    if c in {"C1", "C2"}:
        return "strongly supports the component" if best > 0 and final > 0 else "mixed or weak support"
    if c == "C3":
        return "weak positive support only" if best > 0 else "does not support depth prior"
    if c == "C4":
        return "mixed: additive memory has higher best metric, but full has higher final metric" if best < 0 and final > 0 else "mixed"
    if c == "C5":
        return "supports scheduler for final stability/resource use, but not by best metric" if final > 0 else "mixed"
    if c == "C6":
        return "supports latency penalty on time if time_delta > 0; metric effect is mixed" if time_delta > 0 else "does not support latency penalty"
    return "TODO"


def write_report(rows: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ablation Diagnostic Report",
        "",
        "This restores the full C1-C6 paired ablation view. The main paper figure previously used a filtered claim-supporting view; this diagnostic view keeps negative and mixed results visible.",
        "",
        f"Source table: `{PAIR_TABLE.relative_to(ROOT)}`.",
        "",
        "| Component | Dataset | Distribution | Full | Ablated | Best delta | Final delta | Time delta | Budget delta | Interpretation | Raw evidence |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {component} {name} | {dataset} | {distribution} | {full} | {ablated} | {best:.4f} | {final:.4f} | {time:.2f} | {budget:.2f} | {interp} | `{full_path}` vs `{ab_path}` |".format(
                component=row["component"],
                name=COMPONENT_LABELS.get(row["component"], ""),
                dataset=row["dataset"],
                distribution=row["distribution"],
                full=row["full_method"],
                ablated=row["ablated_method"],
                best=f(row["metric_delta"], 0.0),
                final=f(row["final_metric_delta"], 0.0),
                time=f(row["time_delta"], 0.0),
                budget=f(row["budget_delta"], 0.0),
                interp=interpretation(row),
                full_path=row["full_raw_result_path"],
                ab_path=row["ablated_raw_result_path"],
            )
        )
    lines.extend(
        [
            "",
            "## What changed from the paper-ready Fig. 4",
            "",
            "- The paper-ready `fig4_ablation_delta` filtered the view to positive metric deltas for C1/C2/C3/C5 and positive time delta for C6.",
            "- C4 is visible in the pair table but not emphasized in the paper-ready metric panel because its best metric delta is negative.",
            "- C5 should be interpreted jointly with budget use: the adopted clean C5 pair shows both positive metric delta and much higher useful budget consumption under MCKP.",
            "- C6 is mainly a latency/resource ablation; its metric deltas are mixed, so it should be discussed as a time-efficiency result rather than a pure accuracy/Rouge result.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = read_csv(PAIR_TABLE)
    plot_metric_deltas(rows)
    plot_resource_deltas(rows)
    plot_support_matrix(rows)
    write_report(rows)
    print(f"Wrote restored ablation diagnostics to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote report to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
