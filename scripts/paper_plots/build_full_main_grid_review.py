#!/usr/bin/env python3
"""Build full-main-grid review figures for experiment diagnosis.

This is intentionally separate from the paper-ready final figure script. It
shows all conditions currently present in the full result table so weak tasks
remain visible for parameter tuning and reruns.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FULL_TABLE = ROOT / "paper_assets" / "06_tables" / "final" / "app_table_full_results.csv"
FINAL_OVERRIDES = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
OUT_DIR = ROOT / "paper_assets" / "05_figures" / "full_main_grid_review"
WEAK_CSV = ROOT / "paper_assets" / "02_experiments" / "full_main_grid_review_weak_tasks.csv"
MANIFEST = OUT_DIR / "full_main_grid_review_manifest.md"

METHOD_ORDER = [
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Ours-FIM",
]
METHOD_COLORS = {
    "FedAvg-LoRA homogeneous": "#8c8c8c",
    "Uniform allocation": "#4c78a8",
    "Fed-HeLLo": "#72b7b2",
    "LEGEND": "#54a24b",
    "MS": "#b279a2",
    "Ours-FIM": "#f58518",
}
SHORT = {
    "FedAvg-LoRA homogeneous": "FedAvg",
    "Uniform allocation": "Uniform",
}
X_COLUMNS = [
    ("Round", "round"),
    ("Cumulative_Budget", "budget"),
    ("Cumulative_Time", "time"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(value, default=np.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value) -> str:
    value = f(value)
    return "" if np.isnan(value) else f"{value:.4f}"


def short_method(method: str) -> str:
    return SHORT.get(method, method)


def method_rank(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def group_label(key: tuple[str, str]) -> str:
    return f"{key[0]} / {key[1]}"


def load_full_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv(FULL_TABLE):
        if row.get("status") not in {"complete", "legacy_complete"}:
            continue
        if row.get("metric") == "Loss":
            continue
        if row.get("method") == "Ours-Taylor":
            continue
        if row.get("dataset") in {"qqp", "mnli"} and row.get("distribution") == "dirichlet:0.1":
            continue
        rows.append(row)
    if FINAL_OVERRIDES.exists():
        override_rows = []
        for row in read_csv(FINAL_OVERRIDES):
            if row.get("placement") != "main":
                continue
            if row.get("method") == "Ours-Taylor":
                continue
            converted = {
                "dataset": row.get("dataset", ""),
                "distribution": row.get("distribution", ""),
                "method": row.get("method", ""),
                "seed": row.get("seed", ""),
                "metric": row.get("metric", ""),
                "best_metric": row.get("best_metric", ""),
                "final_metric": row.get("final_metric", ""),
                "final_cumulative_time": "",
                "final_cumulative_budget": "",
                "run_id": row.get("run_id", ""),
                "config_path": row.get("config_path", ""),
                "raw_result_path": row.get("source_raw_path", ""),
                "source": row.get("source", "final_override"),
                "status": row.get("status", "complete"),
                "notes": row.get("notes", ""),
            }
            override_rows.append(converted)
        override_keys = {(row["dataset"], row["distribution"], row["method"]) for row in override_rows}
        rows = [row for row in rows if (row["dataset"], row["distribution"], row["method"]) not in override_keys]
        rows.extend(override_rows)
    return rows


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["distribution"])].append(row)
    return {
        key: sorted(value, key=lambda row: method_rank(row["method"]))
        for key, value in sorted(grouped.items())
    }


def savefig(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")


def result_rows(row: dict[str, str]) -> list[dict[str, str]]:
    path = ROOT / row["raw_result_path"]
    if not path.exists():
        return []
    return read_csv(path)


def plot_matrix(grouped: dict[tuple[str, str], list[dict[str, str]]]) -> None:
    groups = list(grouped.keys())
    values = np.full((len(groups), len(METHOD_ORDER)), np.nan)
    for i, key in enumerate(groups):
        for row in grouped[key]:
            if row["method"] in METHOD_ORDER:
                values[i, method_rank(row["method"])] = f(row["best_metric"])

    fig, ax = plt.subplots(figsize=(12.5, max(5.0, 0.55 * len(groups) + 2.0)))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([short_method(method) for method in METHOD_ORDER], rotation=35, ha="right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([group_label(group) for group in groups])
    ax.set_title("Full main-grid review: best metric matrix")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7.5)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Best Accuracy / Rouge-L")
    fig.tight_layout()
    savefig(fig, "full_main_grid_result_matrix")
    plt.close(fig)


def plot_grouped_bars(grouped: dict[tuple[str, str], list[dict[str, str]]]) -> None:
    groups = list(grouped.keys())
    n_cols = 2
    n_rows = math.ceil(len(groups) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax, key in zip(axes.ravel(), groups):
        rows = grouped[key]
        labels = [short_method(row["method"]) for row in rows]
        vals = [f(row["best_metric"], 0.0) for row in rows]
        colors = [METHOD_COLORS.get(row["method"], "#999999") for row in rows]
        ax.bar(range(len(vals)), vals, color=colors, width=0.72)
        ax.set_title(group_label(key), fontsize=10)
        ax.set_ylim(0, max(1.0, max(vals) * 1.12 if vals else 1.0))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.suptitle("Full main-grid review: grouped best performance", y=1.01)
    fig.tight_layout()
    savefig(fig, "full_main_grid_grouped_best")
    plt.close(fig)


def plot_ours_gap(grouped: dict[tuple[str, str], list[dict[str, str]]]) -> list[dict[str, str]]:
    weak_rows = []
    groups = list(grouped.keys())
    labels = [group_label(group) for group in groups]
    fim_gaps = []
    best_methods = []
    for key in groups:
        rows = grouped[key]
        best = max(rows, key=lambda row: f(row["best_metric"], -1.0))
        best_val = f(best["best_metric"])
        best_methods.append(best["method"])
        ours_fim = next((row for row in rows if row["method"] == "Ours-FIM"), None)
        fim_gap = best_val - f(ours_fim["best_metric"]) if ours_fim else np.nan
        fim_gaps.append(fim_gap)
        weak_rows.append({
            "dataset": key[0],
            "distribution": key[1],
            "best_method": best["method"],
            "best_metric": fmt(best["best_metric"]),
            "ours_fim_best": fmt(ours_fim["best_metric"]) if ours_fim else "[TODO-RESULT]",
            "ours_fim_gap_to_best": fmt(fim_gap),
            "source_best": best.get("source", ""),
            "best_raw_result_path": best.get("raw_result_path", ""),
            "ours_fim_raw_result_path": ours_fim.get("raw_result_path", "") if ours_fim else "",
            "needs_attention": "yes" if (not np.isnan(fim_gap) and fim_gap > 0.01) else "maybe",
        })

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, fim_gaps, width=0.6, label="Ours-FIM gap", color=METHOD_COLORS["Ours-FIM"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Best method minus Ours best metric")
    ax.set_title("Full main-grid review: Ours gap to best")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "full_main_grid_ours_gap_to_best")
    plt.close(fig)

    write_csv(WEAK_CSV, weak_rows, list(weak_rows[0].keys()) if weak_rows else [])
    return weak_rows


def plot_curves(grouped: dict[tuple[str, str], list[dict[str, str]]], x_col: str, suffix: str) -> None:
    groups = list(grouped.keys())
    n_cols = 2
    n_rows = math.ceil(len(groups) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 3.6 * n_rows), squeeze=False)
    for ax, key in zip(axes.ravel(), groups):
        for row in grouped[key]:
            metric = row["metric"]
            rows = result_rows(row)
            xs = [f(item.get(x_col)) for item in rows]
            ys = [f(item.get(metric)) for item in rows]
            if xs and ys:
                ax.plot(
                    xs,
                    ys,
                    color=METHOD_COLORS.get(row["method"], "#555555"),
                    linewidth=2.0 if row["method"].startswith("Ours") else 1.5,
                    label=short_method(row["method"]),
                )
        ax.set_title(group_label(key), fontsize=10)
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(grouped[key][0]["metric"] if grouped[key] else "Metric")
        ax.grid(alpha=0.25, linewidth=0.6)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(7, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Full main-grid review: curves by {suffix}", y=1.04)
    fig.tight_layout()
    savefig(fig, f"full_main_grid_{suffix}_curves")
    plt.close(fig)


def write_manifest(grouped: dict[tuple[str, str], list[dict[str, str]]], weak_rows: list[dict[str, str]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as handle:
        handle.write("# Full Main-Grid Review Figures\n\n")
        handle.write("Purpose: diagnostic review figures after applying final-data overrides and filtering excluded stress settings. These are not filtered paper-ready main figures.\n\n")
        handle.write("Filters: Ours-Taylor rows are excluded; QQP/MNLI Dirichlet-0.1 rows are excluded; rows in `paper_assets/02_experiments/final_result_overrides.csv` replace matching dataset/distribution/method cells.\n\n")
        handle.write("## Figures\n\n")
        for name, meaning in [
            ("full_main_grid_result_matrix.png", "Best metric heatmap for every dataset/distribution/method cell."),
            ("full_main_grid_grouped_best.png", "Grouped bar charts for every full-table main-grid condition."),
            ("full_main_grid_ours_gap_to_best.png", "Gap between Ours-FIM and the best method in each condition."),
            ("full_main_grid_round_curves.png", "Per-round metric curves for every condition."),
            ("full_main_grid_budget_curves.png", "Metric vs cumulative budget curves for every condition."),
            ("full_main_grid_time_curves.png", "Metric vs cumulative time curves for every condition."),
        ]:
            handle.write(f"- `{name}`: {meaning}\n")
        handle.write("\n## Included Groups\n\n")
        for key in grouped:
            sources = sorted({row.get("source", "") for row in grouped[key]})
            handle.write(f"- `{group_label(key)}` sources={sources}\n")
        handle.write("\n## Weak/Attention Table\n\n")
        handle.write(f"- `{WEAK_CSV.relative_to(ROOT)}` lists best method, Ours gaps, raw paths, and attention flags.\n")
        handle.write(f"- attention rows: {sum(1 for row in weak_rows if row.get('needs_attention') == 'yes')}/{len(weak_rows)}\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_full_rows()
    grouped = group_rows(rows)
    plot_matrix(grouped)
    plot_grouped_bars(grouped)
    weak_rows = plot_ours_gap(grouped)
    for x_col, suffix in X_COLUMNS:
        plot_curves(grouped, x_col, suffix)
    write_manifest(grouped, weak_rows)
    print(f"Wrote full main-grid review figures to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote weak-task table to {WEAK_CSV.relative_to(ROOT)}")
    print(f"Included groups: {len(grouped)}; rows: {len(rows)}")


if __name__ == "__main__":
    main()
