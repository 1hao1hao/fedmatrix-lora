#!/usr/bin/env python3
"""Aggregate and plot the CIFAR-100 matrix-position motivation rerun."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "paper_assets/02_experiments/cifar100_matrix_position_motivation"
FIGURE_ROOT = ROOT / "paper_assets/05_figures/cifar100_matrix_position_motivation_1380280"
METHOD_ORDER = [
    "shallow_qkv",
    "deep_qkv",
    "q_only",
    "k_only",
    "v_only",
    "all_qkv_reference",
]
LABELS = {
    "shallow_qkv": "First 4 blocks\nQKV (12)",
    "deep_qkv": "Last 4 blocks\nQKV (12)",
    "q_only": "All blocks\nQ only (12)",
    "k_only": "All blocks\nK only (12)",
    "v_only": "All blocks\nV only (12)",
    "all_qkv_reference": "All blocks\nQKV (36)",
}


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def load_runs():
    runs = []
    for summary_path in sorted(RESULT_ROOT.glob("*/summary.json")):
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["summary_path"] = relative(summary_path)
        summary["metadata_path"] = relative(summary_path.with_name("metadata.json"))
        runs.append(summary)
    if len(runs) != 18:
        raise RuntimeError(f"Expected 18 complete runs, found {len(runs)}")
    return runs


def write_run_registry(runs):
    fields = [
        "run_id",
        "dataset",
        "method",
        "seed",
        "metric",
        "metric_value",
        "matrix_budget",
        "trainable_parameters",
        "config_path",
        "raw_result_path",
        "metadata_path",
        "summary_path",
    ]
    path = RESULT_ROOT / "run_registry.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda row: (row["method"], row["seed"])):
            writer.writerow(
                {
                    "run_id": run["run_id"],
                    "dataset": run["dataset"],
                    "method": run["method"],
                    "seed": run["seed"],
                    "metric": "full_test_accuracy",
                    "metric_value": run["full_test_accuracy"],
                    "matrix_budget": run["matrix_budget"],
                    "trainable_parameters": run["trainable_parameters"],
                    "config_path": run["config_path"],
                    "raw_result_path": run["raw_result_path"],
                    "metadata_path": run["metadata_path"],
                    "summary_path": run["summary_path"],
                }
            )
    return path


def aggregate(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["method"]].append(run)
    rows = []
    for method in METHOD_ORDER:
        group = grouped[method]
        values = np.array([row["full_test_accuracy"] for row in group])
        rows.append(
            {
                "method": method,
                "seeds": ";".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
                "n": len(group),
                "matrix_budget": group[0]["matrix_budget"],
                "trainable_parameters": group[0]["trainable_parameters"],
                "full_test_accuracy_mean": values.mean(),
                "full_test_accuracy_sample_std": values.std(ddof=1),
                "final_validation_accuracy_mean": np.mean(
                    [row["final_validation_accuracy"] for row in group]
                ),
                "best_validation_accuracy_mean": np.mean(
                    [row["best_validation_accuracy"] for row in group]
                ),
                "elapsed_sec_mean": np.mean([row["elapsed_sec"] for row in group]),
                "run_ids": ";".join(row["run_id"] for row in group),
                "raw_result_paths": ";".join(row["raw_result_path"] for row in group),
                "config_path": group[0]["config_path"],
            }
        )
    return rows


def write_aggregate(rows):
    path = RESULT_ROOT / "aggregate_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_candidate(rows):
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    means = np.array([row["full_test_accuracy_mean"] for row in rows]) * 100
    stds = np.array([row["full_test_accuracy_sample_std"] for row in rows]) * 100
    x = np.arange(len(rows))
    colors = ["#4E9F6D", "#7AA6C2", "#5B8FF9", "#61A0A8", "#D65A4A", "#D1D5DB"]
    fig, ax = plt.subplots(figsize=(3.5, 2.48))
    for index, (mean, std, color) in enumerate(zip(means[:-1], stds[:-1], colors[:-1])):
        ax.errorbar(index, mean, yerr=std, fmt="o", markersize=6, color=color,
                    markeredgecolor="#333333", markeredgewidth=0.6, capsize=3,
                    elinewidth=1.2)
    ax.errorbar(x[-1], means[-1], yerr=stds[-1], fmt="D", markersize=5.5,
                markerfacecolor="white", markeredgecolor="#333333", capsize=3,
                color="#333333", elinewidth=1.2)
    ax.axvline(1.5, color="#A3A3A3", linewidth=0.65, linestyle=":")
    ax.axvline(4.5, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Full-test accuracy (%)", fontsize=8)
    ax.set_xticks(x, [LABELS[row["method"]] for row in rows], fontsize=5.5)
    ax.set_ylim(74, 87.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.7)
    for index, mean in enumerate(means):
        ax.text(index, mean + 0.38, f"{mean:.1f}", ha="center", va="bottom", fontsize=6.4)
    fig.tight_layout(pad=0.4)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_ROOT / f"matrix_position_fixed_budget.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic(runs):
    grouped = defaultdict(list)
    for run in runs:
        metric_path = ROOT / run["raw_result_path"]
        points = []
        with metric_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["validation_accuracy"]:
                    points.append((int(row["step"]), float(row["validation_accuracy"])))
        grouped[run["method"]].append(points)

    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    colors = dict(zip(METHOD_ORDER, ["#4E9F6D", "#7AA6C2", "#5B8FF9", "#61A0A8", "#D65A4A", "#444444"]))
    for method in METHOD_ORDER:
        curves = grouped[method]
        steps = np.array([point[0] for point in curves[0]])
        values = np.array([[point[1] for point in curve] for curve in curves])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        label = LABELS[method].replace("\n", " ")
        line_style = "--" if method == "all_qkv_reference" else "-"
        ax.plot(steps, mean * 100, label=label, color=colors[method], linewidth=1.8, linestyle=line_style)
        ax.fill_between(steps, (mean - std) * 100, (mean + std) * 100, color=colors[method], alpha=0.12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(0, 92)
    ax.grid(linestyle=":", alpha=0.65)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_ROOT / f"matrix_position_validation_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    runs = load_runs()
    rows = aggregate(runs)
    registry = write_run_registry(runs)
    summary = write_aggregate(rows)
    plot_candidate(rows)
    plot_diagnostic(runs)
    print(relative(registry))
    print(relative(summary))
    print(relative(FIGURE_ROOT))


if __name__ == "__main__":
    main()
