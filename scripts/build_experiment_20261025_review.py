#!/usr/bin/env python3
"""Collect and plot completed experiment P/A/B without changing raw runs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "paper_assets/05_figures/experiment_20261025"
SCORE = BASE / "matrix_score"
ABLATION = BASE / "sst2_ablation"
OUT = BASE / "P_A_B_review"
DISTS = ("iid", "dirichlet:1.0", "dirichlet:0.1")
STAGES = ("early", "middle")
VARIANTS = ("Full", "Server-Proxy", "Layer-Level", "No-Depth")
COLORS = {"Full": "#087e8b", "Server-Proxy": "#914c91",
          "Layer-Level": "#d57b29", "No-Depth": "#56677c"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def settings():
    return [(dist, stage) for stage in STAGES for dist in DISTS]


def labels():
    names = {"iid": "IID", "dirichlet:1.0": "Dirichlet 1.0",
             "dirichlet:0.1": "Dirichlet 0.1"}
    return [f"{names[dist]}\n{stage.title()}" for dist, stage in settings()]


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def plot_p():
    distance = read_csv(SCORE / "top12_jaccard_distance_summary.csv")
    paired = read_csv(SCORE / "paired_loss_reduction_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), constrained_layout=True)
    x = np.arange(6)
    rows = [next(r for r in distance if (r["distribution"], r["checkpoint"], r["method"]) ==
            (dist, stage, "Local_vs_Server-Proxy")) for dist, stage in settings()]
    axes[0].errorbar(x, [float(r["mean"]) for r in rows],
                     yerr=[float(r["std"]) for r in rows], fmt="o", color=COLORS["Full"],
                     linewidth=1.4, capsize=3, markersize=6)
    axes[0].set(ylabel="Top-12 Jaccard distance", title="P(a) Selection disagreement", ylim=(0, 0.3))
    for offset, comparison, color, marker in ((-0.09, "Local_minus_Server-Proxy", COLORS["Server-Proxy"], "o"),
                                               (0.09, "Local_minus_Local-Mean", COLORS["Layer-Level"], "s")):
        rows = [next(r for r in paired if (r["distribution"], r["checkpoint"], r["comparison"]) ==
                (dist, stage, comparison)) for dist, stage in settings()]
        axes[1].errorbar(x + offset, [float(r["mean"]) for r in rows],
                         yerr=[float(r["std"]) for r in rows], fmt=marker, color=color,
                         linewidth=1.2, capsize=2.5, markersize=5,
                         label=comparison.replace("Local_minus_", "Local - "))
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set(ylabel="Paired validation-loss reduction difference",
                title="P(b) Short-term gain difference")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_xticks(x, labels(), fontsize=7.5)
        ax.grid(axis="y", alpha=0.22)
    save(fig, "experiment_P")


def plot_a():
    summary = read_csv(SCORE / "spearman_summary.csv")
    fig, ax = plt.subplots(figsize=(10.4, 3.8), constrained_layout=True)
    x = np.arange(6)
    for offset, method, color, marker in ((-0.10, "raw", COLORS["Layer-Level"], "s"),
                                           (0.10, "depth", COLORS["Full"], "o")):
        rows = [next(r for r in summary if (r["distribution"], r["checkpoint"], r["method"]) ==
                (dist, stage, method)) for dist, stage in settings()]
        ax.errorbar(x + offset, [float(r["mean"]) for r in rows],
                    yerr=[float(r["std"]) for r in rows], fmt=marker, color=color,
                    capsize=3, linewidth=1.4, markersize=6,
                    label="Raw gradient-squared" if method == "raw" else "Depth calibrated")
    ax.set(xlabel="Data partition and checkpoint", ylabel="Within-client Spearman correlation",
           title="A  Predicted configuration utility vs. 20-step validation gain", ylim=(0.35, 1.02))
    ax.set_xticks(x, labels(), fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save(fig, "experiment_A")


def plot_b():
    runs = read_csv(ABLATION / "run_summary.csv")
    summary = read_csv(ABLATION / "three_seed_mean_std.csv")
    traces = []
    grouped = defaultdict(list)
    for run in runs:
        records = read_csv(ROOT / run["raw_result_path"])
        budget = np.asarray([float(r["Cumulative_Budget"]) for r in records])
        accuracy = np.asarray([float(r["Accuracy"]) for r in records])
        grouped[run["method"]].append((budget, accuracy))
        for r in records:
            traces.append({"run_id": run["run_id"], "dataset": "sst2",
                           "method": run["method"], "seed": run["seed"],
                           "metric": "validation_accuracy", "config_path": run["config_path"],
                           "raw_result_path": run["raw_result_path"], "round": r["Round"],
                           "cumulative_matrix_budget": r["Cumulative_Budget"],
                           "accuracy": r["Accuracy"]})
    write_csv(OUT / "B_round_curves.csv", traces)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), constrained_layout=True)
    minimum = max(min(b) for items in grouped.values() for b, _ in items)
    maximum = min(max(b) for items in grouped.values() for b, _ in items)
    grid = np.linspace(minimum, maximum, 161)
    for method in VARIANTS:
        means = np.vstack([np.interp(grid, budget, accuracy)
                           for budget, accuracy in grouped[method]]).mean(axis=0)
        axes[0].plot(grid, means, label=method, color=COLORS[method], linewidth=2,
                     linestyle="--" if method in ("Layer-Level", "No-Depth") else "-")
    axes[0].set(xlabel="Cumulative matrix-equivalent budget", ylabel="Validation accuracy",
                title="B(a) Three-seed mean trajectory")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    by_method = {r["method"]: r for r in summary}
    for offset, metric, marker in ((-0.08, "final", "o"), (0.08, "best", "s")):
        for index, method in enumerate(VARIANTS):
            r = by_method[method]
            axes[1].errorbar(index + offset, float(r[f"{metric}_accuracy_mean"]),
                             yerr=float(r[f"{metric}_accuracy_std"]), fmt=marker,
                             color=COLORS[method], capsize=3, markersize=6,
                             label=metric.title() if index == 0 else None)
    axes[1].set(ylabel="Validation accuracy, mean ± SD", title="B(b) Endpoints across seeds")
    axes[1].set_xticks(range(4), VARIANTS, rotation=12, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    save(fig, "experiment_B")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    copies = {
        "P_run_ledger.csv": SCORE / "run_ledger.csv",
        "P_selection_distance.csv": SCORE / "top12_jaccard_distance_summary.csv",
        "P_paired_gain.csv": SCORE / "paired_loss_reduction_summary.csv",
        "P_method_gain.csv": SCORE / "method_loss_reduction_summary.csv",
        "P_profiling_overhead.csv": SCORE / "profiling_overhead.csv",
        "A_spearman_cells.csv": SCORE / "client_checkpoint_spearman.csv",
        "A_spearman_summary.csv": SCORE / "spearman_summary.csv",
        "B_run_summary.csv": ABLATION / "run_summary.csv",
        "B_three_seed_summary.csv": ABLATION / "three_seed_mean_std.csv",
    }
    for name, source in copies.items():
        (OUT / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    plot_p()
    plot_a()
    plot_b()
    print(f"Wrote experiment P/A/B review pack to {OUT}")


if __name__ == "__main__":
    main()
