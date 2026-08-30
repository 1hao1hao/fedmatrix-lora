#!/usr/bin/env python3
"""Aggregate and plot the fixed SST-2 matrix-position motivation study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = (
    "deep_qkv",
    "shallow_qkv",
    "q_only",
    "k_only",
    "v_only",
    "all_qkv_reference",
)
LABELS = {
    "deep_qkv": "Deep QKV\n(12)",
    "q_only": "Query only\n(12)",
    "k_only": "Key only\n(12)",
    "shallow_qkv": "Shallow QKV\n(12)",
    "v_only": "Value only\n(12)",
    "all_qkv_reference": "All QKV\n(36)",
}
COLORS = {
    "deep_qkv": "#2E6F9E",
    "q_only": "#5B8DB8",
    "k_only": "#7AA6C2",
    "shallow_qkv": "#D17B47",
    "v_only": "#54A36B",
    "all_qkv_reference": "#6B6B6B",
}
EXPECTED_SEEDS = (2024, 2025, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT
        / "paper_assets/02_experiments/sst2_matrix_position_motivation",
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        ROOT
        / "paper_assets/05_figures"
        / f"sst2_matrix_position_motivation_{args.job_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    missing = []
    for strategy in STRATEGIES:
        for seed in EXPECTED_SEEDS:
            run_id = f"SST2_MATRIX_POSITION_{strategy}_seed{seed}"
            summary_path = input_root / run_id / "summary.json"
            if not summary_path.exists():
                missing.append(str(summary_path.relative_to(ROOT)))
                continue
            with summary_path.open(encoding="utf-8") as handle:
                row = json.load(handle)
            row["_summary_path"] = str(summary_path.relative_to(ROOT))
            summaries.append(row)

    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "Missing expected summaries:\n" + "\n".join(f"- {path}" for path in missing)
        )
    if not summaries:
        raise RuntimeError(f"No completed runs found under {input_root}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        grouped[row["method"]].append(row)

    aggregate_rows = []
    for strategy in STRATEGIES:
        rows = grouped.get(strategy, [])
        values = np.asarray(
            [float(row["best_validation_accuracy"]) for row in rows], dtype=float
        )
        aggregate_rows.append(
            {
                "dataset": "sst2",
                "method": strategy,
                "matrix_budget": 36 if strategy == "all_qkv_reference" else 12,
                "metric": "best_validation_accuracy",
                "n_seeds": len(values),
                "mean": float(values.mean()) if len(values) else "",
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()) if len(values) else "",
                "max": float(values.max()) if len(values) else "",
            }
        )
    write_csv(
        output_dir / "aggregate_summary.csv",
        aggregate_rows,
        ["dataset", "method", "matrix_budget", "metric", "n_seeds", "mean", "std", "min", "max"],
    )

    registry_rows = [
        {
            "run_id": row["run_id"],
            "dataset": row["dataset"],
            "method": row["method"],
            "seed": row["seed"],
            "metric": "best_validation_accuracy",
            "metric_value": row["best_validation_accuracy"],
            "config_path": row["config_path"],
            "raw_result_path": row["raw_result_path"],
            "summary_path": row["_summary_path"],
        }
        for row in summaries
    ]
    write_csv(
        output_dir / "run_registry.csv",
        registry_rows,
        [
            "run_id",
            "dataset",
            "method",
            "seed",
            "metric",
            "metric_value",
            "config_path",
            "raw_result_path",
            "summary_path",
        ],
    )

    shallow = {
        int(row["seed"]): float(row["best_validation_accuracy"])
        for row in grouped.get("shallow_qkv", [])
    }
    deep = {
        int(row["seed"]): float(row["best_validation_accuracy"])
        for row in grouped.get("deep_qkv", [])
    }
    common_seeds = sorted(set(shallow) & set(deep))
    paired_differences = [deep[seed] - shallow[seed] for seed in common_seeds]
    decision = {
        "job_id": str(args.job_id),
        "dataset": "sst2",
        "comparison": "deep_qkv_minus_shallow_qkv",
        "seeds": common_seeds,
        "paired_differences": paired_differences,
        "mean_difference": (
            float(np.mean(paired_differences)) if paired_differences else None
        ),
        "deep_mean_exceeds_shallow_mean": (
            bool(np.mean(paired_differences) > 0) if paired_differences else None
        ),
        "deep_exceeds_shallow_for_every_seed": (
            bool(all(value > 0 for value in paired_differences))
            if paired_differences
            else None
        ),
        "missing_expected_results": missing,
    }
    with (output_dir / "deep_vs_shallow_decision.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=True)

    available_rows = [
        row for row in aggregate_rows if row["mean"] != ""
    ]
    x = np.arange(len(available_rows))
    means = np.asarray([float(row["mean"]) for row in available_rows]) * 100
    errors = np.asarray([float(row["std"]) for row in available_rows]) * 100
    methods = [row["method"] for row in available_rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.48))
    for index, (method, mean, error) in enumerate(zip(methods[:-1], means[:-1], errors[:-1])):
        ax.errorbar(
            index,
            mean,
            yerr=error,
            fmt="o",
            markersize=6,
            color=COLORS[method],
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            capsize=3,
            elinewidth=1.2,
        )
    ax.errorbar(
        x[-1],
        means[-1],
        yerr=errors[-1],
        fmt="D",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor="#333333",
        capsize=3,
        color="#333333",
        elinewidth=1.2,
    )
    lower = max(0.0, float(means.min() - max(2.5, 2.5 * errors.max())))
    upper = min(100.0, float(means.max() + max(2.0, 2.5 * errors.max())))
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Validation accuracy (%)", fontsize=8)
    ax.set_xticks(x, [LABELS[method] for method in methods], fontsize=5.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.axvline(1.5, color="#A3A3A3", linewidth=0.65, linestyle=":")
    ax.axvline(4.5, color="#777777", linewidth=0.7, linestyle="--")
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    for index, value in enumerate(means):
        ax.text(
            index,
            value + 0.035 * (upper - lower),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
        )
    fig.tight_layout(pad=0.4)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"sst2_matrix_position_best_accuracy.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    curves: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in summaries:
        metrics_path = ROOT / row["raw_result_path"]
        for metric_row in read_csv(metrics_path):
            if metric_row["validation_accuracy"] == "":
                continue
            curves[row["method"]][int(metric_row["step"])].append(
                float(metric_row["validation_accuracy"])
            )
    fig, ax = plt.subplots(figsize=(7.15, 3.45))
    for strategy in STRATEGIES:
        if strategy not in curves:
            continue
        steps = sorted(curves[strategy])
        mean_curve = np.asarray(
            [np.mean(curves[strategy][step]) for step in steps], dtype=float
        )
        std_curve = np.asarray(
            [
                np.std(curves[strategy][step], ddof=1)
                if len(curves[strategy][step]) > 1
                else 0.0
                for step in steps
            ],
            dtype=float,
        )
        label = LABELS[strategy].replace("\n", " ")
        ax.plot(steps, mean_curve, label=label, color=COLORS[strategy], linewidth=1.6)
        ax.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=COLORS[strategy],
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Validation accuracy")
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.legend(ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(pad=0.5)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"sst2_matrix_position_training_curves.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    print(json.dumps(decision, indent=2, ensure_ascii=True))
    print(f"Figures and summaries written to {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
