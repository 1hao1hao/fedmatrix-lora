#!/usr/bin/env python3
"""Review the three-way adaptive position-calibration C3 rerun."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/ablation_c3_adaptive_position_threeway_manifest.csv"
VARIANT_LABELS = {
    "none": "FIM only",
    "fixed_deep": "Fixed deep",
    "adaptive": "Adaptive",
}
VARIANT_COLORS = {
    "none": "#6b7280",
    "fixed_deep": "#4c78a8",
    "adaptive": "#d55e00",
}
SETTING_ORDER = [
    ("cifar100", "iid"),
    ("cifar100", "dirichlet:1.0"),
    ("mnli", "dirichlet:1.0"),
]
SETTING_LABELS = {
    ("cifar100", "iid"): "CIFAR100 / IID",
    ("cifar100", "dirichlet:1.0"): "CIFAR100 / D1",
    ("mnli", "dirichlet:1.0"): "MNLI / D1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def trace_statistics(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "direction_shallow": 0,
            "direction_deep": 0,
            "direction_none": 0,
            "mean_position_weight": 0.0,
            "mean_selected_depth": 0.0,
        }
    rows = read_csv(path)
    client_rounds = {}
    selected_depths = []
    for row in rows:
        key = (row["round"], row["client_id"])
        client_rounds[key] = (
            row.get("position_direction", "none") or "none",
            float(row.get("position_weight", 0.0) or 0.0),
        )
        if int(row.get("selected", 0) or 0):
            selected_depths.append(float(row["depth_norm"]))
    directions = Counter(direction for direction, _ in client_rounds.values())
    weights = [weight for _, weight in client_rounds.values()]
    return {
        "direction_shallow": directions["shallow"],
        "direction_deep": directions["deep"],
        "direction_none": directions["none"],
        "mean_position_weight": sum(weights) / max(1, len(weights)),
        "mean_selected_depth": sum(selected_depths) / max(1, len(selected_depths)),
    }


def collect(job_id: str) -> list[dict[str, object]]:
    records = []
    for spec in read_csv(MANIFEST):
        result_root = ROOT / "evaluation_results" / spec["dataset"]
        runs = sorted(result_root.glob(f"{spec['experiment_id']}__Dist-*"))
        if len(runs) != 1:
            raise RuntimeError(
                f"Expected one run for {spec['experiment_id']}; found {len(runs)}"
            )
        run_dir = runs[0]
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        eval_path = run_dir / f"eval_{spec['method_raw']}.csv"
        rows = read_csv(eval_path)
        metric = "Accuracy"
        values = [float(row[metric]) for row in rows]
        trace_path = ROOT / spec["trace_dir"] / f"selection_analysis_{spec['method_raw']}.csv"
        record = {
            "job_id": job_id,
            "run_id": metadata["run_id"],
            "dataset": metadata["dataset"],
            "distribution": metadata["distribution"],
            "method": spec["methods_paper"],
            "method_raw": spec["method_raw"],
            "position_calibration_mode": metadata["position_calibration_mode"],
            "seed": metadata["seed"],
            "metric": metric,
            "rounds": len(rows),
            "best_metric": max(values),
            "best_round": values.index(max(values)) + 1,
            "final_metric": values[-1],
            "last10_mean": sum(values[-10:]) / min(10, len(values)),
            "final_cumulative_budget": float(rows[-1]["Cumulative_Budget"]),
            "final_cumulative_time": float(rows[-1]["Cumulative_Time"]),
            "config_path": str(MANIFEST.relative_to(ROOT)),
            "raw_result_path": str(eval_path.relative_to(ROOT)),
            "metadata_path": str(metadata_path.relative_to(ROOT)),
            "trace_path": str(trace_path.relative_to(ROOT)),
            "status": "complete" if len(rows) == int(metadata["max_rounds"]) else "incomplete",
            "rows": rows,
        }
        record.update(trace_statistics(trace_path))
        records.append(record)
    return records


def write_summary(records: list[dict[str, object]], output: Path) -> None:
    fields = [
        "job_id", "run_id", "dataset", "distribution", "method", "method_raw",
        "position_calibration_mode", "seed", "metric", "rounds", "best_metric",
        "best_round", "final_metric", "last10_mean", "final_cumulative_budget",
        "final_cumulative_time", "direction_shallow", "direction_deep",
        "direction_none", "mean_position_weight", "mean_selected_depth",
        "config_path", "raw_result_path", "metadata_path", "trace_path", "status",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def plot(records: list[dict[str, object]], output: Path) -> None:
    lookup = {
        ((str(row["dataset"]), str(row["distribution"])), str(row["position_calibration_mode"])): row
        for row in records
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
    for axis, setting in zip(axes, SETTING_ORDER):
        values = []
        for variant in VARIANT_LABELS:
            row = lookup[(setting, variant)]
            rounds = [int(item["Round"]) for item in row["rows"]]
            accuracy = [float(item["Accuracy"]) for item in row["rows"]]
            values.extend(accuracy)
            axis.plot(
                rounds,
                accuracy,
                linewidth=1.35,
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
            )
        axis.set_title(SETTING_LABELS[setting], fontsize=8.5)
        axis.set_xlabel("Round")
        axis.set_ylim(max(0.0, min(values) - 0.03), min(1.0, max(values) + 0.025))
        axis.grid(alpha=0.22, linewidth=0.45)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_decision(records: list[dict[str, object]], output: Path, job_id: str) -> None:
    lookup = {
        ((str(row["dataset"]), str(row["distribution"])), str(row["position_calibration_mode"])): row
        for row in records
    }
    lines = [f"# C3 自适应位置校准三变体任务 {job_id} 审查", ""]
    all_pass = True
    for setting in SETTING_ORDER:
        none = lookup[(setting, "none")]
        fixed = lookup[(setting, "fixed_deep")]
        adaptive = lookup[(setting, "adaptive")]
        pass_setting = (
            adaptive["status"] == "complete"
            and float(adaptive["best_metric"]) >= float(none["best_metric"])
            and float(adaptive["best_metric"]) >= float(fixed["best_metric"])
        )
        all_pass = all_pass and pass_setting
        lines.extend([
            f"## {SETTING_LABELS[setting]}",
            "",
            f"- FIM only best/final: {float(none['best_metric']):.4f}/{float(none['final_metric']):.4f}",
            f"- Fixed deep best/final: {float(fixed['best_metric']):.4f}/{float(fixed['final_metric']):.4f}",
            f"- Adaptive best/final: {float(adaptive['best_metric']):.4f}/{float(adaptive['final_metric']):.4f}",
            f"- Adaptive direction counts (shallow/deep/none): {adaptive['direction_shallow']}/{adaptive['direction_deep']}/{adaptive['direction_none']}",
            f"- Adaptive mean weight: {float(adaptive['mean_position_weight']):.4f}",
            f"- Three-way best-metric check: {'pass' if pass_setting else 'not pass'}",
            "",
        ])
    lines.extend([
        "## Overall",
        "",
        "- Automatic screen: " + ("pass" if all_pass else "not pass"),
        "- This screen does not modify final-result ledgers or paper figures; adoption requires curve and mechanism review.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = collect(args.job_id)
    summary = ROOT / f"paper_assets/02_experiments/ablation_c3_adaptive_position_{args.job_id}_summary.csv"
    decision = ROOT / f"paper_assets/02_experiments/ablation_c3_adaptive_position_{args.job_id}_decision.md"
    figure = ROOT / f"paper_assets/05_figures/ablation_c3_adaptive_position_{args.job_id}/threeway_round_curves.png"
    write_summary(records, summary)
    plot(records, figure)
    write_decision(records, decision, args.job_id)
    print(summary.relative_to(ROOT))
    print(figure.relative_to(ROOT))
    print(decision.relative_to(ROOT))


if __name__ == "__main__":
    main()
