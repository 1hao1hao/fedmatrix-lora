#!/usr/bin/env python3
"""Summarize depth-scaling pilot runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper_assets" / "02_experiments" / "depth_scaling_pilot_summary.csv"

TRACE_ALIASES = {
    "Pure_FIM": ["Pure_FIM", "Random_FIM_NoDepth"],
    "Old_FIM_Depth": ["Old_FIM_Depth", "Random_FIM"],
    "Lite_FIM_WeakDepth": ["Lite_FIM_WeakDepth"],
    "MS": ["MS"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize depth_scaling_pilot outputs.")
    parser.add_argument("--prefix", default="depth_scaling_pilot")
    parser.add_argument("--results-root", default="evaluation_results")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=None):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else ""


def std(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return ""
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def metric_column(rows: list[dict[str, str]]) -> str:
    for col in ["Accuracy", "Macro_F1", "Rouge_l", "Rouge_L"]:
        if rows and col in rows[0]:
            return col
    return "Accuracy"


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def discover_runs(results_root: Path, prefix: str) -> list[Path]:
    runs = []
    for metadata_path in results_root.glob("*/**/metadata.json"):
        run_dir = metadata_path.parent
        metadata = load_metadata(run_dir)
        if prefix in str(metadata.get("experiment_id", "")) or prefix in str(metadata.get("run_id", run_dir.name)):
            runs.append(run_dir)
    return sorted(set(runs))


def summarize_eval(eval_path: Path) -> dict[str, str]:
    rows = read_csv(eval_path)
    if not rows:
        return {}
    metric = metric_column(rows)
    values = [to_float(row.get(metric)) for row in rows]
    values = [v for v in values if v is not None]
    final_metric = values[-1] if values else ""
    best_metric = max(values) if values else ""
    return {
        "final_metric": final_metric,
        "best_metric": best_metric,
        "final_best_gap": (best_metric - final_metric) if values else "",
        "cumulative_budget": to_float(rows[-1].get("Cumulative_Budget"), ""),
        "average_latency": mean(to_float(row.get("T_Actual")) for row in rows),
        "average_memory": mean(to_float(row.get("Avg_Memory_GB")) for row in rows),
    }


def selected_depth_stats(trace_dir: Path, method: str) -> tuple[str, str]:
    depths = []
    for trace_method in TRACE_ALIASES.get(method, [method]):
        for path in trace_dir.glob(f"selection_analysis_{trace_method}.csv"):
            for row in read_csv(path):
                if str(row.get("selected", "")).strip() not in {"1", "1.0", "true", "True"}:
                    continue
                depth = to_float(row.get("depth_norm"))
                if depth is not None:
                    depths.append(depth)
    return mean(depths), std(depths)


def decision_for_group(rows: list[dict[str, str]]) -> str:
    by_method = {row["method"]: to_float(row["best_metric"], -1.0) for row in rows}
    pure = by_method.get("Pure_FIM", -1.0)
    old = by_method.get("Old_FIM_Depth", -1.0)
    lite = by_method.get("Lite_FIM_WeakDepth", -1.0)
    ms = by_method.get("MS", -1.0)
    best = max(by_method, key=by_method.get) if by_method else ""
    if ms >= max(pure, old, lite):
        return "MS beats all FIM variants: do not claim depth-aware FIM as a core contribution."
    if lite > pure and lite > old:
        return "Lite_FIM_WeakDepth wins: use weak additive depth prior."
    if pure > old:
        return "Pure_FIM beats Old_FIM_Depth: remove/disable multiplicative depth scaling by default."
    if old >= pure and old >= lite:
        return "Old_FIM_Depth wins: keep current multiplicative depth scaling."
    return f"Mixed result; best={best}."


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in discover_runs(results_root, args.prefix):
        metadata = load_metadata(run_dir)
        trace_dir_text = metadata.get("trace_dir", "")
        trace_dir = ROOT / trace_dir_text if trace_dir_text else run_dir / "traces"
        for eval_path in sorted(run_dir.glob("eval_*.csv")):
            method = eval_path.stem.replace("eval_", "", 1)
            depth_mean, depth_std = selected_depth_stats(trace_dir, method)
            row = {
                "dataset": metadata.get("dataset", run_dir.parent.name),
                "distribution": metadata.get("distribution", ""),
                "method": method,
                "run_id": metadata.get("run_id", run_dir.name),
                "raw_result_path": str(eval_path.relative_to(ROOT)),
                "trace_dir": str(trace_dir.relative_to(ROOT)) if trace_dir.exists() else "",
                "selected_depth_mean": depth_mean,
                "selected_depth_std": depth_std,
            }
            row.update(summarize_eval(eval_path))
            rows.append(row)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["distribution"]), []).append(row)
    decisions = {
        key: decision_for_group(value)
        for key, value in grouped.items()
    }
    for row in rows:
        row["decision"] = decisions.get((row["dataset"], row["distribution"]), "")

    fieldnames = [
        "dataset", "distribution", "method", "final_metric", "best_metric",
        "final_best_gap", "cumulative_budget", "average_latency",
        "average_memory", "selected_depth_mean", "selected_depth_std",
        "decision", "run_id", "raw_result_path", "trace_dir",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)

    print(f"Wrote {len(rows)} rows to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
