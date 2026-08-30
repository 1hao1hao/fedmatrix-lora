#!/usr/bin/env python3
"""Summarize scoring-v2 pilot runs without modifying raw result files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper_assets" / "02_experiments" / "scoring_v2_pilot_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scoring_v2_pilot eval/trace CSVs.")
    parser.add_argument("--prefix", default="scoring_v2_pilot")
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


def trace_method_candidates(method: str) -> list[str]:
    aliases = {
        "Old_Random_FIM": ["Old_Random_FIM", "Random_FIM"],
        "Ours_FIM": ["Ours_FIM", "Random_FIM"],
        "Ours-FIM": ["Ours-FIM", "Random_FIM"],
        "Matrix_FIM": ["Matrix_FIM", "Random_FIM"],
    }
    return aliases.get(method, [method])


def metric_column(rows: list[dict[str, str]]) -> str:
    preferred = ["Accuracy", "Macro_F1", "Rouge_l", "Rouge_L", "rougeL", "Loss"]
    columns = rows[0].keys() if rows else []
    for col in preferred:
        if col in columns:
            return col
    for col in columns:
        if col.lower() not in {"round", "t_actual", "cumulative_time", "consumed_budget", "cumulative_budget"}:
            if to_float(rows[-1].get(col)) is not None:
                return col
    return ""


def summarize_eval(eval_path: Path) -> dict[str, str]:
    rows = read_csv(eval_path)
    if not rows:
        return {}
    metric = metric_column(rows)
    metric_values = [to_float(row.get(metric)) for row in rows] if metric else []
    metric_values = [v for v in metric_values if v is not None]
    final_metric = metric_values[-1] if metric_values else ""
    best_metric = max(metric_values) if metric_values else ""
    final_best_gap = (best_metric - final_metric) if metric_values else ""
    final_row = rows[-1]
    return {
        "metric": metric,
        "final_metric": final_metric,
        "best_metric": best_metric,
        "final_best_gap": final_best_gap,
        "final_cumulative_budget": to_float(final_row.get("Cumulative_Budget"), ""),
        "avg_latency": mean(to_float(row.get("T_Actual")) for row in rows),
        "avg_memory": mean(to_float(row.get("Avg_Memory_GB")) for row in rows),
        "avg_communication": mean(to_float(row.get("Avg_Comm_MB")) for row in rows),
        "rounds_recorded": len(rows),
    }


def summarize_update_cosine(trace_dir: Path, method: str) -> str:
    candidates = []
    for trace_method in trace_method_candidates(method):
        candidates.extend(trace_dir.glob(f"update_cosine_{trace_method}.csv"))
    if not candidates:
        return ""
    rows = []
    for path in candidates:
        rows.extend(read_csv(path))
    return mean(to_float(row.get("cosine_mean")) for row in rows)


def summarize_depth(trace_dir: Path, method: str) -> dict[str, str]:
    candidates = []
    for trace_method in trace_method_candidates(method):
        candidates.extend(trace_dir.glob(f"selection_analysis_{trace_method}.csv"))
    if not candidates:
        return {
            "selected_depth_mean": "",
            "selected_depth_min": "",
            "selected_depth_max": "",
            "selected_depth_count": "",
        }
    depths = []
    for path in candidates:
        for row in read_csv(path):
            if str(row.get("selected", "")).strip() not in {"1", "1.0", "true", "True"}:
                continue
            depth = to_float(row.get("depth_norm"))
            if depth is not None:
                depths.append(depth)
    return {
        "selected_depth_mean": mean(depths),
        "selected_depth_min": min(depths) if depths else "",
        "selected_depth_max": max(depths) if depths else "",
        "selected_depth_count": len(depths) if depths else "",
    }


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def discover_runs(results_root: Path, prefix: str) -> list[Path]:
    run_dirs = []
    for metadata_path in results_root.glob("*/**/metadata.json"):
        run_dir = metadata_path.parent
        metadata = load_metadata(run_dir)
        experiment_id = str(metadata.get("experiment_id", ""))
        run_id = str(metadata.get("run_id", run_dir.name))
        if prefix in experiment_id or prefix in run_id:
            run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for run_dir in discover_runs(results_root, args.prefix):
        metadata = load_metadata(run_dir)
        dataset = metadata.get("dataset", run_dir.parent.name)
        distribution = metadata.get("distribution", "")
        run_id = metadata.get("run_id", run_dir.name)
        experiment_id = metadata.get("experiment_id", "")
        seed = metadata.get("seed", "")
        trace_dir_text = metadata.get("trace_dir", "")
        trace_dir = ROOT / trace_dir_text if trace_dir_text else run_dir / "traces"

        for eval_path in sorted(run_dir.glob("eval_*.csv")):
            method = eval_path.stem.replace("eval_", "", 1)
            row = {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "dataset": dataset,
                "distribution": distribution,
                "method": method,
                "seed": seed,
                "score_utility_mode": metadata.get("score_utility_mode", ""),
                "use_score_ema": metadata.get("use_score_ema", ""),
                "adaptive_depth_mix": metadata.get("adaptive_depth_mix", ""),
                "agreement_weight": metadata.get("agreement_weight", ""),
                "config_path": "[direct main_drl.py pilot launcher]",
                "raw_result_path": str(eval_path.relative_to(ROOT)),
                "trace_dir": str(trace_dir.relative_to(ROOT)) if trace_dir.exists() else "",
            }
            row.update(summarize_eval(eval_path))
            row["mean_update_cosine"] = summarize_update_cosine(trace_dir, method) if trace_dir.exists() else ""
            row.update(summarize_depth(trace_dir, method) if trace_dir.exists() else {
                "selected_depth_mean": "",
                "selected_depth_min": "",
                "selected_depth_max": "",
                "selected_depth_count": "",
            })
            summary_rows.append(row)

    fieldnames = [
        "experiment_id", "run_id", "dataset", "distribution", "method", "seed",
        "metric", "final_metric", "best_metric", "final_best_gap",
        "final_cumulative_budget", "avg_latency", "avg_memory", "avg_communication",
        "mean_update_cosine", "selected_depth_mean", "selected_depth_min",
        "selected_depth_max", "selected_depth_count", "score_utility_mode",
        "use_score_ema", "adaptive_depth_mix", "agreement_weight", "config_path",
        "raw_result_path", "trace_dir", "rounds_recorded",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} rows to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
