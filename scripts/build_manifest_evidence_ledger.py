#!/usr/bin/env python3
"""Align manifest rows with Slurm logs and result CSVs.

This script is intentionally read-only for raw logs/results. It writes derived
paper assets that can be used for plotting and claim audit.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper_assets/02_experiments/evidence_ledger.csv"
DEFAULT_GAPS = ROOT / "paper_assets/02_experiments/manifest_gap_report.csv"

CRITICAL_PATTERNS = [
    "Traceback",
    "num_samples should be",
    "Rouge 计算过程中出错",
    "CUDA out of memory",
    "实验失败",
]


@dataclass
class LogAttempt:
    manifest: str
    row: int
    experiment_id: str
    run_id: str
    out_path: Path
    err_path: Path
    success: bool
    critical_errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest evidence ledger.")
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifests",
        default=[],
        help="Manifest CSV to align. Can be provided multiple times.",
    )
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--results-dir", default=str(ROOT / "evaluation_results"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gap-output", default=str(DEFAULT_GAPS))
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def norm_manifest_path(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)
    return str(path)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_logs(logs_dir: Path) -> list[LogAttempt]:
    attempts: list[LogAttempt] = []
    for out_path in sorted(logs_dir.glob("manifest_*_*.out")):
        text = read_text(out_path)
        manifest_matches = re.findall(r"^Manifest:\s+(.+)$", text, flags=re.MULTILINE)
        row_match = re.search(r"^Row:\s+(\d+)\s+(.+)$", text, flags=re.MULTILINE)
        run_match = re.search(r"本次实验 Run ID:\s+(.+?)\s+---", text)
        if not (manifest_matches and row_match and run_match):
            continue

        err_path = out_path.with_suffix(".err")
        err_text = read_text(err_path)
        combined = text + "\n" + err_text
        errors = [pattern for pattern in CRITICAL_PATTERNS if pattern in combined]
        attempts.append(
            LogAttempt(
                manifest=norm_manifest_path(manifest_matches[-1]),
                row=int(row_match.group(1)),
                experiment_id=row_match.group(2).strip(),
                run_id=run_match.group(1).strip(),
                out_path=out_path,
                err_path=err_path,
                success="所有对比实验已顺利完成" in text,
                critical_errors=errors,
            )
        )
    return attempts


def split_methods(row: dict) -> list[str]:
    value = row.get("test_method") or row.get("method_raw") or ""
    return [part.strip() for part in value.split(",") if part.strip()]


def read_last_and_best(csv_path: Path, dataset: str) -> dict[str, str]:
    if not csv_path.exists():
        return {
            "num_rounds": "0",
            "final_metric": "[TODO-RESULT]",
            "best_metric": "[TODO-RESULT]",
            "metric_name": "[TODO-RESULT]",
            "final_loss": "[TODO-RESULT]",
            "best_loss": "[TODO-RESULT]",
            "final_cumulative_time": "[TODO-RESULT]",
            "final_cumulative_budget": "[TODO-RESULT]",
        }

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return {
            "num_rounds": "0",
            "final_metric": "[TODO-RESULT]",
            "best_metric": "[TODO-RESULT]",
            "metric_name": "[TODO-RESULT]",
            "final_loss": "[TODO-RESULT]",
            "best_loss": "[TODO-RESULT]",
            "final_cumulative_time": "[TODO-RESULT]",
            "final_cumulative_budget": "[TODO-RESULT]",
        }

    metric_name = "Rouge_l" if dataset == "samsum" else "Accuracy"

    def f(row: dict, key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    final = rows[-1]
    metrics = [f(row, metric_name) for row in rows]
    losses = [f(row, "Loss") for row in rows]
    return {
        "num_rounds": str(len(rows)),
        "final_metric": f"{f(final, metric_name):.8g}",
        "best_metric": f"{max(metrics):.8g}",
        "metric_name": metric_name,
        "final_loss": f"{f(final, 'Loss'):.8g}",
        "best_loss": f"{min(losses):.8g}",
        "final_cumulative_time": f"{f(final, 'Cumulative_Time'):.8g}",
        "final_cumulative_budget": f"{f(final, 'Cumulative_Budget'):.8g}",
    }


def choose_attempt(attempts: list[LogAttempt], methods: list[str], dataset: str, results_dir: Path) -> tuple[LogAttempt | None, str]:
    def complete_for(attempt: LogAttempt) -> bool:
        return all((results_dir / dataset / attempt.run_id / f"eval_{method}.csv").exists() for method in methods)

    valid = [
        attempt for attempt in attempts
        if attempt.success and not attempt.critical_errors and complete_for(attempt)
    ]
    if valid:
        return valid[-1], "complete"

    partial = [attempt for attempt in attempts if attempt.success and complete_for(attempt)]
    if partial:
        return partial[-1], "complete_with_log_errors"

    if attempts:
        return attempts[-1], "missing_result_or_failed"

    return None, "missing_log"


def paper_method_for(row: dict, method: str, index: int) -> str:
    paper = row.get("methods_paper") or ""
    parts = [part.strip() for part in paper.split(";") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if index < len(parts):
        return parts[index]
    return method


def main() -> int:
    args = parse_args()
    manifests = args.manifests or [
        "experiments/main_grid_manifest.csv",
        "experiments/ablation_representative_manifest.csv",
    ]
    logs_dir = Path(args.logs_dir)
    results_dir = Path(args.results_dir)
    output = Path(args.output)
    gap_output = Path(args.gap_output)
    if not logs_dir.is_absolute():
        logs_dir = ROOT / logs_dir
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    if not output.is_absolute():
        output = ROOT / output
    if not gap_output.is_absolute():
        gap_output = ROOT / gap_output

    attempts = parse_logs(logs_dir)
    attempts_by_key: dict[tuple[str, int, str], list[LogAttempt]] = {}
    attempts_by_experiment: dict[str, list[LogAttempt]] = {}
    for attempt in attempts:
        attempts_by_key.setdefault((attempt.manifest, attempt.row, attempt.experiment_id), []).append(attempt)
        attempts_by_experiment.setdefault(attempt.experiment_id, []).append(attempt)

    evidence_rows = []
    gap_rows = []
    for manifest_arg in manifests:
        manifest_path = Path(manifest_arg)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        manifest_rel = rel(manifest_path)
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
        for row_num, row in enumerate(rows, start=1):
            experiment_id = row["experiment_id"]
            dataset = row["dataset"]
            methods = split_methods(row)
            exact_attempts = attempts_by_key.get((manifest_rel, row_num, experiment_id), [])
            rerun_attempts = [
                attempt for attempt in attempts_by_experiment.get(experiment_id, [])
                if attempt not in exact_attempts
            ]
            row_attempts = exact_attempts + rerun_attempts
            chosen, row_status = choose_attempt(row_attempts, methods, dataset, results_dir)

            row_method_statuses = []
            for index, method in enumerate(methods):
                if chosen is None:
                    raw_result_path = None
                    overhead_result_path = None
                else:
                    raw_result_path = results_dir / dataset / chosen.run_id / f"eval_{method}.csv"
                    overhead_result_path = results_dir / dataset / chosen.run_id / f"overhead_{method}.csv"
                metrics = read_last_and_best(raw_result_path, dataset) if raw_result_path else read_last_and_best(Path(), dataset)
                method_status = row_status
                if raw_result_path is None or not raw_result_path.exists():
                    method_status = "missing_result_csv"
                row_method_statuses.append(method_status)
                evidence_rows.append({
                    "manifest": manifest_rel,
                    "manifest_row": row_num,
                    "experiment_id": experiment_id,
                    "dataset": dataset,
                    "distribution": row.get("distribution_arg", ""),
                    "method_raw": method,
                    "method_paper": paper_method_for(row, method, index),
                    "seed": row.get("seed", ""),
                    "total_budget": row.get("total_budget", ""),
                    "max_rounds": row.get("max_rounds", ""),
                    "local_bs": row.get("local_bs", ""),
                    "lr": row.get("lr", ""),
                    "local_ep": row.get("local_ep", ""),
                    "lambda_prime": row.get("lambda_prime", ""),
                    "memory_model": row.get("memory_model", ""),
                    "enable_trace": row.get("enable_trace", ""),
                    "run_id": chosen.run_id if chosen else "[TODO-RESULT]",
                    "raw_result_path": rel(raw_result_path) if raw_result_path and raw_result_path.exists() else "[TODO-RESULT]",
                    "overhead_result_path": rel(overhead_result_path) if overhead_result_path and overhead_result_path.exists() else "[TODO-RESULT]",
                    "log_out_path": rel(chosen.out_path) if chosen else "[TODO-RESULT]",
                    "log_err_path": rel(chosen.err_path) if chosen else "[TODO-RESULT]",
                    "log_success": str(chosen.success) if chosen else "False",
                    "critical_errors": ";".join(chosen.critical_errors) if chosen else "[TODO-RESULT]",
                    "attempt_count": len(row_attempts),
                    "status": method_status,
                    **metrics,
                })

            complete_methods = sum(status in {"complete", "complete_with_log_errors"} for status in row_method_statuses)
            gap_rows.append({
                "manifest": manifest_rel,
                "manifest_row": row_num,
                "experiment_id": experiment_id,
                "dataset": dataset,
                "distribution": row.get("distribution_arg", ""),
                "methods": ",".join(methods),
                "attempt_count": len(row_attempts),
                "complete_methods": complete_methods,
                "expected_methods": len(methods),
                "status": "complete" if complete_methods == len(methods) else "incomplete",
                "chosen_run_id": chosen.run_id if chosen else "[TODO-RESULT]",
                "log_out_path": rel(chosen.out_path) if chosen else "[TODO-RESULT]",
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(evidence_rows)

    with gap_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gap_rows)

    incomplete = [row for row in gap_rows if row["status"] != "complete"]
    print(f"Wrote {len(evidence_rows)} rows to {rel(output)}")
    print(f"Wrote {len(gap_rows)} rows to {rel(gap_output)}")
    print(f"Incomplete manifest rows: {len(incomplete)}")
    for row in incomplete:
        print(f"  {row['manifest']}:{row['manifest_row']} {row['experiment_id']} status={row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
