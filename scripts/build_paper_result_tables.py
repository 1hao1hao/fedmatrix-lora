#!/usr/bin/env python3
"""Build compact paper result tables from the evidence ledger."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "paper_assets/02_experiments/evidence_ledger.csv"
DEFAULT_MAIN = ROOT / "paper_assets/02_experiments/main_results_table.csv"
DEFAULT_ABLATION = ROOT / "paper_assets/02_experiments/ablation_results_table.csv"
MAIN_MANIFESTS = {
    "experiments/main_grid_manifest.csv",
    "experiments/cifar100_full_baseline_manifest.csv",
    "experiments/rerun_severe_budget_depth_manifest.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper result tables.")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--main-output", default=str(DEFAULT_MAIN))
    parser.add_argument("--ablation-output", default=str(DEFAULT_ABLATION))
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt_float(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return value


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({col: row.get(col, "") for col in columns} for row in rows)


def table_row(row: dict) -> dict:
    source = "current_manifest"
    if row["manifest"] == "experiments/rerun_severe_budget_depth_manifest.csv":
        source = "current_rerun_budget_depth"
    return {
        "config_path": row["manifest"],
        "manifest_row": row["manifest_row"],
        "experiment_id": row["experiment_id"],
        "dataset": row["dataset"],
        "distribution": row["distribution"],
        "method": row["method_paper"],
        "seed": row["seed"],
        "metric": row["metric_name"],
        "best_metric": fmt_float(row["best_metric"]),
        "final_metric": fmt_float(row["final_metric"]),
        "best_loss": fmt_float(row["best_loss"]),
        "final_loss": fmt_float(row["final_loss"]),
        "rounds": row["num_rounds"],
        "final_cumulative_time": fmt_float(row["final_cumulative_time"]),
        "final_cumulative_budget": fmt_float(row["final_cumulative_budget"]),
        "run_id": row["run_id"],
        "raw_result_path": row["raw_result_path"],
        "log_out_path": row["log_out_path"],
        "source": source,
        "status": row["status"],
    }


def main() -> int:
    args = parse_args()
    ledger = Path(args.ledger)
    if not ledger.is_absolute():
        ledger = ROOT / ledger
    main_output = Path(args.main_output)
    if not main_output.is_absolute():
        main_output = ROOT / main_output
    ablation_output = Path(args.ablation_output)
    if not ablation_output.is_absolute():
        ablation_output = ROOT / ablation_output

    rows = list(csv.DictReader(ledger.open(encoding="utf-8")))
    main_rows = [table_row(row) for row in rows if row["manifest"] in MAIN_MANIFESTS]
    ablation_rows = [table_row(row) for row in rows if row["manifest"] == "experiments/ablation_representative_manifest.csv"]
    columns = [
        "config_path", "manifest_row", "experiment_id", "dataset", "distribution", "method",
        "seed", "metric", "best_metric", "final_metric", "best_loss", "final_loss",
        "rounds", "final_cumulative_time", "final_cumulative_budget",
        "run_id", "raw_result_path", "log_out_path", "source", "status",
    ]
    write_csv(main_output, main_rows, columns)
    write_csv(ablation_output, ablation_rows, columns)
    print(f"Wrote {len(main_rows)} rows to {rel(main_output)}")
    print(f"Wrote {len(ablation_rows)} rows to {rel(ablation_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
