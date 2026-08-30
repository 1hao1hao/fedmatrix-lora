#!/usr/bin/env python3
"""Build a metadata ledger from evaluation result directories.

The script reads generated metadata.json files and eval_*.csv outputs, then
writes a paper asset CSV. It does not modify raw experiment CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "evaluation_results"
DEFAULT_OUTPUT = ROOT / "paper_assets/02_experiments/metadata_ledger.csv"


FIELDNAMES = [
    "experiment_id",
    "run_id",
    "dataset",
    "distribution",
    "method_raw",
    "method_paper",
    "seed",
    "test_seed",
    "total_budget",
    "max_rounds",
    "local_bs",
    "local_ep",
    "lr",
    "memory_model",
    "lambda_prime",
    "enable_trace",
    "raw_result_path",
    "overhead_result_path",
    "metadata_path",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment metadata ledger.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def rows_from_metadata(metadata_path: Path) -> list[dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_dir = metadata_path.parent
    raw_methods = metadata.get("raw_methods") or []
    paper_methods = metadata.get("paper_methods") or {}
    raw_paths = metadata.get("raw_result_paths") or {}
    overhead_paths = metadata.get("overhead_result_paths") or {}

    rows = []
    for method in raw_methods:
        raw_result = ROOT / raw_paths.get(method, "")
        overhead_result = ROOT / overhead_paths.get(method, "")
        rows.append({
            "experiment_id": metadata.get("experiment_id", "[TODO-METADATA]"),
            "run_id": metadata.get("run_id", run_dir.name),
            "dataset": metadata.get("dataset", run_dir.parent.name),
            "distribution": metadata.get("distribution", "[TODO-METADATA]"),
            "method_raw": method,
            "method_paper": paper_methods.get(method, method),
            "seed": metadata.get("seed", "[TODO-SEED]"),
            "test_seed": metadata.get("test_seed", "[TODO-SEED]"),
            "total_budget": metadata.get("total_budget", "[TODO-METADATA]"),
            "max_rounds": metadata.get("max_rounds", "[TODO-METADATA]"),
            "local_bs": metadata.get("local_bs", "[TODO-METADATA]"),
            "local_ep": metadata.get("local_ep", "[TODO-METADATA]"),
            "lr": metadata.get("lr", "[TODO-METADATA]"),
            "memory_model": metadata.get("memory_model", "[TODO-METADATA]"),
            "lambda_prime": metadata.get("lambda_prime", "[TODO-METADATA]"),
            "enable_trace": metadata.get("enable_trace", "[TODO-METADATA]"),
            "raw_result_path": rel(raw_result) if raw_result.exists() else "[TODO-RESULT]",
            "overhead_result_path": rel(overhead_result) if overhead_result.exists() else "[TODO-RESULT]",
            "metadata_path": rel(metadata_path),
            "status": "complete" if raw_result.exists() else "missing_result_csv",
        })
    return rows


def rows_from_legacy_eval(eval_path: Path) -> dict:
    method = eval_path.stem.replace("eval_", "", 1)
    run_dir = eval_path.parent
    return {
        "experiment_id": "[TODO-METADATA]",
        "run_id": run_dir.name,
        "dataset": run_dir.parent.name,
        "distribution": "[TODO-METADATA]",
        "method_raw": method,
        "method_paper": method,
        "seed": "[TODO-SEED]",
        "test_seed": "[TODO-SEED]",
        "total_budget": "[TODO-METADATA]",
        "max_rounds": "[TODO-METADATA]",
        "local_bs": "[TODO-METADATA]",
        "local_ep": "[TODO-METADATA]",
        "lr": "[TODO-METADATA]",
        "memory_model": "[TODO-METADATA]",
        "lambda_prime": "[TODO-METADATA]",
        "enable_trace": "[TODO-METADATA]",
        "raw_result_path": rel(eval_path),
        "overhead_result_path": "[TODO-RESULT]",
        "metadata_path": "[TODO-METADATA]",
        "status": "legacy_missing_metadata",
    }


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    rows = []
    seen_eval_paths = set()
    for metadata_path in sorted(results_dir.glob("*/*/metadata.json")):
        metadata_rows = rows_from_metadata(metadata_path)
        rows.extend(metadata_rows)
        for row in metadata_rows:
            if row["raw_result_path"] != "[TODO-RESULT]":
                seen_eval_paths.add(row["raw_result_path"])

    for eval_path in sorted(results_dir.glob("*/*/eval_*.csv")):
        if rel(eval_path) not in seen_eval_paths:
            rows.append(rows_from_legacy_eval(eval_path))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {rel(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
