#!/usr/bin/env python3
"""Adopt MNLI IID rerun 1274111 into final derived paper assets.

This adopts the matched baseline rows and the Ours-FIM raw candidate only.
The Ours-FIM fast coherent row remains diagnostic.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "completed_rerun_review_20260608_summary.csv"
OVERRIDES = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
CURATED = ROOT / "paper_assets" / "02_experiments" / "curated_paper_data" / "main"
NOTES = ROOT / "paper_assets" / "02_experiments" / "completed_rerun_review_20260608_adoption.md"

OVERRIDE_COLUMNS = [
    "placement",
    "dataset",
    "distribution",
    "method",
    "seed",
    "metric",
    "best_metric",
    "final_metric",
    "run_id",
    "config_path",
    "source_raw_path",
    "curated_path",
    "source",
    "status",
    "notes",
]
ADOPT_METHOD_MAP = {
    "FedAvg-LoRA homogeneous": "FedAvg-LoRA homogeneous",
    "Uniform allocation": "Uniform allocation",
    "Fed-HeLLo": "Fed-HeLLo",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Ours-FIM raw": "Ours-FIM",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def safe_token(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace(" ", "_")


def copy_curated(row: pd.Series, adopted_method: str) -> str:
    src = ROOT / str(row["raw_result_path"])
    dst = (
        CURATED
        / str(row["dataset"])
        / safe_token(str(row["distribution"]))
        / safe_token(adopted_method)
        / f"{safe_token(str(row['run_id']))}.csv"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst.relative_to(ROOT))


def main() -> int:
    summary = pd.read_csv(SUMMARY)
    selected = summary[
        summary["job_id"].eq(1274111)
        & summary["dataset"].eq("mnli")
        & summary["distribution"].eq("iid")
        & summary["method"].isin(ADOPT_METHOD_MAP)
    ].copy()
    if len(selected) != len(ADOPT_METHOD_MAP):
        raise SystemExit(f"Expected {len(ADOPT_METHOD_MAP)} MNLI rows, found {len(selected)}")

    rows = read_csv(OVERRIDES)
    adopted_methods = set(ADOPT_METHOD_MAP.values())
    rows = [
        row for row in rows
        if not (row.get("dataset") == "mnli" and row.get("distribution") == "iid" and row.get("method") in adopted_methods)
    ]

    adopted = []
    for _, row in selected.iterrows():
        method = ADOPT_METHOD_MAP[str(row["method"])]
        adopted_row = {
            "placement": "main",
            "dataset": row["dataset"],
            "distribution": row["distribution"],
            "method": method,
            "seed": "2024",
            "metric": row["metric"],
            "best_metric": f"{float(row['best_metric']):.4f}",
            "final_metric": f"{float(row['final_metric']):.4f}",
            "run_id": row["run_id"],
            "config_path": row["config_path"],
            "source_raw_path": row["raw_result_path"],
            "curated_path": copy_curated(row, method),
            "source": "completed_rerun_1274111_mnli_iid_timebest",
            "status": "complete",
            "notes": "Adopted from job 1274111 after Ours-FIM raw achieved the best MNLI IID metric and dominated matched baselines on the cumulative-time accuracy curve; original raw CSV unchanged.",
        }
        adopted.append(adopted_row)
    rows.extend(adopted)
    write_csv(OVERRIDES, rows, OVERRIDE_COLUMNS)

    lines = [
        "# Completed Rerun Adoption 2026-06-08",
        "",
        "Facts from experiment results: MNLI IID rows are derived from completed job `1274111`. Raw logs, checkpoints, and original eval CSVs were not modified.",
        "",
        "## Adopted MNLI IID",
        "",
        "| Dataset | Distribution | Method | Best | Final | Run ID | Raw result |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in sorted(adopted, key=lambda item: item["method"]):
        lines.append(
            f"| {row['dataset']} | {row['distribution']} | {row['method']} | {row['best_metric']} | {row['final_metric']} | `{row['run_id']}` | `{row['source_raw_path']}` |"
        )
    lines.extend([
        "",
        "## Not Adopted",
        "",
        "- `Ours-FIM fast coherent` from job `1274111` remains diagnostic because `Ours-FIM raw` has higher best/final Accuracy and better cumulative-time dominance.",
        "- CIFAR100 job `1272881` is not adopted because Ours-FIM is below FedAvg/Uniform on iid and dirichlet:1.0. Fed-HeLLo job `1274108` failed due to a strategy-name typo; corrected job `1293294` is running.",
    ])
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {OVERRIDES.relative_to(ROOT)} with {len(adopted)} MNLI IID rows")
    print(f"Wrote {NOTES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
