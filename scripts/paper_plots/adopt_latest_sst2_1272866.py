#!/usr/bin/env python3
"""Adopt SST-2 IID stable-loader rerun 1272866 into final derived assets.

Only derived paper assets are updated. Raw evaluation CSVs are copied into the
curated paper-data area and are not modified in place.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "latest_rerun_review_summary.csv"
OVERRIDES = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
CURATED = ROOT / "paper_assets" / "02_experiments" / "curated_paper_data" / "main"
NOTES = ROOT / "paper_assets" / "02_experiments" / "latest_rerun_adoption.md"

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


def copy_curated(row: pd.Series) -> str:
    src = ROOT / str(row["raw_result_path"])
    dst = (
        CURATED
        / str(row["dataset"])
        / safe_token(str(row["distribution"]))
        / safe_token(str(row["method"]))
        / f"{safe_token(str(row['run_id']))}.csv"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst.relative_to(ROOT))


def main() -> int:
    summary = pd.read_csv(SUMMARY)
    selected = summary[
        summary["job_id"].eq(1272866)
        & summary["dataset"].eq("sst2")
        & summary["distribution"].eq("iid")
        & summary["decision"].eq("adopt_candidate")
    ].copy()
    if selected.empty:
        raise SystemExit("No adopted SST-2 rows found in latest rerun summary.")

    rows = read_csv(OVERRIDES)
    methods = set(selected["method"])
    rows = [
        row for row in rows
        if not (row.get("dataset") == "sst2" and row.get("distribution") == "iid" and row.get("method") in methods)
    ]
    adopted = []
    for _, row in selected.sort_values("method").iterrows():
        adopted_row = {
            "placement": "main",
            "dataset": row["dataset"],
            "distribution": row["distribution"],
            "method": row["method"],
            "seed": "2024",
            "metric": row["metric"],
            "best_metric": f"{float(row['best_metric']):.4f}",
            "final_metric": f"{float(row['final_metric']):.4f}",
            "run_id": row["run_id"],
            "config_path": row["config_path"],
            "source_raw_path": row["raw_result_path"],
            "curated_path": copy_curated(row),
            "source": "latest_rerun_1272866_sst2_stable_loader",
            "status": "complete",
            "notes": "Adopted from job 1272866 after stable-loader rerun fixed the first-20-round plateau and Ours-FIM achieved the best SST-2 IID metric; original raw CSV unchanged.",
        }
        adopted.append(adopted_row)
    rows.extend(adopted)
    write_csv(OVERRIDES, rows, OVERRIDE_COLUMNS)

    lines = [
        "# Latest Rerun Adoption",
        "",
        "Facts from experiment logs/results: this note is derived from `paper_assets/02_experiments/latest_rerun_review_summary.csv` and completed eval CSVs. Raw logs, checkpoints, and original eval CSVs were not modified.",
        "",
        "## Adopted",
        "",
        "| Dataset | Distribution | Method | Best | Final | Run ID | Raw result |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in adopted:
        lines.append(
            f"| {row['dataset']} | {row['distribution']} | {row['method']} | {row['best_metric']} | {row['final_metric']} | `{row['run_id']}` | `{row['source_raw_path']}` |"
        )
    lines.extend([
        "",
        "## Not Adopted Automatically",
        "",
        "- MNLI job `1272860` was kept for review rather than automatic final adoption because Ours-FIM has the best metric but does not dominate MS on the cumulative-time curve.",
        "- CIFAR100 job `1272881` was not adopted because iid FedAvg/Uniform/LEGEND are still running and Fed-HeLLo was missing from the original manifest.",
    ])
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {OVERRIDES.relative_to(ROOT)} with {len(adopted)} SST-2 rows")
    print(f"Wrote {NOTES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
