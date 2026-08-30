#!/usr/bin/env python3
"""Adopt CIFAR100 supplementary baselines from job 1347206.

Only derived files under paper_assets are updated. Raw eval CSVs are not
modified.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "cifar100_complete_1344067_1347206_summary.csv"
OVERRIDES = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
NOTES = ROOT / "paper_assets" / "02_experiments" / "cifar100_othermethods_1347206_adoption.md"

SOURCE = "cifar100_anchor_othermethods_1347206"
CONFIG_PATH = "experiments/rerun_cifar100_anchor_othermethods_b7200_r180_manifest.csv"
OVERRIDE_COLUMNS = [
    "placement", "dataset", "distribution", "method", "seed", "metric",
    "best_metric", "final_metric", "run_id", "config_path", "source_raw_path",
    "curated_path", "source", "status", "notes",
]
TARGET_METHODS = {"Fed-HeLLo", "LEGEND", "MS"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fmt(value: str) -> str:
    return f"{float(value):.4f}"


def dist_dir(distribution: str) -> str:
    return distribution.replace(":", "_")


def copy_curated(raw_path: str, distribution: str, method: str, run_id: str) -> str:
    safe_method = method.replace(" ", "_").replace("/", "_")
    dest = (
        ROOT / "paper_assets" / "02_experiments" / "curated_paper_data"
        / "main" / "cifar100" / dist_dir(distribution) / safe_method / f"{run_id}.csv"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / raw_path, dest)
    return str(dest.relative_to(ROOT))


def main() -> int:
    summary = read_csv(SUMMARY)
    selected = [
        row for row in summary
        if row["dataset"] == "cifar100"
        and row["distribution"] in {"iid", "dirichlet:1.0"}
        and row["method"] in TARGET_METHODS
        and row["job_id"] == "1347206"
        and row["decision"] == "adopt_candidate"
    ]
    if len(selected) != 6:
        raise RuntimeError(f"Expected 6 adoptable CIFAR100 supplementary rows, found {len(selected)}")

    overrides = read_csv(OVERRIDES)
    adopted_keys = {
        ("main", row["dataset"], row["distribution"], row["method"])
        for row in selected
    }
    overrides = [
        row for row in overrides
        if (
            row.get("placement"),
            row.get("dataset"),
            row.get("distribution"),
            row.get("method"),
        ) not in adopted_keys
    ]

    new_rows = []
    for row in selected:
        curated = copy_curated(row["raw_result_path"], row["distribution"], row["method"], row["run_id"])
        notes = (
            "Adopted from supplementary job 1347206 to complete CIFAR100 Fed-HeLLo, "
            "LEGEND, and MS evidence under the same B7200/R180 setting as job 1344067. "
            "Original raw CSV unchanged."
        )
        new_rows.append({
            "placement": "main",
            "dataset": "cifar100",
            "distribution": row["distribution"],
            "method": row["method"],
            "seed": row["seed"],
            "metric": row["metric"],
            "best_metric": fmt(row["best_metric"]),
            "final_metric": fmt(row["final_metric"]),
            "run_id": row["run_id"],
            "config_path": CONFIG_PATH,
            "source_raw_path": row["raw_result_path"],
            "curated_path": curated,
            "source": SOURCE,
            "status": "complete",
            "notes": notes,
        })
    overrides.extend(new_rows)
    write_csv(OVERRIDES, overrides, OVERRIDE_COLUMNS)

    lines = [
        "# CIFAR100 Other Methods 1347206 Adoption",
        "",
        "Facts from experiment results: derived from completed job `1347206` and `eval_*.csv` files. Raw logs, checkpoints, and original eval CSVs were not modified.",
        "",
        "| Dataset | Distribution | Method | Best | Final | Run ID | Raw result |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in sorted(new_rows, key=lambda item: (item["distribution"], item["method"])):
        lines.append(
            f"| {row['dataset']} | {row['distribution']} | {row['method']} | "
            f"{row['best_metric']} | {row['final_metric']} | `{row['run_id']}` | `{row['source_raw_path']}` |"
        )
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {OVERRIDES.relative_to(ROOT)}")
    print(f"Wrote {NOTES.relative_to(ROOT)}")
    for row in new_rows:
        print(f"Curated {row['curated_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
