#!/usr/bin/env python3
"""Adopt CIFAR100 rerun 1344067 into final derived paper assets.

Only derived files under paper_assets are updated. Raw eval CSVs are not
modified.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "cifar100_anchor_basicuniform_1344067_summary.csv"
OVERRIDES = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
NOTES = ROOT / "paper_assets" / "02_experiments" / "cifar100_1344067_adoption.md"

SOURCE = "cifar100_anchor_basicuniform_1344067"
CONFIG_PATH = "experiments/rerun_cifar100_anchor_basicuniform_b7200_r180_manifest.csv"
OVERRIDE_COLUMNS = [
    "placement", "dataset", "distribution", "method", "seed", "metric",
    "best_metric", "final_metric", "run_id", "config_path", "source_raw_path",
    "curated_path", "source", "status", "notes",
]
CANONICAL_METHOD = {
    "FedAvg-LoRA basic random": "FedAvg-LoRA homogeneous",
    "Uniform matrix allocation": "Uniform allocation",
    "Ours-FIM anchor-balanced": "Ours-FIM",
}


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
        and row["decision"] == "adopt_candidate"
    ]
    if len(selected) != 6:
        raise RuntimeError(f"Expected 6 adoptable CIFAR100 rows, found {len(selected)}")

    overrides = read_csv(OVERRIDES)
    adopted_keys = {
        ("main", row["dataset"], row["distribution"], CANONICAL_METHOD[row["method"]])
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
        method = CANONICAL_METHOD[row["method"]]
        curated = copy_curated(row["raw_result_path"], row["distribution"], method, row["run_id"])
        notes = (
            "Adopted from job 1344067 for CIFAR100 after Ours-FIM anchor-balanced "
            "achieved the best and final-best Accuracy against the CIFAR100-only "
            "basic random FedAvg baseline and matrix-uniform allocation baseline. "
            "Original raw CSV unchanged."
        )
        new_rows.append({
            "placement": "main",
            "dataset": "cifar100",
            "distribution": row["distribution"],
            "method": method,
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
        "# CIFAR100 1344067 Adoption",
        "",
        "Facts from experiment results: derived from completed job `1344067` and `eval_*.csv` files. Raw logs, checkpoints, and original eval CSVs were not modified.",
        "",
        "Adopted `cifar100 / iid` and `cifar100 / dirichlet:1.0` for FedAvg-LoRA homogeneous, Uniform allocation, and Ours-FIM. Other CIFAR100 baselines without 1344067 reruns keep their existing final evidence.",
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
