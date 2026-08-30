#!/usr/bin/env python3
"""Summarize final low-budget missing-grid runs and register complete rows as overrides."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "final_missing_lowbudget_manifest.csv"
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "final_missing_lowbudget_summary.csv"
OVERRIDES = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
CURATED = ROOT / "paper_assets" / "02_experiments" / "curated_paper_data"
FAILED_RERUN = ROOT / "experiments" / "rerun_final_missing_failed_manifest.csv"

METRIC_PRIORITY = ("Accuracy", "Macro_F1", "Rouge_l", "Rouge_L")
FAILED_ROWS = {2, 4, 5, 9, 11}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_column(df: pd.DataFrame) -> str:
    for col in METRIC_PRIORITY:
        if col in df.columns:
            return col
    raise ValueError(f"No metric column found in {list(df.columns)}")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def safe(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace(":", "_")


def sacct_states(job_id: str = "1259277") -> dict[int, str]:
    try:
        proc = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,State", "-P", "-n"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {}
    states = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or "_" not in parts[0] or "." in parts[0]:
            continue
        try:
            states[int(parts[0].split("_", 1)[1])] = parts[1]
        except ValueError:
            continue
    return states


def find_metadata(row: dict[str, str]) -> Path | None:
    matches = sorted(
        (ROOT / "evaluation_results" / row["dataset"]).glob(
            f"{row['experiment_id']}__Dist-*Bgt-{row['total_budget']}_*/metadata.json"
        )
    )
    return matches[-1] if matches else None


def summarize() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    manifest = read_csv(MANIFEST)
    states = sacct_states()
    rows: list[dict[str, object]] = []
    failed_manifest_rows: list[dict[str, str]] = []
    for idx, row in enumerate(manifest, 1):
        metadata_path = find_metadata(row)
        state = states.get(idx, "FAILED" if idx in FAILED_ROWS else "[TODO-METADATA]")
        if metadata_path is None:
            status = "missing_metadata"
            failed_manifest_rows.append(row)
            rows.append({
                "manifest_row": idx,
                "dataset": row["dataset"],
                "distribution": row["distribution_arg"],
                "method": row["methods_paper"],
                "status": status,
                "slurm_state": state,
                "metric": "[TODO-RESULT]",
                "best_metric": "",
                "final_metric": "",
                "rounds": 0,
                "expected_rounds": row["max_rounds"],
                "run_id": "[TODO-RESULT]",
                "raw_result_path": "[TODO-RESULT]",
                "config_path": rel(MANIFEST),
            })
            continue
        metadata = json.load(metadata_path.open(encoding="utf-8"))
        raw_method = metadata.get("raw_methods", [row["test_method"]])[0]
        method = metadata.get("paper_methods", {}).get(raw_method, row["methods_paper"])
        raw_path = ROOT / metadata["raw_result_paths"][raw_method]
        complete_csv = raw_path.exists() and raw_path.stat().st_size > 0
        expected = int(row["max_rounds"])
        if complete_csv:
            df = pd.read_csv(raw_path)
            metric = metric_column(df)
            rounds = len(df)
            best = float(df[metric].max())
            final = float(df[metric].iloc[-1])
            final_time = float(df["Cumulative_Time"].iloc[-1]) if "Cumulative_Time" in df else ""
            final_budget = float(df["Cumulative_Budget"].iloc[-1]) if "Cumulative_Budget" in df else ""
        else:
            metric = "[TODO-RESULT]"
            rounds = 0
            best = ""
            final = ""
            final_time = ""
            final_budget = ""
        known_failed = idx in FAILED_ROWS or state == "FAILED"
        status = "complete" if (not known_failed and complete_csv and rounds >= expected) else "failed_or_incomplete"
        if status != "complete":
            failed_manifest_rows.append(row)
        rows.append({
            "manifest_row": idx,
            "dataset": metadata.get("dataset", row["dataset"]),
            "distribution": metadata.get("distribution", row["distribution_arg"]),
            "method": method,
            "status": status,
            "slurm_state": state,
            "metric": metric,
            "best_metric": best,
            "final_metric": final,
            "final_cumulative_time": final_time,
            "final_cumulative_budget": final_budget,
            "rounds": rounds,
            "expected_rounds": expected,
            "run_id": metadata["run_id"],
            "seed": metadata.get("seed", row["seed"]),
            "raw_result_path": rel(raw_path),
            "config_path": rel(MANIFEST),
        })
    return rows, failed_manifest_rows


def add_ranks(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["status"] != "complete":
            continue
        groups.setdefault((str(row["dataset"]), str(row["distribution"])), []).append(row)
    for group_rows in groups.values():
        best_max = max(float(row["best_metric"]) for row in group_rows)
        final_max = max(float(row["final_metric"]) for row in group_rows)
        best_method = next(row["method"] for row in group_rows if float(row["best_metric"]) == best_max)
        final_method = next(row["method"] for row in group_rows if float(row["final_metric"]) == final_max)
        for row in group_rows:
            row["best_rank"] = 1 + sum(float(other["best_metric"]) > float(row["best_metric"]) for other in group_rows)
            row["final_rank"] = 1 + sum(float(other["final_metric"]) > float(row["final_metric"]) for other in group_rows)
            row["gap_to_best"] = best_max - float(row["best_metric"])
            row["group_best_method"] = best_method
            row["group_final_method"] = final_method


def write_failed_manifest(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    write_csv(FAILED_RERUN, rows, columns)


def update_overrides(summary_rows: list[dict[str, object]]) -> None:
    existing = read_csv(OVERRIDES) if OVERRIDES.exists() else []
    complete_rows = [row for row in summary_rows if row["status"] == "complete"]
    new_rows = []
    for row in complete_rows:
        raw_path = ROOT / str(row["raw_result_path"])
        dst_dir = CURATED / "main" / str(row["dataset"]) / safe(str(row["distribution"])) / safe(str(row["method"]))
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{safe(str(row['run_id']))}.csv"
        shutil.copy2(raw_path, dst)
        new_rows.append({
            "placement": "main",
            "dataset": row["dataset"],
            "distribution": row["distribution"],
            "method": row["method"],
            "seed": row["seed"],
            "metric": row["metric"],
            "best_metric": f"{float(row['best_metric']):.4f}",
            "final_metric": f"{float(row['final_metric']):.4f}",
            "run_id": row["run_id"],
            "config_path": row["config_path"],
            "source_raw_path": row["raw_result_path"],
            "curated_path": rel(dst),
            "source": "final_missing_lowbudget",
            "status": "complete",
            "notes": "Final missing-grid low per-round budget run; original raw CSV unchanged.",
        })
    replacement_keys = {(row["dataset"], row["distribution"], row["method"]) for row in new_rows}
    kept = [
        row for row in existing
        if (row.get("dataset"), row.get("distribution"), row.get("method")) not in replacement_keys
    ]
    out_rows = kept + new_rows
    if out_rows:
        write_csv(OVERRIDES, out_rows, list(out_rows[0].keys()))


def main() -> int:
    rows, failed = summarize()
    add_ranks(rows)
    fields = [
        "manifest_row", "dataset", "distribution", "method", "status", "slurm_state",
        "metric", "best_metric", "final_metric", "final_cumulative_time", "final_cumulative_budget",
        "rounds", "expected_rounds", "best_rank", "final_rank", "gap_to_best",
        "group_best_method", "group_final_method", "run_id", "seed", "config_path", "raw_result_path",
    ]
    for row in rows:
        for field in fields:
            row.setdefault(field, "")
    write_csv(SUMMARY, rows, fields)
    write_failed_manifest(failed)
    update_overrides(rows)
    print(f"Wrote {rel(SUMMARY)}")
    print(f"Complete rows: {sum(row['status'] == 'complete' for row in rows)}/{len(rows)}")
    print(f"Failed/incomplete rerun manifest: {rel(FAILED_RERUN) if failed else 'none'}")
    print(f"Updated {rel(OVERRIDES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
