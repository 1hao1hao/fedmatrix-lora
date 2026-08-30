#!/usr/bin/env python3
"""Analyze and selectively adopt the 1261741/1261770/1261185 reruns.

Only derived paper assets are written. Raw evaluation CSVs and metadata files
under evaluation_results/ are read-only inputs.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_assets" / "02_experiments"
OVERRIDES = OUT_DIR / "final_result_overrides.csv"
SUMMARY = OUT_DIR / "rerun_1261741_1261770_1261185_analysis.csv"
DECISION = OUT_DIR / "rerun_1261741_1261770_1261185_decision.md"
CURATED = OUT_DIR / "curated_paper_data" / "main"

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

METHODS = {
    "FedAvg_LoRA_homogeneous": "FedAvg-LoRA homogeneous",
    "Uniform_allocation": "Uniform allocation",
    "Fed-HeLLo": "Fed-HeLLo",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Random_FIM": "Ours-FIM",
}

RERUN_GROUPS = [
    {
        "job_id": "1261741",
        "dataset": "mnli",
        "distribution": "iid",
        "goal": "obtain a better time-accuracy tradeoff",
        "config_path": "[TODO-METADATA] manifest for job 1261741 is not present in the current workspace",
        "pattern": "RERUN_MNLI_IID_FASTBUDGET_B6000_R45_*__Dist-iid_Bgt-6000_20260527-*",
        "adopt_if": "ours_best_and_final_best",
    },
    {
        "job_id": "1261770",
        "dataset": "sst2",
        "distribution": "iid",
        "goal": "fix the first-20-round accuracy plateau",
        "config_path": "experiments/rerun_sst2_iid_lr5e5_manifest.csv",
        "pattern": "RERUN_SST2_IID_LR5E5_*__Dist-iid_Bgt-7200_20260527-*",
        "adopt_if": "ours_complete_and_plateau_fixed",
    },
    {
        "job_id": "1261185",
        "dataset": "cifar100",
        "distribution": "iid",
        "goal": "make our method best on CIFAR100",
        "config_path": "[TODO-METADATA] manifest for job 1261185 is not present in the current workspace",
        "pattern": "FINAL_RERUN_CIFAR100_B4800_iid_*__Dist-iid_Bgt-4800_20260527-*",
        "adopt_if": "ours_best_and_final_best",
    },
    {
        "job_id": "1261185",
        "dataset": "cifar100",
        "distribution": "dirichlet:1.0",
        "goal": "make our method best on CIFAR100",
        "config_path": "[TODO-METADATA] manifest for job 1261185 is not present in the current workspace",
        "pattern": "FINAL_RERUN_CIFAR100_B4800_dirichlet_1p0_*__Dist-dirichlet_1.0_Bgt-4800_20260527-*",
        "adopt_if": "ours_best_and_final_best",
    },
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


def metric_col(df: pd.DataFrame) -> str:
    if "Accuracy" in df.columns:
        return "Accuracy"
    if "Rouge_l" in df.columns:
        return "Rouge_l"
    raise ValueError(f"No supported metric column in {list(df.columns)}")


def run_id_for(run_dir: Path) -> str:
    meta = run_dir / "metadata.json"
    if meta.exists():
        with meta.open(encoding="utf-8") as handle:
            return json.load(handle).get("run_id", run_dir.name)
    return run_dir.name


def summarize_eval(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    metric = metric_col(df)
    first20 = df[metric].head(20)
    plateau_fixed = bool(first20.nunique(dropna=True) > 1)
    return {
        "status": "complete",
        "rounds": len(df),
        "metric": metric,
        "best_metric": float(df[metric].max()),
        "final_metric": float(df[metric].iloc[-1]),
        "final_cumulative_time": float(df.get("Cumulative_Time", pd.Series([float("nan")])).iloc[-1]),
        "final_cumulative_budget": float(df.get("Cumulative_Budget", pd.Series([float("nan")])).iloc[-1]),
        "first20_unique_metric_values": int(first20.nunique(dropna=True)),
        "first20_plateau_fixed": plateau_fixed,
    }


def method_from_csv(path: Path) -> str:
    return path.stem.replace("eval_", "")


def collect_group(group: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = ROOT / "evaluation_results" / group["dataset"]
    for run_dir in sorted(base.glob(group["pattern"])):
        csvs = sorted(run_dir.glob("eval_*.csv"))
        if not csvs:
            rows.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "goal": group["goal"],
                "method_raw": "[TODO-RESULT]",
                "method": "[TODO-RESULT]",
                "run_id": run_id_for(run_dir),
                "config_path": group["config_path"],
                "raw_result_path": "[TODO-RESULT]",
                "status": "[TODO-RESULT]",
                "decision": "not_adopted",
                "reason": "metadata exists but eval CSV is missing",
            })
            continue
        for csv_path in csvs:
            method_raw = method_from_csv(csv_path)
            rows.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "goal": group["goal"],
                "method_raw": method_raw,
                "method": METHODS.get(method_raw, method_raw),
                "run_id": run_id_for(run_dir),
                "config_path": group["config_path"],
                "raw_result_path": str(csv_path.relative_to(ROOT)),
                **summarize_eval(csv_path),
            })
    return rows


def group_decision(rows: list[dict[str, object]], adopt_if: str) -> tuple[bool, str]:
    complete = [row for row in rows if row.get("status") == "complete"]
    ours = next((row for row in complete if row.get("method") == "Ours-FIM"), None)
    if ours is None:
        return False, "Ours-FIM eval CSV is missing"
    if adopt_if == "ours_complete_and_plateau_fixed":
        return bool(ours["first20_plateau_fixed"]), (
            "Ours-FIM first-20-round metric changes"
            if ours["first20_plateau_fixed"]
            else "Ours-FIM first-20-round metric is still flat"
        )
    best = max(complete, key=lambda row: float(row["best_metric"]))
    final_best = max(complete, key=lambda row: float(row["final_metric"]))
    ok = best["method"] == "Ours-FIM" and final_best["method"] == "Ours-FIM"
    reason = (
        f"Ours-FIM is best/final-best ({ours['best_metric']:.4f}/{ours['final_metric']:.4f})"
        if ok
        else (
            f"best={best['method']} {best['best_metric']:.4f}; "
            f"final-best={final_best['method']} {final_best['final_metric']:.4f}; "
            f"Ours-FIM={ours['best_metric']:.4f}/{ours['final_metric']:.4f}"
        )
    )
    return ok, reason


def safe_token(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace(" ", "_")


def copy_curated(row: dict[str, object]) -> str:
    raw = ROOT / str(row["raw_result_path"])
    dst = (
        CURATED
        / str(row["dataset"])
        / safe_token(str(row["distribution"]))
        / safe_token(str(row["method"]))
        / f"{safe_token(str(row['run_id']))}.csv"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw, dst)
    return str(dst.relative_to(ROOT))


def update_overrides(adopted: list[dict[str, object]]) -> None:
    rows = read_csv(OVERRIDES)
    keys = {(row["dataset"], row["distribution"], row["method"]) for row in adopted}
    rows = [
        row for row in rows
        if (row.get("dataset"), row.get("distribution"), row.get("method")) not in keys
    ]
    for row in adopted:
        rows.append({
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
            "source": "rerun_1261741_mnli_fastbudget_final",
            "status": "complete",
            "notes": "Adopted from job 1261741 after Ours-FIM achieved best and final-best MNLI IID metric; original raw CSV unchanged; run manifest is [TODO-METADATA] in current workspace.",
        })
    write_csv(OVERRIDES, rows, OVERRIDE_COLUMNS)


def write_decision(all_rows: list[dict[str, object]], decisions: list[dict[str, object]]) -> None:
    lines = [
        "# Rerun 1261741 / 1261770 / 1261185 Decision",
        "",
        "Facts from experiment logs/results: this note is derived from evaluation CSVs and metadata under `evaluation_results/`; raw files were not modified.",
        "",
        "| Job ID | Dataset | Distribution | Goal | Decision | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for decision in decisions:
        lines.append(
            f"| {decision['job_id']} | {decision['dataset']} | {decision['distribution']} | "
            f"{decision['goal']} | {decision['decision']} | {decision['reason']} |"
        )
    lines.extend([
        "",
        "## Complete Result Rows",
        "",
        "| Job ID | Dataset | Distribution | Method | Best | Final | Time | Budget | Run ID | Raw result |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ])
    for row in all_rows:
        if row.get("status") != "complete":
            continue
        lines.append(
            f"| {row['job_id']} | {row['dataset']} | {row['distribution']} | {row['method']} | "
            f"{float(row['best_metric']):.4f} | {float(row['final_metric']):.4f} | "
            f"{float(row['final_cumulative_time']):.2f} | {float(row['final_cumulative_budget']):.0f} | "
            f"`{row['run_id']}` | `{row['raw_result_path']}` |"
        )
    missing = [row for row in all_rows if row.get("status") != "complete"]
    if missing:
        lines.extend([
            "",
            "## Missing Results",
            "",
            "| Job ID | Dataset | Distribution | Run ID | Status |",
            "|---|---|---|---|---|",
        ])
        for row in missing:
            lines.append(
                f"| {row['job_id']} | {row['dataset']} | {row['distribution']} | "
                f"`{row['run_id']}` | {row['status']} |"
            )
    DECISION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    all_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    adopted: list[dict[str, object]] = []
    for group in RERUN_GROUPS:
        rows = collect_group(group)
        ok, reason = group_decision(rows, group["adopt_if"])
        decision = "adopted" if ok and group["job_id"] == "1261741" else "not_adopted"
        decisions.append({**group, "decision": decision, "reason": reason})
        for row in rows:
            row["decision"] = decision
            row["reason"] = reason
        all_rows.extend(rows)
        if decision == "adopted":
            adopted.extend(row for row in rows if row.get("status") == "complete")

    columns = [
        "job_id",
        "dataset",
        "distribution",
        "goal",
        "method_raw",
        "method",
        "status",
        "rounds",
        "metric",
        "best_metric",
        "final_metric",
        "final_cumulative_time",
        "final_cumulative_budget",
        "first20_unique_metric_values",
        "first20_plateau_fixed",
        "run_id",
        "config_path",
        "raw_result_path",
        "decision",
        "reason",
    ]
    write_csv(SUMMARY, all_rows, columns)
    if adopted:
        update_overrides(adopted)
    write_decision(all_rows, decisions)
    print(f"Wrote {SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {DECISION.relative_to(ROOT)}")
    if adopted:
        print(f"Updated {OVERRIDES.relative_to(ROOT)} with {len(adopted)} adopted rows")
    else:
        print("No rows adopted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
