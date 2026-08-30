#!/usr/bin/env python3
"""Adopt selected rerun/pilot evidence into final derived paper assets.

This script only updates derived files under paper_assets. It does not modify
raw logs, checkpoints, or original evaluation CSV files.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FINAL_OVERRIDES = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
FINAL_RERUN_SUMMARY = ROOT / "paper_assets" / "02_experiments" / "final_lowbudget_stable_rerun_summary.csv"
DEPTH_SUMMARY = ROOT / "paper_assets" / "02_experiments" / "depth_scaling_pilot_summary.csv"
C5_SUMMARY = ROOT / "paper_assets" / "02_experiments" / "ablation_clean_c5_summary.csv"

PAIR_TABLE = ROOT / "paper_assets" / "06_tables" / "final" / "table2_ablation_pairs.csv"
APP_C3 = ROOT / "paper_assets" / "06_tables" / "final" / "app_c3_depth_scaling_pilot_pairs.csv"
APP_C5 = ROOT / "paper_assets" / "06_tables" / "final" / "app_c5_clean_scheduler_pairs.csv"
ADOPTION_NOTES = ROOT / "paper_assets" / "02_experiments" / "selected_final_result_adoption.md"

METHOD_FOR_OVERRIDE = {
    "FedAvg-LoRA": "FedAvg-LoRA homogeneous",
    "Uniform": "Uniform allocation",
    "Fed-HeLLo": "Fed-HeLLo",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Ours-FIM": "Ours-FIM",
}

OVERRIDE_COLUMNS = [
    "placement", "dataset", "distribution", "method", "seed", "metric",
    "best_metric", "final_metric", "run_id", "config_path", "source_raw_path",
    "curated_path", "source", "status", "notes",
]

PAIR_COLUMNS = [
    "component", "dataset", "distribution", "full_method", "ablated_method",
    "metric", "metric_delta", "final_metric_delta", "time_delta", "budget_delta",
    "full_seed", "ablated_seed", "full_config_path", "ablated_config_path",
    "full_run_id", "ablated_run_id", "full_raw_result_path", "ablated_raw_result_path",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def final_eval_stats(raw_path: str) -> tuple[float, float]:
    df = pd.read_csv(ROOT / raw_path)
    return float(df["Cumulative_Time"].iloc[-1]), float(df["Cumulative_Budget"].iloc[-1])


def copy_curated(raw_path: str, dataset: str, distribution: str, method: str, run_id: str) -> str:
    safe_dist = distribution.replace(":", "_").replace("/", "_")
    safe_method = method.replace(" ", "_").replace("/", "_")
    dest = (
        ROOT / "paper_assets" / "02_experiments" / "curated_paper_data"
        / "main" / dataset / safe_dist / safe_method / f"{run_id}.csv"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / raw_path, dest)
    return str(dest.relative_to(ROOT))


def adopt_qnli_iid_overrides() -> list[dict[str, object]]:
    summary = pd.read_csv(FINAL_RERUN_SUMMARY)
    selected = summary[
        summary["dataset"].eq("qnli")
        & summary["distribution"].eq("iid")
        & summary["status"].eq("complete")
    ].copy()
    rows = read_csv(FINAL_OVERRIDES)
    adopted_methods = {METHOD_FOR_OVERRIDE[m] for m in selected["method"]}
    rows = [
        row for row in rows
        if not (row.get("dataset") == "qnli" and row.get("distribution") == "iid" and row.get("method") in adopted_methods)
    ]
    new_rows = []
    for _, row in selected.iterrows():
        method = METHOD_FOR_OVERRIDE[row["method"]]
        curated_path = copy_curated(row["raw_result_path"], "qnli", "iid", method, row["run_id"])
        new_rows.append({
            "placement": "main",
            "dataset": "qnli",
            "distribution": "iid",
            "method": method,
            "seed": int(row["seed"]),
            "metric": row["metric"],
            "best_metric": fmt(row["best_metric"]),
            "final_metric": fmt(row["final_metric"]),
            "run_id": row["run_id"],
            "config_path": row["config_path"],
            "source_raw_path": row["raw_result_path"],
            "curated_path": curated_path,
            "source": "final_lowbudget_stable_rerun_1260098",
            "status": "complete",
            "notes": "User-selected final QNLI IID evidence from stable low-budget rerun; original raw CSV unchanged.",
        })
    rows.extend(new_rows)
    write_csv(FINAL_OVERRIDES, rows, OVERRIDE_COLUMNS)
    return new_rows


def build_pair_from_rows(component: str, full: pd.Series, ablated: pd.Series, full_method: str, ablated_method: str) -> dict[str, object]:
    full_time, full_budget = final_eval_stats(full["raw_result_path"])
    ab_time, ab_budget = final_eval_stats(ablated["raw_result_path"])
    return {
        "component": component,
        "dataset": full["dataset"],
        "distribution": full["distribution"],
        "full_method": full_method,
        "ablated_method": ablated_method,
        "metric": "Accuracy",
        "metric_delta": fmt(float(full["best_metric"]) - float(ablated["best_metric"])),
        "final_metric_delta": fmt(float(full["final_metric"]) - float(ablated["final_metric"])),
        "time_delta": fmt(ab_time - full_time),
        "budget_delta": fmt(full_budget - ab_budget),
        "full_seed": "2024",
        "ablated_seed": "2024",
        "full_config_path": "scripts/run_depth_scaling_pilot.sh" if component == "C3" else full["config_path"],
        "ablated_config_path": "scripts/run_depth_scaling_pilot.sh" if component == "C3" else ablated["config_path"],
        "full_run_id": full["run_id"],
        "ablated_run_id": ablated["run_id"],
        "full_raw_result_path": full["raw_result_path"],
        "ablated_raw_result_path": ablated["raw_result_path"],
    }


def adopt_ablation_pairs() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    depth = pd.read_csv(DEPTH_SUMMARY)
    depth_selected = depth[
        depth["dataset"].eq("cifar100")
        & depth["distribution"].eq("dirichlet:0.1")
        & depth["method"].isin(["Old_FIM_Depth", "Pure_FIM"])
    ].copy()
    old = depth_selected[depth_selected["method"].eq("Old_FIM_Depth")].iloc[0]
    pure = depth_selected[depth_selected["method"].eq("Pure_FIM")].iloc[0]
    c3_pair = build_pair_from_rows("C3", old, pure, "Old_FIM_Depth", "Pure_FIM")

    c5 = pd.read_csv(C5_SUMMARY)
    c5_selected = c5[
        c5["dataset"].eq("qnli")
        & c5["distribution"].eq("iid")
        & c5["method"].isin(["Ours-FIM", "Greedy"])
    ].copy()
    ours = c5_selected[c5_selected["method"].eq("Ours-FIM")].iloc[0]
    greedy = c5_selected[c5_selected["method"].eq("Greedy")].iloc[0]
    c5_pair = build_pair_from_rows("C5", ours, greedy, "Ours-FIM MCKP", "greedy value/cost scheduler")

    pairs = read_csv(PAIR_TABLE)
    pairs = [row for row in pairs if row.get("component") not in {"C3", "C5"}]
    pairs.extend([c3_pair, c5_pair])
    pairs = sorted(pairs, key=lambda row: int(str(row["component"]).replace("C", "")))
    write_csv(PAIR_TABLE, pairs, PAIR_COLUMNS)

    app_c3_rows = []
    for dataset, distribution in [("cifar100", "dirichlet:0.1"), ("mnli", "dirichlet:1.0")]:
        part = depth[
            depth["dataset"].eq(dataset)
            & depth["distribution"].eq(distribution)
            & depth["method"].isin(["Old_FIM_Depth", "Pure_FIM"])
        ]
        app_c3_rows.extend(part.to_dict("records"))
    write_csv(APP_C3, app_c3_rows, list(depth.columns))

    app_c5_rows = c5.to_dict("records")
    write_csv(APP_C5, app_c5_rows, list(c5.columns))
    return c3_pair, c5_pair, app_c3_rows, app_c5_rows


def write_notes(qnli_rows: list[dict[str, object]], c3_pair: dict[str, object], c5_pair: dict[str, object]) -> None:
    lines = [
        "# Selected Final Result Adoption",
        "",
        "This file records user-selected derived evidence adopted into final figures/tables. Raw logs, checkpoints, and original eval CSV files were not modified.",
        "",
        "## Main-result override",
        "",
        "- Adopted `qnli / iid` rows from `paper_assets/02_experiments/final_lowbudget_stable_rerun_summary.csv` into `paper_assets/02_experiments/final_result_overrides.csv`.",
        "- Source Slurm job: `1260098`.",
        "",
        "| Dataset | Distribution | Method | Best | Final | Run ID | Raw result |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in sorted(qnli_rows, key=lambda r: r["method"]):
        lines.append(
            f"| {row['dataset']} | {row['distribution']} | {row['method']} | {row['best_metric']} | {row['final_metric']} | `{row['run_id']}` | `{row['source_raw_path']}` |"
        )
    lines.extend([
        "",
        "## Ablation replacements",
        "",
        "- C3 now uses Depth Scaling Pilot `Old_FIM_Depth` vs `Pure_FIM` on `cifar100 / dirichlet:0.1`.",
        "- C5 now uses clean scheduler ablation `Ours-FIM MCKP` vs `greedy value/cost scheduler` on `qnli / iid`.",
        "",
        "| Component | Dataset | Distribution | Full | Ablated | Best delta | Final delta | Full raw | Ablated raw |",
        "|---|---|---|---|---|---:|---:|---|---|",
        f"| C3 | {c3_pair['dataset']} | {c3_pair['distribution']} | {c3_pair['full_method']} | {c3_pair['ablated_method']} | {c3_pair['metric_delta']} | {c3_pair['final_metric_delta']} | `{c3_pair['full_raw_result_path']}` | `{c3_pair['ablated_raw_result_path']}` |",
        f"| C5 | {c5_pair['dataset']} | {c5_pair['distribution']} | {c5_pair['full_method']} | {c5_pair['ablated_method']} | {c5_pair['metric_delta']} | {c5_pair['final_metric_delta']} | `{c5_pair['full_raw_result_path']}` | `{c5_pair['ablated_raw_result_path']}` |",
        "",
        "Appendix support tables:",
        "",
        f"- `{APP_C3.relative_to(ROOT)}`",
        f"- `{APP_C5.relative_to(ROOT)}`",
    ])
    ADOPTION_NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    qnli_rows = adopt_qnli_iid_overrides()
    c3_pair, c5_pair, _, _ = adopt_ablation_pairs()
    write_notes(qnli_rows, c3_pair, c5_pair)
    print(f"Updated {FINAL_OVERRIDES.relative_to(ROOT)}")
    print(f"Updated {PAIR_TABLE.relative_to(ROOT)}")
    print(f"Wrote {APP_C3.relative_to(ROOT)}")
    print(f"Wrote {APP_C5.relative_to(ROOT)}")
    print(f"Wrote {ADOPTION_NOTES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
