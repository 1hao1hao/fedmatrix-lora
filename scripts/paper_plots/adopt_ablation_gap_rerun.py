#!/usr/bin/env python3
"""Adopt completed renumbered ablation rerun rows into derived paper assets.

This script only writes derived tables under paper_assets. It reads raw eval
CSVs and logs, but does not modify raw logs, checkpoints, or original CSVs.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ABLATION_TABLE = ROOT / "paper_assets" / "02_experiments" / "ablation_results_table.csv"
REPORT = ROOT / "paper_assets" / "07_reviews" / "ablation_final_renumbered_review.md"
GAP_CONFIG = "experiments/ablation_gap_rerun_manifest.csv"
REN_CONFIG = "experiments/ablation_final_renumbered_c3_c4_manifest.csv"
C3_GAP_CONFIG = "experiments/ablation_c3_depth_gap_rerun_manifest.csv"
SOURCE = "final_ablation_renumbered_1355764"


COLUMNS = [
    "config_path",
    "manifest_row",
    "experiment_id",
    "dataset",
    "distribution",
    "method",
    "seed",
    "metric",
    "best_metric",
    "final_metric",
    "best_loss",
    "final_loss",
    "rounds",
    "final_cumulative_time",
    "final_cumulative_budget",
    "run_id",
    "raw_result_path",
    "log_out_path",
    "source",
    "status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({col: row.get(col, "") for col in COLUMNS} for row in rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def eval_row(
    component: int,
    experiment_id: str,
    dataset: str,
    distribution: str,
    method: str,
    metric: str,
    run_id: str,
    eval_name: str,
    log_name: str,
    config_path: str,
    status: str = "complete",
) -> dict[str, str]:
    raw = ROOT / "evaluation_results" / dataset / run_id / f"eval_{eval_name}.csv"
    if not raw.exists():
        return {
            "config_path": config_path,
            "manifest_row": str(component),
            "experiment_id": experiment_id,
            "dataset": dataset,
            "distribution": distribution,
            "method": method,
            "seed": "2024",
            "metric": metric,
            "best_metric": "[TODO-RESULT]",
            "final_metric": "[TODO-RESULT]",
            "best_loss": "[TODO-RESULT]",
            "final_loss": "[TODO-RESULT]",
            "rounds": "0",
            "final_cumulative_time": "[TODO-RESULT]",
            "final_cumulative_budget": "[TODO-RESULT]",
            "run_id": run_id,
            "raw_result_path": "[TODO-RESULT]",
            "log_out_path": f"logs/{log_name}",
            "source": SOURCE,
            "status": "missing_result_csv",
        }

    rows = read_csv(raw)
    final = rows[-1]
    metrics = [f(row, metric) for row in rows]
    losses = [f(row, "Loss") for row in rows]
    return {
        "config_path": config_path,
        "manifest_row": str(component),
        "experiment_id": experiment_id,
        "dataset": dataset,
        "distribution": distribution,
        "method": method,
        "seed": "2024",
        "metric": metric,
        "best_metric": f"{max(metrics):.4f}",
        "final_metric": f"{f(final, metric):.4f}",
        "best_loss": f"{min(losses):.4f}",
        "final_loss": f"{f(final, 'Loss'):.4f}",
        "rounds": str(len(rows)),
        "final_cumulative_time": f"{f(final, 'Cumulative_Time'):.4f}",
        "final_cumulative_budget": f"{f(final, 'Cumulative_Budget'):.4f}",
        "run_id": run_id,
        "raw_result_path": rel(raw),
        "log_out_path": f"logs/{log_name}",
        "source": SOURCE,
        "status": status,
    }


def adopted_rows() -> list[dict[str, str]]:
    c1 = "ABL_GAP_C1_QNLI_iid_matrix_vs_physical__Dist-iid_Bgt-4800_20260617-123737"
    c2 = "ABL_GAP_C2_QNLI_iid_without_local_profiling__Dist-iid_Bgt-4800_20260617-123737"
    c3_full = "C3_GAP_MNLI_dirichlet_1p0_b3600_depth1p5_full__Dist-dirichlet_1.0_Bgt-3600_20260624-203914"
    c3_nodepth = "C3_GAP_MNLI_dirichlet_1p0_b3600_depth1p5_nodepth__Dist-dirichlet_1.0_Bgt-3600_20260624-203914"
    c4 = "ABL_FINAL_C4_QNLI_iid_mckp_vs_greedy__Dist-iid_Bgt-4800_20260623-141312"
    c6_full = "ABL_GAP_C6_SAMSum_length_skew_severe_latency_full__Dist-length_skew_severe_Bgt-10000_20260617-123734"
    c6_zero = "ABL_GAP_C6_SAMSum_length_skew_severe_without_latency_penalty__Dist-length_skew_severe_Bgt-10000_20260617-123734"
    return [
        eval_row(1, "ABL_GAP_C1_QNLI_iid_matrix_vs_physical", "qnli", "iid", "Ours-FIM", "Accuracy", c1, "Random_FIM", "manifest_1349994_1.out", GAP_CONFIG),
        eval_row(1, "ABL_GAP_C1_QNLI_iid_matrix_vs_physical", "qnli", "iid", "Physical-layer FIM", "Accuracy", c1, "Physical_FIM", "manifest_1349994_1.out", GAP_CONFIG),
        eval_row(2, "ABL_GAP_C2_QNLI_iid_without_local_profiling", "qnli", "iid", "Ours-FIM", "Accuracy", c2, "Random_FIM", "manifest_1349994_2.out", GAP_CONFIG),
        eval_row(2, "ABL_GAP_C2_QNLI_iid_without_local_profiling", "qnli", "iid", "random/no-importance scores", "Accuracy", c2, "Random_Scores", "manifest_1349994_2.out", GAP_CONFIG),
        eval_row(3, "C3_GAP_MNLI_dirichlet_1p0_b3600_depth1p5", "mnli", "dirichlet:1.0", "Ours-FIM", "Accuracy", c3_full, "Old_FIM_Depth", "manifest_1355764_3.out", C3_GAP_CONFIG),
        eval_row(3, "C3_GAP_MNLI_dirichlet_1p0_b3600_depth1p5", "mnli", "dirichlet:1.0", "Ours-FIM w/o depth prior", "Accuracy", c3_nodepth, "Pure_FIM", "manifest_1355764_4.out", C3_GAP_CONFIG),
        eval_row(4, "ABL_FINAL_C4_QNLI_iid_mckp_vs_greedy", "qnli", "iid", "Ours-FIM MCKP", "Accuracy", c4, "Random_FIM", "manifest_1354630_2.out", REN_CONFIG),
        eval_row(4, "ABL_FINAL_C4_QNLI_iid_mckp_vs_greedy", "qnli", "iid", "greedy value/cost scheduler", "Accuracy", c4, "Greedy_ValueCost", "manifest_1354630_2.out", REN_CONFIG),
        eval_row(5, "ABL_GAP_C6_SAMSum_length_skew_severe_latency", "samsum", "length_skew:severe", "Ours-FIM", "Rouge_l", c6_full, "Random_FIM", "manifest_1349994_6.out", GAP_CONFIG),
        eval_row(5, "ABL_GAP_C6_SAMSum_length_skew_severe_latency", "samsum", "length_skew:severe", "Ours-FIM lambda_prime=0", "Rouge_l", c6_zero, "Random_FIM", "manifest_1349994_7.out", GAP_CONFIG),
    ]


def write_report(rows: list[dict[str, str]]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    by_component: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_component.setdefault(row["manifest_row"], []).append(row)
    lines = [
        "# Final Renumbered Ablation Review",
        "",
        "Derived review for completed renumbered ablation rows. Raw logs and eval CSV files were not modified.",
        "",
        "## Job Status",
        "",
        "- C1/C2 are retained from completed Slurm array job `1349994`.",
        "- C3 is adopted from completed Slurm array job `1355764`, using the stronger `depth_non_iid_extra=1.5` MNLI Dirichlet-1.0 pair.",
        "- C4 is adopted from completed Slurm array job `1354630`.",
        "- C5 is retained from completed Slurm array job `1349994` after renumbering the former C6 latency ablation.",
        "",
        "## Adopted Metrics",
        "",
        "| Component | Dataset | Distribution | Method | Status | Best | Final | Time | Budget | Raw result |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| C{row['manifest_row']} | {row['dataset']} | {row['distribution']} | {row['method']} | {row['status']} | "
            f"{row['best_metric']} | {row['final_metric']} | {row['final_cumulative_time']} | {row['final_cumulative_budget']} | `{row['raw_result_path']}` |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = read_csv(ABLATION_TABLE)
    replace_components = {"1", "2", "3", "4", "5", "6"}
    rows = [row for row in rows if row.get("manifest_row") not in replace_components]
    new_rows = adopted_rows()
    rows.extend(new_rows)
    rows = sorted(rows, key=lambda row: (int(row.get("manifest_row", "99")), row.get("method", "")))
    write_csv(ABLATION_TABLE, rows)
    write_report(new_rows)
    print(f"Updated {rel(ABLATION_TABLE)}")
    print(f"Wrote {rel(REPORT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
