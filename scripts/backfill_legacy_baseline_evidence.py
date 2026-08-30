#!/usr/bin/env python3
"""Backfill legacy baseline evidence without modifying raw logs or CSVs.

The current manifest-driven ledger is the clean evidence source. This script
adds legacy QQP/MNLI/CIFAR100 baseline CSVs into separate derived assets so the
paper can distinguish current-protocol evidence from older reusable results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation_results"
CURRENT_TABLE = ROOT / "paper_assets/02_experiments/main_results_table.csv"
EXTENDED_EVIDENCE = ROOT / "paper_assets/02_experiments/extended_baseline_evidence_ledger.csv"
EXTENDED_TABLE = ROOT / "paper_assets/02_experiments/extended_main_results_table.csv"
GAP_REPORT = ROOT / "paper_assets/02_experiments/extended_baseline_gap_report.csv"

METHOD_NAME = {
    "Fed-HeLLo": "Fed-HeLLo",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Random_FIM": "Ours-FIM",
    "Random_Taylor": "Ours-Taylor",
    "FedAvg_LoRA_homogeneous": "FedAvg-LoRA homogeneous",
    "Uniform_allocation": "Uniform allocation",
    "FedRA": "FedRA",
    "MH": "MH",
}

TEXT_DISTRIBUTIONS = {
    "iid": "Dist-iid",
    "dirichlet:1.0": "Dist-dirichlet_1.0",
    "dirichlet:0.1": "Dist-dirichlet_0.1",
}

TEXT_COMPLETE_METHODS = [
    "FedAvg_LoRA_homogeneous",
    "Uniform_allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Random_FIM",
    "Random_Taylor",
]
SAMSUM_LEGACY_DISTRIBUTIONS = {
    "iid": "Dist-iid",
}

CIFAR_RUNS = [
    "20260318-105247",
    "20260318-180715",
    "20260320-171540",
    "20260324-153107",
]
CIFAR_TARGET_METHODS = [
    "FedAvg_LoRA_homogeneous",
    "Uniform_allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Random_FIM",
    "Random_Taylor",
    "FedRA",
    "MH",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill legacy baseline evidence.")
    parser.add_argument("--current-table", default=str(CURRENT_TABLE))
    parser.add_argument("--output", default=str(EXTENDED_EVIDENCE))
    parser.add_argument("--table-output", default=str(EXTENDED_TABLE))
    parser.add_argument("--gap-output", default=str(GAP_REPORT))
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({col: row.get(col, "") for col in columns} for row in rows)


def f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float) -> str:
    return f"{value:.4f}"


def infer_distribution(run_id: str) -> str:
    if run_id.startswith("Dist-iid"):
        return "iid"
    if run_id.startswith("Dist-dirichlet_1.0"):
        return "dirichlet:1.0"
    if run_id.startswith("Dist-dirichlet_0.1"):
        return "dirichlet:0.1"
    return "UNKNOWN"


def infer_budget(run_id: str) -> str:
    marker = "_Bgt-"
    if marker not in run_id:
        return "[TODO-METADATA]"
    tail = run_id.split(marker, 1)[1]
    return tail.split("_", 1)[0]


def summarize_result(path: Path, dataset: str) -> dict:
    if not path.exists():
        return {
            "metric": "[TODO-RESULT]",
            "best_metric": "[TODO-RESULT]",
            "final_metric": "[TODO-RESULT]",
            "best_loss": "[TODO-RESULT]",
            "final_loss": "[TODO-RESULT]",
            "rounds": "0",
            "final_cumulative_time": "[TODO-RESULT]",
            "final_cumulative_budget": "[TODO-RESULT]",
        }
    rows = read_csv(path)
    if not rows:
        return summarize_result(Path("__missing__"), dataset)
    cols = rows[0].keys()
    if dataset == "samsum" and "Rouge_l" in cols:
        metric = "Rouge_l"
        higher_is_better = True
    elif "Accuracy" in cols:
        metric = "Accuracy"
        higher_is_better = True
    else:
        metric = "Loss"
        higher_is_better = False
    values = [f(row.get(metric)) for row in rows]
    losses = [f(row.get("Loss")) for row in rows if row.get("Loss") not in (None, "")]
    best_metric = max(values) if higher_is_better else min(values)
    return {
        "metric": metric,
        "best_metric": fmt(best_metric),
        "final_metric": fmt(values[-1]),
        "best_loss": fmt(min(losses)) if losses else "[TODO-RESULT]",
        "final_loss": fmt(losses[-1]) if losses else "[TODO-RESULT]",
        "rounds": str(len(rows)),
        "final_cumulative_time": fmt(f(rows[-1].get("Cumulative_Time"))),
        "final_cumulative_budget": fmt(f(rows[-1].get("Cumulative_Budget"))),
    }


def current_rows(current_table: Path) -> list[dict]:
    rows = []
    for row in read_csv(current_table):
        rows.append({
            "source": row.get("source") or "current_manifest",
            "config_path": row.get("config_path", "[TODO-METADATA]"),
            "experiment_id": row["experiment_id"],
            "dataset": row["dataset"],
            "distribution": row["distribution"],
            "method_raw": row["method"],
            "method": row["method"],
            "seed": row.get("seed", "[TODO-SEED]"),
            "metric": row["metric"],
            "best_metric": row["best_metric"],
            "final_metric": row["final_metric"],
            "best_loss": row["best_loss"],
            "final_loss": row["final_loss"],
            "rounds": row["rounds"],
            "final_cumulative_time": row["final_cumulative_time"],
            "final_cumulative_budget": row["final_cumulative_budget"],
            "run_id": row["run_id"],
            "raw_result_path": row["raw_result_path"],
            "log_out_path": row["log_out_path"],
            "status": row["status"],
            "notes": "current manifest evidence",
        })
    return rows


def find_text_legacy(dataset: str, distribution: str, method_raw: str) -> Path | None:
    prefix = TEXT_DISTRIBUTIONS[distribution]
    candidates = sorted((RESULTS / dataset).glob(f"{prefix}_Bgt-*/*"))
    matches = [
        path for path in candidates
        if path.name == f"eval_{method_raw}.csv"
    ]
    if not matches:
        return None
    # Prefer current-protocol reruns when available, otherwise newest legacy path.
    return sorted(matches)[-1]


def build_text_rows(existing_keys: set[tuple[str, str, str]]) -> tuple[list[dict], list[dict]]:
    rows, gaps = [], []
    for dataset in ["qqp", "mnli"]:
        for distribution in TEXT_DISTRIBUTIONS:
            for method_raw in TEXT_COMPLETE_METHODS:
                method = METHOD_NAME[method_raw]
                key = (dataset, distribution, method)
                if key in existing_keys:
                    continue
                path = find_text_legacy(dataset, distribution, method_raw)
                if path is None:
                    gaps.append({
                        "dataset": dataset,
                        "distribution": distribution,
                        "method": method,
                        "status": "missing_result_csv",
                        "raw_result_path": "[TODO-RESULT]",
                        "notes": "No current or legacy CSV found.",
                    })
                    continue
                run_id = path.parent.name
                summary = summarize_result(path, dataset)
                rows.append({
                    "source": "legacy_backfill",
                    "config_path": "[TODO-METADATA]",
                    "experiment_id": f"LEGACY_{dataset.upper()}_{distribution.replace(':', '_').replace('.', 'p')}_{method.replace('-', '_').replace(' ', '_')}",
                    "dataset": dataset,
                    "distribution": distribution,
                    "method_raw": method_raw,
                    "method": method,
                    "seed": "[TODO-SEED]",
                    **summary,
                    "run_id": run_id,
                    "raw_result_path": rel(path),
                    "log_out_path": "[TODO-METADATA]",
                    "status": "legacy_complete",
                    "notes": "Legacy CSV backfilled by dataset/distribution encoded in run_id; raw log/metadata may be unavailable.",
                })
    return rows, gaps


def find_samsum_legacy(distribution: str, method_raw: str) -> Path | None:
    prefix = SAMSUM_LEGACY_DISTRIBUTIONS[distribution]
    candidates = sorted((RESULTS / "samsum").glob(f"{prefix}_Bgt-*/*"))
    matches = [
        path for path in candidates
        if path.name == f"eval_{method_raw}.csv"
    ]
    if not matches:
        return None
    return sorted(matches)[-1]


def build_samsum_rows(existing_keys: set[tuple[str, str, str]]) -> tuple[list[dict], list[dict]]:
    rows, gaps = [], []
    for distribution in SAMSUM_LEGACY_DISTRIBUTIONS:
        for method_raw in TEXT_COMPLETE_METHODS:
            method = METHOD_NAME[method_raw]
            key = ("samsum", distribution, method)
            if key in existing_keys:
                continue
            path = find_samsum_legacy(distribution, method_raw)
            if path is None:
                gaps.append({
                    "dataset": "samsum",
                    "distribution": distribution,
                    "method": method,
                    "status": "missing_result_csv",
                    "raw_result_path": "[TODO-RESULT]",
                    "notes": "No current or legacy SAMSum CSV found for this baseline.",
                })
                continue
            run_id = path.parent.name
            summary = summarize_result(path, "samsum")
            rows.append({
                "source": "legacy_backfill",
                "config_path": "[TODO-METADATA]",
                "experiment_id": f"LEGACY_SAMSUM_{distribution}_{method.replace('-', '_').replace(' ', '_')}",
                "dataset": "samsum",
                "distribution": distribution,
                "method_raw": method_raw,
                "method": method,
                "seed": "[TODO-SEED]",
                **summary,
                "run_id": run_id,
                "raw_result_path": rel(path),
                "log_out_path": "[TODO-METADATA]",
                "status": "legacy_complete",
                "notes": "Legacy SAMSum CSV backfilled by dataset/distribution encoded in run_id; raw log/metadata may be unavailable.",
            })
    return rows, gaps


def best_cifar_csv(method_raw: str) -> Path | None:
    matches = []
    for run_id in CIFAR_RUNS:
        path = RESULTS / "cifar100" / run_id / f"eval_{method_raw}.csv"
        if path.exists():
            matches.append(path)
    if not matches:
        return None
    # Prefer Accuracy-bearing runs; otherwise choose the latest available path.
    accuracy_matches = []
    for path in matches:
        rows = read_csv(path)
        if rows and "Accuracy" in rows[0]:
            accuracy_matches.append(path)
    return sorted(accuracy_matches or matches)[-1]


def build_cifar_rows(existing_methods: set[str]) -> tuple[list[dict], list[dict]]:
    rows, gaps = [], []
    for method_raw in CIFAR_TARGET_METHODS:
        method = METHOD_NAME[method_raw]
        if method in existing_methods:
            continue
        path = best_cifar_csv(method_raw)
        if path is None:
            gaps.append({
                "dataset": "cifar100",
                "distribution": "UNKNOWN",
                "method": method,
                "status": "missing_result_csv",
                "raw_result_path": "[TODO-RESULT]",
                "notes": "No CIFAR100 CSV found for this baseline in historical result directories.",
            })
            continue
        summary = summarize_result(path, "cifar100")
        status = "legacy_partial_metadata"
        notes = "CIFAR100 legacy CSV exists, but distribution/seed/config metadata are not recoverable from the run directory alone."
        if summary["metric"] == "Loss":
            status = "legacy_loss_only"
            notes += " CSV has no Accuracy column, so metric is Loss and lower is better."
        rows.append({
            "source": "legacy_backfill",
            "config_path": "[TODO-METADATA]",
            "experiment_id": f"LEGACY_CIFAR100_UNKNOWN_{method.replace('-', '_').replace(' ', '_')}",
            "dataset": "cifar100",
            "distribution": "UNKNOWN",
            "method_raw": method_raw,
            "method": method,
            "seed": "[TODO-SEED]",
            **summary,
            "run_id": path.parent.name,
            "raw_result_path": rel(path),
            "log_out_path": "logs/job_1138668.out" if path.parent.name == "20260324-153107" else "[TODO-METADATA]",
            "status": status,
            "notes": notes,
        })
    return rows, gaps


def main() -> int:
    args = parse_args()
    current_table = Path(args.current_table)
    if not current_table.is_absolute():
        current_table = ROOT / current_table
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    table_output = Path(args.table_output)
    if not table_output.is_absolute():
        table_output = ROOT / table_output
    gap_output = Path(args.gap_output)
    if not gap_output.is_absolute():
        gap_output = ROOT / gap_output

    rows = current_rows(current_table)
    existing_keys = {(row["dataset"], row["distribution"], row["method"]) for row in rows}
    text_rows, text_gaps = build_text_rows(existing_keys)
    samsum_rows, samsum_gaps = build_samsum_rows(existing_keys)
    existing_cifar_methods = {
        row["method"] for row in rows
        if row["dataset"] == "cifar100" and row["distribution"] != "UNKNOWN"
    }
    cifar_rows, cifar_gaps = build_cifar_rows(existing_cifar_methods)
    rows.extend(text_rows)
    rows.extend(samsum_rows)
    rows.extend(cifar_rows)
    gaps = text_gaps + samsum_gaps + cifar_gaps

    columns = [
        "source", "config_path", "experiment_id", "dataset", "distribution", "method_raw", "method",
        "seed", "metric", "best_metric", "final_metric", "best_loss", "final_loss", "rounds",
        "final_cumulative_time", "final_cumulative_budget", "run_id", "raw_result_path",
        "log_out_path", "status", "notes",
    ]
    write_csv(output, rows, columns)
    write_csv(table_output, rows, [
        "dataset", "distribution", "method", "seed", "metric", "best_metric", "final_metric",
        "final_cumulative_time", "final_cumulative_budget", "run_id", "raw_result_path",
        "source", "config_path", "status", "notes",
    ])
    write_csv(gap_output, gaps, ["dataset", "distribution", "method", "status", "raw_result_path", "notes"])
    print(f"Wrote {len(rows)} rows to {rel(output)}")
    print(f"Wrote {len(rows)} rows to {rel(table_output)}")
    print(f"Wrote {len(gaps)} gaps to {rel(gap_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
