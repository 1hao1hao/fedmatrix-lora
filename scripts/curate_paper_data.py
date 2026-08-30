#!/usr/bin/env python3
"""Copy paper-used result CSVs into a curated, traceable folder.

The script intentionally copies raw result CSVs instead of moving or editing
them, preserving the original experiment evidence.
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "02_experiments" / "curated_paper_data"

MAIN_TABLE = ROOT / "paper_assets" / "06_tables" / "final" / "table1_main_results.csv"
FULL_TABLE = ROOT / "paper_assets" / "06_tables" / "final" / "app_table_full_results.csv"
ABLATION_TABLE = ROOT / "paper_assets" / "06_tables" / "final" / "table2_ablation_pairs.csv"

MAIN_KEYS = {
    ("cifar100", "dirichlet:0.1"),
    ("samsum", "length_skew:mild"),
    ("samsum", "length_skew:severe"),
}

STRESS_KEYS = {
    ("qqp", "dirichlet:0.1"),
    ("mnli", "dirichlet:0.1"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_token(value: str) -> str:
    value = value.strip().replace(":", "_").replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def copy_result(src_text: str, dst: Path) -> tuple[str, str]:
    if not src_text:
        return "", "[TODO-RESULT] missing raw_result_path"
    src = ROOT / src_text
    if not src.exists():
        return "", f"[TODO-RESULT] missing source file: {src_text}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(ROOT)), "copied"


def emit_row(
    rows: list[dict[str, str]],
    *,
    placement: str,
    dataset: str,
    distribution: str,
    method: str,
    seed: str,
    metric: str,
    best_metric: str,
    final_metric: str,
    run_id: str,
    config_path: str,
    raw_result_path: str,
    source: str,
    curated_path: str,
    status: str,
    notes: str,
) -> None:
    rows.append(
        {
            "placement": placement,
            "dataset": dataset,
            "distribution": distribution,
            "method": method,
            "seed": seed,
            "metric": metric,
            "best_metric": best_metric,
            "final_metric": final_metric,
            "run_id": run_id,
            "config_path": config_path,
            "source_raw_path": raw_result_path,
            "curated_path": curated_path,
            "source": source,
            "status": status,
            "notes": notes,
        }
    )


def curate_main_and_stress(index_rows: list[dict[str, str]]) -> None:
    for row in read_csv(FULL_TABLE):
        key = (row.get("dataset", ""), row.get("distribution", ""))
        if key in MAIN_KEYS:
            placement = "main"
        elif key in STRESS_KEYS:
            placement = "appendix_stress"
        else:
            continue

        dataset = row.get("dataset", "")
        distribution = row.get("distribution", "")
        method = row.get("method", "")
        dst = (
            OUT_DIR
            / placement
            / safe_token(dataset)
            / safe_token(distribution)
            / safe_token(method)
            / f"{safe_token(row.get('run_id', 'run'))}.csv"
        )
        curated_path, status = copy_result(row.get("raw_result_path", ""), dst)
        notes = "paper-ready curated copy; original file unchanged"
        if status != "copied":
            notes = status

        emit_row(
            index_rows,
            placement=placement,
            dataset=dataset,
            distribution=distribution,
            method=method,
            seed=row.get("seed", ""),
            metric=row.get("metric", ""),
            best_metric=row.get("best_metric", ""),
            final_metric=row.get("final_metric", ""),
            run_id=row.get("run_id", ""),
            config_path=row.get("config_path", ""),
            raw_result_path=row.get("raw_result_path", ""),
            source=row.get("source", ""),
            curated_path=curated_path,
            status="complete" if status == "copied" else "missing",
            notes=notes,
        )


def curate_ablation(index_rows: list[dict[str, str]]) -> None:
    for row in read_csv(ABLATION_TABLE):
        component = row.get("component", "")
        dataset = row.get("dataset", "")
        distribution = row.get("distribution", "")
        group = f"{safe_token(component)}_{safe_token(dataset)}_{safe_token(distribution)}"

        for role, method_col, seed_col, run_col, config_col, raw_col in [
            ("full", "full_method", "full_seed", "full_run_id", "full_config_path", "full_raw_result_path"),
            (
                "ablated",
                "ablated_method",
                "ablated_seed",
                "ablated_run_id",
                "ablated_config_path",
                "ablated_raw_result_path",
            ),
        ]:
            method = row.get(method_col, "")
            dst = (
                OUT_DIR
                / "ablation"
                / group
                / safe_token(role)
                / safe_token(method)
                / f"{safe_token(row.get(run_col, 'run'))}.csv"
            )
            curated_path, status = copy_result(row.get(raw_col, ""), dst)
            notes = "paper-ready curated copy; original file unchanged"
            if status != "copied":
                notes = status

            emit_row(
                index_rows,
                placement="ablation",
                dataset=dataset,
                distribution=distribution,
                method=method,
                seed=row.get(seed_col, ""),
                metric=row.get("metric", ""),
                best_metric="",
                final_metric="",
                run_id=row.get(run_col, ""),
                config_path=row.get(config_col, ""),
                raw_result_path=row.get(raw_col, ""),
                source="ablation_table",
                curated_path=curated_path,
                status="complete" if status == "copied" else "missing",
                notes=notes,
            )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, str]] = []
    curate_main_and_stress(index_rows)
    curate_ablation(index_rows)

    index_path = OUT_DIR / "curated_data_index.csv"
    fieldnames = [
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
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    complete = sum(1 for row in index_rows if row["status"] == "complete")
    missing = len(index_rows) - complete
    print(f"Wrote {len(index_rows)} curated index rows to {index_path.relative_to(ROOT)}")
    print(f"Complete copies: {complete}; missing: {missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
