"""Shared helpers for paper figure/table scripts.

The helpers read only generated ledgers or result CSVs and write derived paper
assets under paper_assets/05_figures. They never modify raw experiment results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "paper_assets/02_experiments/experiment_todo_registry.csv"
FIGURE_DIR = ROOT / "paper_assets/05_figures"
MANIFEST_PATH = FIGURE_DIR / "figure_data_manifest.md"

TODO_MARKERS = {"[TODO-RESULT]", "[TODO-METADATA]", "[TODO-SEED]", "[TODO-BASELINE]", ""}
METRIC_CANDIDATES = [
    "Accuracy",
    "Rouge_l",
    "Rouge-L",
    "Loss",
    "Raw_LM_Loss",
    "T_Actual",
    "Cumulative_Budget",
    "Avg_Memory_GB",
]


def default_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="Experiment todo registry CSV.")
    parser.add_argument("--out-dir", default=str(FIGURE_DIR), help="Output directory for figures/tables.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Figure data manifest path.")
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Figure format for plotting scripts.",
    )
    return parser


def ensure_out_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_registry(path: str | Path = REGISTRY_PATH) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean_path(raw_path: str) -> str:
    raw_path = (raw_path or "").strip()
    if raw_path in TODO_MARKERS or raw_path.startswith("[TODO"):
        return ""
    return raw_path


def resolve_data_path(row: dict) -> Path | None:
    raw_path = clean_path(row.get("raw_result_path", ""))
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def is_missing(row: dict) -> bool:
    status = (row.get("current_status") or "").strip().lower()
    path = resolve_data_path(row)
    return status == "missing" or path is None or not path.exists()


def read_result_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def choose_metric_column(result_rows: list[dict], requested_metric: str) -> str | None:
    if not result_rows:
        return None
    columns = result_rows[0].keys()
    requested = (requested_metric or "").strip()
    if requested in columns:
        return requested
    for candidate in METRIC_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def summarize_metric(row: dict) -> tuple[str, float | None, float | None]:
    path = resolve_data_path(row)
    if path is None or not path.exists():
        return row.get("metric", ""), None, None
    result_rows = read_result_rows(path)
    metric_col = choose_metric_column(result_rows, row.get("metric", ""))
    if metric_col is None:
        return row.get("metric", ""), None, None
    values = [numeric(r.get(metric_col)) for r in result_rows]
    values = [v for v in values if v is not None]
    if not values:
        return metric_col, None, None
    return metric_col, values[-1], max(values)


def manifest_row(figure_name: str, row: dict, notes: str = "") -> dict:
    path = resolve_data_path(row)
    missing = "missing" if is_missing(row) else (row.get("current_status") or "partial")
    raw_path = clean_path(row.get("raw_result_path", ""))
    if path is not None and not path.exists():
        missing = "missing"
        notes = (notes + " raw data path does not exist.").strip()
    return {
        "figure name": figure_name,
        "experiment_id": row.get("experiment_id", ""),
        "raw data path": raw_path or "[TODO-RESULT]",
        "metric": row.get("metric", ""),
        "missing status": missing,
        "notes": notes or row.get("missing_reason", "") or row.get("notes", ""),
    }


def write_manifest(rows: list[dict], manifest_path: str | Path = MANIFEST_PATH) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["figure name", "experiment_id", "raw data path", "metric", "missing status", "notes"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Figure Data Manifest\n\n")
        handle.write("Generated from `paper_assets/02_experiments/experiment_todo_registry.csv`.\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for row in rows:
            values = [str(row.get(col, "")).replace("\n", " ").replace("|", "/") for col in columns]
            handle.write("| " + " | ".join(values) + " |\n")


def write_csv(rows: list[dict], path: str | Path, columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def group_rows(rows: list[dict], predicate) -> list[dict]:
    return [row for row in rows if predicate(row)]


def mean_or_none(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None
