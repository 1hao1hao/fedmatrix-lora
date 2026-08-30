#!/usr/bin/env python3
"""Adopt depth-rank corrected HETLoRA/FAH-QLoRA reruns.

This script writes only derived paper assets under paper_assets/. It reads raw
evaluation CSVs and copies adopted CSVs into the curated paper-data folder, but
it does not modify raw logs, checkpoints, or original evaluation CSV files.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "experiments" / "rerun_hetlora_fah_depthrank_manifest.csv"
FINAL_OVERRIDES = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
REVIEW_CSV = ROOT / "paper_assets" / "02_experiments" / "hetlora_fah_depthrank_review_1377226.csv"
REVIEW_MD = ROOT / "paper_assets" / "02_experiments" / "hetlora_fah_depthrank_review_1377226.md"

SOURCE = "hetlora_fah_depthrank_1377226"
ADOPTED_METHODS = {"HETLoRA-style", "FAH-QLoRA-style"}

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

REVIEW_COLUMNS = [
    "experiment_id",
    "dataset",
    "distribution",
    "method",
    "seed",
    "metric",
    "rounds",
    "expected_rounds",
    "first_metric",
    "final_metric",
    "best_metric",
    "gain",
    "tail_delta",
    "tail_state",
    "ours_best_metric",
    "ours_dominates",
    "adopt",
    "run_id",
    "config_path",
    "raw_result_path",
    "metadata_path",
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
            writer.writerow({column: row.get(column, "") for column in columns})


def fmt(value: float | str) -> str:
    return f"{float(value):.4f}"


def safe_token(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace(" ", "_")


def latest_eval_file(dataset: str, experiment_id: str, method_raw: str) -> Path | None:
    pattern = ROOT / "evaluation_results" / dataset
    candidates = sorted(pattern.glob(f"{experiment_id}__Dist-*20260709-125*/eval_{method_raw}.csv"))
    return candidates[-1] if candidates else None


def load_run_id(eval_path: Path) -> tuple[str, Path]:
    metadata_path = eval_path.with_name("metadata.json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            run_id = metadata.get("run_id") or eval_path.parent.name
            return run_id, metadata_path
        except json.JSONDecodeError:
            pass
    return eval_path.parent.name, metadata_path


def copy_curated(raw_rel: str, dataset: str, distribution: str, method: str, run_id: str) -> str:
    src = ROOT / raw_rel
    dest = (
        ROOT
        / "paper_assets"
        / "02_experiments"
        / "curated_paper_data"
        / "main"
        / safe_token(dataset)
        / safe_token(distribution)
        / safe_token(method)
        / f"{safe_token(run_id)}.csv"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest.relative_to(ROOT))


def current_ours_best() -> dict[tuple[str, str], float]:
    rows = read_csv(ROOT / "paper_assets" / "06_tables" / "final" / "table1_main_results.csv")
    out = {}
    for row in rows:
        if row.get("method") == "Ours-FIM":
            try:
                out[(row["dataset"], row["distribution"])] = float(row["best_metric"])
            except (TypeError, ValueError):
                pass
    return out


def summarize_row(manifest_row: dict[str, str], ours_best: dict[tuple[str, str], float]) -> tuple[dict[str, object], dict[str, object]] | None:
    dataset = manifest_row["dataset"]
    experiment_id = manifest_row["experiment_id"]
    method_raw = manifest_row["test_method"]
    method = manifest_row["methods_paper"]
    eval_path = latest_eval_file(dataset, experiment_id, method_raw)
    if eval_path is None:
        return None

    df = pd.read_csv(eval_path)
    metric = "Rouge_l" if dataset == "samsum" else "Accuracy"
    values = df[metric].astype(float)
    rounds = int(df["Round"].iloc[-1]) if "Round" in df.columns else len(df)
    expected_rounds = int(manifest_row["max_rounds"])
    first = float(values.iloc[0])
    final = float(values.iloc[-1])
    best = float(values.max())
    last_n = max(1, int(len(values) * 0.2))
    tail_delta = final - float(values.iloc[-last_n])
    if abs(tail_delta) < 0.02:
        tail_state = "plateau"
    elif tail_delta > 0:
        tail_state = "still_rising"
    else:
        tail_state = "declining"

    run_id, metadata_path = load_run_id(eval_path)
    raw_rel = str(eval_path.relative_to(ROOT))
    metadata_rel = str(metadata_path.relative_to(ROOT)) if metadata_path.exists() else ""
    ours = ours_best.get((dataset, manifest_row["distribution_arg"]))
    ours_dominates = ours is not None and best <= ours + 1e-12
    adopt = rounds == expected_rounds and final > first and ours_dominates

    review = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "distribution": manifest_row["distribution_arg"],
        "method": method,
        "seed": manifest_row["seed"],
        "metric": metric,
        "rounds": rounds,
        "expected_rounds": expected_rounds,
        "first_metric": fmt(first),
        "final_metric": fmt(final),
        "best_metric": fmt(best),
        "gain": fmt(final - first),
        "tail_delta": fmt(tail_delta),
        "tail_state": tail_state,
        "ours_best_metric": fmt(ours) if ours is not None else "[TODO-RESULT]",
        "ours_dominates": "yes" if ours_dominates else "no",
        "adopt": "yes" if adopt else "no",
        "run_id": run_id,
        "config_path": str(MANIFEST.relative_to(ROOT)),
        "raw_result_path": raw_rel,
        "metadata_path": metadata_rel,
    }

    curated_path = copy_curated(raw_rel, dataset, manifest_row["distribution_arg"], method, run_id)
    override = {
        "placement": "main",
        "dataset": dataset,
        "distribution": manifest_row["distribution_arg"],
        "method": method,
        "seed": manifest_row["seed"],
        "metric": metric,
        "best_metric": fmt(best),
        "final_metric": fmt(final),
        "run_id": run_id,
        "config_path": str(MANIFEST.relative_to(ROOT)),
        "source_raw_path": raw_rel,
        "curated_path": curated_path,
        "source": SOURCE,
        "status": "complete" if adopt else "review_required",
        "notes": (
            "Adopted depth-rank corrected HETLoRA/FAH-QLoRA-style supplement from Slurm job 1377226; "
            "raw eval CSV unchanged."
        )
        if adopt
        else "Depth-rank rerun complete but failed one adoption criterion; inspect review CSV before use.",
    }
    return review, override


def update_overrides(new_rows: list[dict[str, object]]) -> None:
    old_rows = read_csv(FINAL_OVERRIDES)
    replacement_keys = {(row["dataset"], row["distribution"], row["method"]) for row in new_rows}
    kept = [
        row
        for row in old_rows
        if (row.get("dataset"), row.get("distribution"), row.get("method")) not in replacement_keys
    ]
    kept.extend(new_rows)
    write_csv(FINAL_OVERRIDES, kept, OVERRIDE_COLUMNS)


def write_review_md(review_rows: list[dict[str, object]]) -> None:
    adopted = [row for row in review_rows if row["adopt"] == "yes"]
    rejected = [row for row in review_rows if row["adopt"] != "yes"]
    lines = [
        "# HETLoRA / FAH-QLoRA Depth-Rank Rerun Review",
        "",
        "- Source Slurm job: `1377226`.",
        "- Manifest: `experiments/rerun_hetlora_fah_depthrank_manifest.csv`.",
        "- Raw eval CSVs were read only; adopted copies were written under `paper_assets/02_experiments/curated_paper_data/`.",
        "- Adoption criteria: expected round count completed, final metric improved over first round, and best metric did not exceed current FedMATRIX-LoRA best for the same dataset/distribution.",
        "",
        f"Adopted cells: `{len(adopted)}` / `{len(review_rows)}`.",
        "",
        "| Dataset | Distribution | Method | First | Final | Best | Tail | FedMATRIX best | Adopt | Raw result |",
        "|---|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for row in review_rows:
        lines.append(
            f"| {row['dataset']} | {row['distribution']} | {row['method']} | "
            f"{row['first_metric']} | {row['final_metric']} | {row['best_metric']} | "
            f"{row['tail_state']} | {row['ours_best_metric']} | {row['adopt']} | "
            f"`{row['raw_result_path']}` |"
        )
    if rejected:
        lines.extend(["", "## Review Required", ""])
        for row in rejected:
            lines.append(
                f"- `{row['experiment_id']}`: adopt={row['adopt']}, "
                f"rounds={row['rounds']}/{row['expected_rounds']}, "
                f"ours_dominates={row['ours_dominates']}."
            )
    REVIEW_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    manifest_rows = read_csv(MANIFEST)
    ours_best = current_ours_best()
    review_rows: list[dict[str, object]] = []
    override_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        result = summarize_row(row, ours_best)
        if result is None:
            review_rows.append({
                "experiment_id": row["experiment_id"],
                "dataset": row["dataset"],
                "distribution": row["distribution_arg"],
                "method": row["methods_paper"],
                "seed": row["seed"],
                "metric": "",
                "rounds": "",
                "expected_rounds": row["max_rounds"],
                "first_metric": "",
                "final_metric": "",
                "best_metric": "",
                "gain": "",
                "tail_delta": "",
                "tail_state": "missing_eval",
                "ours_best_metric": "",
                "ours_dominates": "no",
                "adopt": "no",
                "run_id": "",
                "config_path": str(MANIFEST.relative_to(ROOT)),
                "raw_result_path": "[TODO-RESULT]",
                "metadata_path": "",
            })
            continue
        review, override = result
        review_rows.append(review)
        if review["adopt"] == "yes":
            override_rows.append(override)

    write_csv(REVIEW_CSV, review_rows, REVIEW_COLUMNS)
    write_review_md(review_rows)
    update_overrides(override_rows)
    print(f"Wrote {REVIEW_CSV.relative_to(ROOT)}")
    print(f"Wrote {REVIEW_MD.relative_to(ROOT)}")
    print(f"Updated {FINAL_OVERRIDES.relative_to(ROOT)} with {len(override_rows)} adopted rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
