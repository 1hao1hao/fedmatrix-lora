#!/usr/bin/env python3
"""Normalize legacy SAMSum experiment names to length-skew settings.

This script updates paper assets only. It does not touch raw experiment logs,
checkpoints, or original result CSVs.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "paper_assets/02_experiments/experiment_todo_registry.csv"
PAPER_ASSETS = ROOT / "paper_assets"
LEGACY_MILD_TAG = "dirichlet_1p0"
LEGACY_SEVERE_TAG = "dirichlet_0p1"


def legacy_main_prefix(tag: str) -> str:
    return f"MAIN_SAMSum_{tag}_"


def normalized_main_prefix(severity: str) -> str:
    return f"MAIN_SAMSum_length_skew_{severity}_"

MILD_RAW_PATHS = {
    "Fed_HeLLo": "evaluation_results/samsum/Dist-length_skew_mild_Bgt-10000_20260511-105247/eval_Fed-HeLLo.csv",
    "LEGEND": "evaluation_results/samsum/Dist-length_skew_mild_Bgt-10000_20260511-105247/eval_LEGEND.csv",
    "MS": "evaluation_results/samsum/Dist-length_skew_mild_Bgt-10000_20260511-105247/eval_MS.csv",
    "Ours_FIM": "evaluation_results/samsum/Dist-length_skew_mild_Bgt-10000_20260511-105247/eval_Random_FIM.csv",
}

METHOD_SUFFIX_TO_RAW = {
    "FedAvg_LoRA_homogeneous": "FedAvg_LoRA_homogeneous",
    "Uniform_allocation": "Uniform_allocation",
    "Fed_HeLLo": "Fed-HeLLo",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Ours_FIM": "Random_FIM",
    "Ours_Taylor": "Random_Taylor",
}


def normalize_registry() -> None:
    with REGISTRY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = handle.readline()

    if rows:
        fieldnames = list(rows[0].keys())

    for row in rows:
        exp_id = row["experiment_id"]
        if exp_id.startswith(legacy_main_prefix(LEGACY_MILD_TAG)):
            suffix = exp_id.replace(legacy_main_prefix(LEGACY_MILD_TAG), "")
            row["experiment_id"] = f"{normalized_main_prefix('mild')}{suffix}"
            row["distribution"] = "length_skew_mild"
            row["paper_section"] = "Experiment/Main Results"
            row["table_or_figure"] = "Main table + performance-vs-round/budget/time figures"
            row["priority"] = "must"
            row["notes"] = "SAMSum uses length-skew heterogeneity; this row replaces the former mild placeholder entry."

            if suffix in MILD_RAW_PATHS:
                row["current_status"] = "partial"
                row["raw_result_path"] = MILD_RAW_PATHS[suffix]
                row["missing_reason"] = "raw CSV exists for SAMSum length_skew_mild, but run-level metadata/seed/config path are not archived; result needs schema/metric validation before camera-ready use"
            elif suffix in {"FedAvg_LoRA_homogeneous", "Uniform_allocation"}:
                row["current_status"] = "missing"
                row["raw_result_path"] = "[TODO-RESULT]"
                row["missing_reason"] = "baseline not implemented/indexed as evaluation CSV for SAMSum length_skew_mild; no raw result path found"
            else:
                raw_name = METHOD_SUFFIX_TO_RAW.get(suffix, suffix)
                row["current_status"] = "missing"
                row["raw_result_path"] = "[TODO-RESULT]"
                row["missing_reason"] = f"expected CSV evaluation_results/samsum/Dist-length_skew_mild_Bgt-10000_*/eval_{raw_name}.csv not found"

        elif exp_id.startswith(legacy_main_prefix(LEGACY_SEVERE_TAG)):
            suffix = exp_id.replace(legacy_main_prefix(LEGACY_SEVERE_TAG), "")
            row["experiment_id"] = f"{normalized_main_prefix('severe')}{suffix}"
            row["distribution"] = "length_skew_severe"
            row["current_status"] = "missing"
            row["raw_result_path"] = "[TODO-RESULT]"
            row["missing_reason"] = "required SAMSum length_skew_severe run not indexed; no raw result path found"
            row["priority"] = "must"
            row["notes"] = "SAMSum uses length-skew heterogeneity; this row replaces the former severe placeholder entry."

        elif exp_id.startswith("EXTRA_SAMSum_length_skew_mild_"):
            row["notes"] = "Legacy appendix row retained for traceability; the corresponding length_skew_mild setting is now represented in the MAIN_SAMSum_length_skew_mild_* grid."

    with REGISTRY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text_assets() -> None:
    replacements = {
        legacy_main_prefix(LEGACY_MILD_TAG).rstrip("_"): normalized_main_prefix("mild").rstrip("_"),
        legacy_main_prefix(LEGACY_SEVERE_TAG).rstrip("_"): normalized_main_prefix("severe").rstrip("_"),
        "SAMSum / " + "dirichlet_1.0": "SAMSum / length_skew_mild",
        "SAMSum / " + "dirichlet_0.1": "SAMSum / length_skew_severe",
        "SAMSum/" + "dirichlet_1.0": "SAMSum/length_skew_mild",
        "SAMSum/" + "dirichlet_0.1": "SAMSum/length_skew_severe",
        "SAMSum " + "Dirichlet `1.0` and `0.1`": "SAMSum length-skew mild and severe",
        "SAMSum lacks Dirichlet 1.0/0.1": "SAMSum lacks length-skew severe and complete length-skew mild",
        "SAMSum lacks required Dirichlet 1.0 and Dirichlet 0.1": "SAMSum lacks required length-skew mild/severe coverage",
        "SAMSum are missing Ours-Taylor in several settings. SAMSum lacks Dirichlet 1.0/0.1 runs": "SAMSum are missing Ours-Taylor in several settings. SAMSum length-skew coverage is incomplete",
        "SAMSum IID/length-skew/Dirichlet-style runs": "SAMSum IID/length-skew runs",
        "SAMSum currently has IID partial results but missing Dirichlet main-grid results": "SAMSum currently has IID partial results and incomplete length-skew main-grid results",
    }

    for path in PAPER_ASSETS.rglob("*"):
        if path.suffix.lower() not in {".md", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        new_text = text
        for old, new in replacements.items():
            new_text = new_text.replace(old, new)

        lines = []
        for line in new_text.splitlines(keepends=True):
            if "SAMSum" in line:
                line = line.replace("dirichlet_1.0", "length_skew_mild")
                line = line.replace("dirichlet_0.1", "length_skew_severe")
                line = line.replace("dirichlet 1.0", "length_skew_mild")
                line = line.replace("dirichlet 0.1", "length_skew_severe")
                line = line.replace("Dirichlet 1.0", "length-skew mild")
                line = line.replace("Dirichlet 0.1", "length-skew severe")
                line = line.replace("IID/Dirichlet grid", "IID/length-skew grid")
            lines.append(line)
        new_text = "".join(lines)

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")


def main() -> None:
    normalize_registry()
    normalize_text_assets()


if __name__ == "__main__":
    main()
