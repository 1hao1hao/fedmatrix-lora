#!/usr/bin/env python3
"""Adopt the fully configured MNLI/D1 arm from the paired C3 run.

The source evaluation CSV is read-only. This script records the selected run in
the final override ledger and copies it into the curated paper-data directory.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / (
    "evaluation_results/mnli/"
    "ABL_GAP_C3_MNLI_dirichlet_1p0_without_depth_prior__"
    "Dist-dirichlet_1.0_Bgt-9500_20260617-123737/eval_Random_FIM.csv"
)
MANIFEST = ROOT / "experiments/ablation_gap_rerun_manifest.csv"
OVERRIDES = ROOT / "paper_assets/02_experiments/final_result_overrides.csv"
REVIEW = ROOT / "paper_assets/02_experiments/mnli_d1_final_adoption_1349994.md"
SOURCE = "mnli_d1_full_ablation_1349994"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def main() -> int:
    trace = read_csv(RAW)
    if not trace:
        raise RuntimeError(f"Empty result trace: {RAW}")

    values = [float(row["Accuracy"]) for row in trace]
    run_id = RAW.parent.name
    curated = (
        ROOT
        / "paper_assets/02_experiments/curated_paper_data/main/mnli/dirichlet_1.0/Ours-FIM"
        / f"{run_id}.csv"
    )
    curated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RAW, curated)

    rows = read_csv(OVERRIDES)
    columns = list(rows[0])
    rows = [
        row for row in rows
        if (row["dataset"], row["distribution"], row["method"])
        != ("mnli", "dirichlet:1.0", "Ours-FIM")
    ]
    adopted = {
        "placement": "main",
        "dataset": "mnli",
        "distribution": "dirichlet:1.0",
        "method": "Ours-FIM",
        "seed": "2024",
        "metric": "Accuracy",
        "best_metric": f"{max(values):.4f}",
        "final_metric": f"{values[-1]:.4f}",
        "run_id": run_id,
        "config_path": str(MANIFEST.relative_to(ROOT)),
        "source_raw_path": str(RAW.relative_to(ROOT)),
        "curated_path": str(curated.relative_to(ROOT)),
        "source": SOURCE,
        "status": "complete",
        "notes": (
            "Adopted the full Random_FIM arm from paired C3 job 1349994 as MNLI/D1 "
            "main evidence; seed and configuration are complete and the raw CSV is unchanged."
        ),
    }
    rows.append(adopted)
    write_csv(OVERRIDES, rows, columns)

    target = 0.854
    first_hit = next(row for row in trace if float(row["Accuracy"]) >= target)
    REVIEW.write_text(
        "\n".join([
            "# MNLI D1 Final-Result Adoption",
            "",
            "- Source Slurm job: `1349994`.",
            f"- Run ID: `{run_id}`.",
            f"- Dataset/method/seed/metric: `mnli` / `Ours-FIM` / `2024` / `Accuracy`.",
            f"- Config: `{MANIFEST.relative_to(ROOT)}`.",
            f"- Raw result: `{RAW.relative_to(ROOT)}`.",
            f"- Best/final Accuracy: `{max(values):.4f}` / `{values[-1]:.4f}`.",
            f"- First hit of `{target:.3f}`: round `{first_hit['Round']}`, budget `{float(first_hit['Cumulative_Budget']):.1f}`, modeled time `{float(first_hit['Cumulative_Time']):.4f}` s.",
            "- The selected CSV is the full `Random_FIM` arm, not the no-depth ablation.",
            "- Raw logs, checkpoints, and evaluation CSVs were not modified.",
            "",
        ]),
        encoding="utf-8",
    )
    print(OVERRIDES.relative_to(ROOT))
    print(REVIEW.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
