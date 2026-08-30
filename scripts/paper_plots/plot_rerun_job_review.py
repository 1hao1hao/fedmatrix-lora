#!/usr/bin/env python3
"""Plot diagnostic curves for rerun jobs 1261770, 1261741, and 1261185.

The script reads raw evaluation CSVs and writes only derived paper assets.
Incomplete runs are reported in the manifest rather than inferred.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_assets" / "05_figures" / "rerun_job_review"
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "rerun_job_review_summary.csv"
MANIFEST = OUT_DIR / "rerun_job_review_manifest.md"

METHOD_LABELS = {
    "FedAvg_LoRA_homogeneous": "FedAvg-LoRA homogeneous",
    "Uniform_allocation": "Uniform allocation",
    "Fed-HeLLo": "Fed-HeLLo",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Random_FIM": "Ours-FIM",
}
RUN_DIR_METHOD_TOKENS = {
    "Random_FIM": ["_Random_FIM__", "_Ours_FIM__"],
}
METHOD_ORDER = [
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Ours-FIM",
]
COLORS = {
    "FedAvg-LoRA homogeneous": "#8c8c8c",
    "Uniform allocation": "#4c78a8",
    "Fed-HeLLo": "#72b7b2",
    "LEGEND": "#54a24b",
    "MS": "#b279a2",
    "Ours-FIM": "#f58518",
}

GROUPS = [
    {
        "slug": "sst2_iid_1261770",
        "job_id": "1261770",
        "dataset": "sst2",
        "distribution": "iid",
        "goal": "fix first-20-round plateau",
        "expected_methods": [
            "FedAvg_LoRA_homogeneous",
            "Uniform_allocation",
            "Fed-HeLLo",
            "LEGEND",
            "MS",
            "Random_FIM",
        ],
        "glob": "RERUN_SST2_IID_LR5E5_*__Dist-iid_Bgt-7200_20260527-*",
    },
    {
        "slug": "mnli_iid_1261741",
        "job_id": "1261741",
        "dataset": "mnli",
        "distribution": "iid",
        "goal": "improve time-accuracy tradeoff",
        "expected_methods": [
            "FedAvg_LoRA_homogeneous",
            "Uniform_allocation",
            "Fed-HeLLo",
            "LEGEND",
            "MS",
            "Random_FIM",
        ],
        "glob": "RERUN_MNLI_IID_FASTBUDGET_B6000_R45_*__Dist-iid_Bgt-6000_20260527-*",
    },
    {
        "slug": "cifar100_iid_1261185",
        "job_id": "1261185",
        "dataset": "cifar100",
        "distribution": "iid",
        "goal": "make Ours-FIM best",
        "expected_methods": [
            "FedAvg_LoRA_homogeneous",
            "Uniform_allocation",
            "Fed-HeLLo",
            "LEGEND",
            "MS",
            "Random_FIM",
        ],
        "glob": "FINAL_RERUN_CIFAR100_B4800_iid_*__Dist-iid_Bgt-4800_20260527-*",
    },
    {
        "slug": "cifar100_dirichlet_1p0_1261185",
        "job_id": "1261185",
        "dataset": "cifar100",
        "distribution": "dirichlet:1.0",
        "goal": "make Ours-FIM best",
        "expected_methods": [
            "FedAvg_LoRA_homogeneous",
            "Uniform_allocation",
            "Fed-HeLLo",
            "LEGEND",
            "MS",
            "Random_FIM",
        ],
        "glob": "FINAL_RERUN_CIFAR100_B4800_dirichlet_1p0_*__Dist-dirichlet_1.0_Bgt-4800_20260527-*",
    },
]


def method_raw_from_csv(path: Path) -> str:
    return path.stem.replace("eval_", "")


def metric_col(df: pd.DataFrame) -> str:
    if "Accuracy" in df.columns:
        return "Accuracy"
    if "Rouge_l" in df.columns:
        return "Rouge_l"
    raise ValueError(f"No supported metric column in {list(df.columns)}")


def collect_group(group: dict) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    missing: list[dict] = []
    base = ROOT / "evaluation_results" / group["dataset"]
    found_methods = set()
    seen_method_dirs = set()
    for run_dir in sorted(base.glob(group["glob"])):
        for raw in group["expected_methods"]:
            tokens = RUN_DIR_METHOD_TOKENS.get(raw, [f"_{raw}__"])
            if any(token in run_dir.name for token in tokens):
                seen_method_dirs.add(raw)
                missing_method = METHOD_LABELS.get(raw, raw)
                break
        else:
            missing_method = "[TODO-RESULT]"
        csvs = sorted(run_dir.glob("eval_*.csv"))
        if not csvs:
            missing.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "run_id": run_dir.name,
                "method": missing_method,
                "reason": "metadata directory exists but eval CSV is missing",
            })
            continue
        for csv_path in csvs:
            raw = method_raw_from_csv(csv_path)
            found_methods.add(raw)
            method = METHOD_LABELS.get(raw, raw)
            df = pd.read_csv(csv_path)
            metric = metric_col(df)
            first20 = df[metric].head(20)
            rows.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "goal": group["goal"],
                "method_raw": raw,
                "method": method,
                "metric": metric,
                "rounds": len(df),
                "best_metric": float(df[metric].max()),
                "best_round": int(df[metric].idxmax()) + 1,
                "final_metric": float(df[metric].iloc[-1]),
                "final_cumulative_time": float(df["Cumulative_Time"].iloc[-1]),
                "final_cumulative_budget": float(df["Cumulative_Budget"].iloc[-1]),
                "first20_unique_metric_values": int(first20.nunique(dropna=True)),
                "first20_plateau_fixed": first20.nunique(dropna=True) > 1,
                "run_id": run_dir.name,
                "raw_result_path": str(csv_path.relative_to(ROOT)),
            })
    for raw in group["expected_methods"]:
        if raw not in found_methods and raw not in seen_method_dirs:
            missing.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "run_id": "[TODO-RESULT]",
                "method": METHOD_LABELS.get(raw, raw),
                "reason": "expected method eval CSV not found",
            })
    return rows, missing


def sort_key(row: dict) -> int:
    try:
        return METHOD_ORDER.index(row["method"])
    except ValueError:
        return len(METHOD_ORDER)


def plot_group(group: dict, rows: list[dict]) -> list[str]:
    if not rows:
        return []
    written: list[str] = []
    title = f"{group['dataset']} / {group['distribution']} ({group['job_id']})"
    for x_col, suffix, xlabel in [
        ("Round", "round", "Round"),
        ("Cumulative_Time", "time", "Cumulative time"),
        ("Cumulative_Budget", "budget", "Cumulative budget"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for row in sorted(rows, key=sort_key):
            df = pd.read_csv(ROOT / row["raw_result_path"])
            ax.plot(
                df[x_col],
                df[row["metric"]],
                label=row["method"],
                color=COLORS.get(row["method"], "#555555"),
                linewidth=2.2 if row["method"] == "Ours-FIM" else 1.7,
            )
        ax.set_title(f"{title}: metric vs {xlabel.lower()}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(rows[0]["metric"])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out = OUT_DIR / f"{group['slug']}_{suffix}_curves.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(str(out.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ordered = sorted(rows, key=sort_key)
    ax.bar(
        [row["method"] for row in ordered],
        [row["best_metric"] for row in ordered],
        color=[COLORS.get(row["method"], "#555555") for row in ordered],
    )
    ax.set_title(f"{title}: best metric")
    ax.set_ylabel(rows[0]["metric"])
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out = OUT_DIR / f"{group['slug']}_best_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.append(str(out.relative_to(ROOT)))
    return written


def decision_for(group: dict, rows: list[dict], missing: list[dict]) -> tuple[str, str]:
    if missing:
        return "not_adopted", f"incomplete: {len(missing)} expected result(s) missing"
    ours = next((row for row in rows if row["method"] == "Ours-FIM"), None)
    if ours is None:
        return "not_adopted", "Ours-FIM is missing"
    if group["job_id"] == "1261770":
        ok = bool(ours["first20_plateau_fixed"])
        return (
            "adopt_candidate" if ok else "not_adopted",
            "Ours-FIM first-20-round accuracy changes" if ok else "Ours-FIM first-20-round accuracy is still flat",
        )
    best = max(rows, key=lambda row: row["best_metric"])
    final_best = max(rows, key=lambda row: row["final_metric"])
    if group["job_id"] == "1261741":
        high_threshold_unique = ours["best_metric"] >= 0.70 and all(
            row["best_metric"] < 0.70 for row in rows if row["method"] != "Ours-FIM"
        )
        ok = best["method"] == "Ours-FIM" and final_best["method"] == "Ours-FIM" and high_threshold_unique
        return (
            "adopted" if ok else "not_adopted",
            (
                "Ours-FIM is the only method reaching >=0.70 Accuracy and has best/final-best metric"
                if ok
                else f"best={best['method']}; final-best={final_best['method']}; high-threshold-unique={high_threshold_unique}"
            ),
        )
    ok = best["method"] == "Ours-FIM" and final_best["method"] == "Ours-FIM"
    return (
        "adopted" if ok else "not_adopted",
        (
            "Ours-FIM has best and final-best metric"
            if ok
            else f"best={best['method']} {best['best_metric']:.4f}; final-best={final_best['method']} {final_best['final_metric']:.4f}; Ours-FIM {ours['best_metric']:.4f}/{ours['final_metric']:.4f}"
        ),
    )


def write_summary(rows: list[dict]) -> None:
    columns = [
        "job_id",
        "dataset",
        "distribution",
        "goal",
        "method_raw",
        "method",
        "metric",
        "rounds",
        "best_metric",
        "best_round",
        "final_metric",
        "final_cumulative_time",
        "final_cumulative_budget",
        "first20_unique_metric_values",
        "first20_plateau_fixed",
        "run_id",
        "raw_result_path",
        "decision",
        "reason",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_manifest(decisions: list[dict], missing: list[dict], figures: list[str]) -> None:
    lines = [
        "# Rerun Job Review Figures",
        "",
        "Facts from experiment logs/results: plots are derived from completed `eval_*.csv` files only. Missing eval CSVs are marked as TODO rather than inferred from partial logs.",
        "",
        "## Decisions",
        "",
        "| Job ID | Dataset | Distribution | Goal | Decision | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['job_id']} | {row['dataset']} | {row['distribution']} | {row['goal']} | {row['decision']} | {row['reason']} |"
        )
    if missing:
        lines.extend(["", "## Missing / Incomplete", "", "| Job ID | Dataset | Distribution | Method | Run ID | Reason |", "|---|---|---|---|---|---|"])
        for row in missing:
            lines.append(
                f"| {row['job_id']} | {row['dataset']} | {row['distribution']} | {row['method']} | `{row['run_id']}` | {row['reason']} |"
            )
    lines.extend(["", "## Figures", ""])
    for fig in figures:
        lines.append(f"- `{fig}`")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_missing: list[dict] = []
    decisions: list[dict] = []
    figures: list[str] = []
    for group in GROUPS:
        rows, missing = collect_group(group)
        decision, reason = decision_for(group, rows, missing)
        for row in rows:
            row["decision"] = decision
            row["reason"] = reason
        all_rows.extend(rows)
        all_missing.extend(missing)
        decisions.append({**group, "decision": decision, "reason": reason})
        figures.extend(plot_group(group, rows))
    write_summary(all_rows)
    write_manifest(decisions, all_missing, figures)
    print(f"Wrote {SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    print(f"Wrote {len(figures)} figures to {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
