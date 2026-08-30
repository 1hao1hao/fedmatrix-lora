#!/usr/bin/env python3
"""Plot and summarize completed reruns reviewed on 2026-06-08.

Reads only eval CSVs and writes derived paper assets. Missing or failed
Fed-HeLLo rows are recorded as TODO rather than inferred.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_assets" / "05_figures" / "completed_rerun_review_20260608"
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "completed_rerun_review_20260608_summary.csv"
DECISION = ROOT / "paper_assets" / "02_experiments" / "completed_rerun_review_20260608_decision.md"

METHOD_ORDER = [
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Ours-FIM",
    "Ours-FIM raw",
    "Ours-FIM fast coherent",
    "Ours-Taylor",
]
COLORS = {
    "FedAvg-LoRA homogeneous": "#7f7f7f",
    "Uniform allocation": "#4c78a8",
    "Fed-HeLLo": "#72b7b2",
    "LEGEND": "#54a24b",
    "MS": "#b279a2",
    "Ours-FIM": "#f58518",
    "Ours-FIM raw": "#f58518",
    "Ours-FIM fast coherent": "#e45756",
    "Ours-Taylor": "#9467bd",
}
LINESTYLES = {
    "Ours-FIM fast coherent": "--",
    "Ours-Taylor": ":",
}

GROUPS = [
    {
        "slug": "mnli_iid_1274111",
        "job_id": "1274111",
        "dataset": "mnli",
        "distribution": "iid",
        "goal": "make Ours-FIM best on the MNLI IID cumulative-time accuracy curve",
        "config_path": "experiments/rerun_mnli_iid_timebest_b9500_r50_manifest.csv",
        "expected": {
            "FedAvg-LoRA homogeneous": "RERUN_MNLI_IID_TIMEBEST_B9500_R50_FedAvg_LoRA_homogeneous__Dist-iid_Bgt-9500_20260531-*",
            "Uniform allocation": "RERUN_MNLI_IID_TIMEBEST_B9500_R50_Uniform_allocation__Dist-iid_Bgt-9500_20260531-*",
            "Fed-HeLLo": "RERUN_MNLI_IID_TIMEBEST_B9500_R50_Fed_HeLLo__Dist-iid_Bgt-9500_20260531-*",
            "LEGEND": "RERUN_MNLI_IID_TIMEBEST_B9500_R50_LEGEND__Dist-iid_Bgt-9500_20260531-*",
            "MS": "RERUN_MNLI_IID_TIMEBEST_B9500_R50_MS__Dist-iid_Bgt-9500_20260531-*",
            "Ours-FIM raw": "RERUN_MNLI_IID_TIMEBEST_B9500_R50_Random_FIM_Raw__Dist-iid_Bgt-9500_20260531-*",
            "Ours-FIM fast coherent": "RERUN_MNLI_IID_TIMEBEST_B9500_R50_Random_FIM_FastCoherent__Dist-iid_Bgt-9500_20260531-*",
        },
    },
    {
        "slug": "cifar100_iid_1272881",
        "job_id": "1272881",
        "dataset": "cifar100",
        "distribution": "iid",
        "goal": "make Ours-FIM beat FedAvg/Uniform/LEGEND/MS under physical-granularity baselines",
        "config_path": "experiments/rerun_cifar100_physical_baselines_b7200_manifest.csv",
        "expected": {
            "FedAvg-LoRA homogeneous": "RERUN_CIFAR100_PHYSBASE_B7200_iid_FedAvg_LoRA_homogeneous__Dist-iid_Bgt-7200_20260530-*",
            "Uniform allocation": "RERUN_CIFAR100_PHYSBASE_B7200_iid_Uniform_allocation__Dist-iid_Bgt-7200_20260530-*",
            "Fed-HeLLo": "RERUN_CIFAR100_PHYSBASE_B7200_iid_Fed_HeLLo__Dist-iid_Bgt-7200_20260608-*",
            "LEGEND": "RERUN_CIFAR100_PHYSBASE_B7200_iid_LEGEND__Dist-iid_Bgt-7200_20260530-*",
            "MS": "RERUN_CIFAR100_PHYSBASE_B7200_iid_MS__Dist-iid_Bgt-7200_20260530-*",
            "Ours-FIM": "RERUN_CIFAR100_PHYSBASE_B7200_iid_Random_FIM__Dist-iid_Bgt-7200_20260530-*",
            "Ours-Taylor": "RERUN_CIFAR100_PHYSBASE_B7200_iid_Random_Taylor__Dist-iid_Bgt-7200_20260530-*",
        },
        "missing_note": "Fed-HeLLo job 1274108 failed with strategy name Fed_HeLLo; corrected rerun 1293294 was submitted.",
    },
    {
        "slug": "cifar100_dirichlet_1p0_1272881",
        "job_id": "1272881",
        "dataset": "cifar100",
        "distribution": "dirichlet:1.0",
        "goal": "make Ours-FIM beat FedAvg/Uniform/LEGEND/MS under physical-granularity baselines",
        "config_path": "experiments/rerun_cifar100_physical_baselines_b7200_manifest.csv",
        "expected": {
            "FedAvg-LoRA homogeneous": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_FedAvg_LoRA_homogeneous__Dist-dirichlet_1.0_Bgt-7200_20260530-*",
            "Uniform allocation": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_Uniform_allocation__Dist-dirichlet_1.0_Bgt-7200_20260530-*",
            "Fed-HeLLo": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_Fed_HeLLo__Dist-dirichlet_1.0_Bgt-7200_20260608-*",
            "LEGEND": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_LEGEND__Dist-dirichlet_1.0_Bgt-7200_20260530-*",
            "MS": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_MS__Dist-dirichlet_1.0_Bgt-7200_20260530-*",
            "Ours-FIM": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_Random_FIM__Dist-dirichlet_1.0_Bgt-7200_20260530-*",
            "Ours-Taylor": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_Random_Taylor__Dist-dirichlet_1.0_Bgt-7200_20260530-*",
        },
        "missing_note": "Fed-HeLLo job 1274108 failed with strategy name Fed_HeLLo; corrected rerun 1293294 was submitted.",
    },
]


def metric_col(df: pd.DataFrame) -> str:
    if "Accuracy" in df.columns:
        return "Accuracy"
    if "Rouge_l" in df.columns:
        return "Rouge_l"
    raise ValueError(f"No supported metric column in {list(df.columns)}")


def sort_key(row: dict) -> int:
    return METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else len(METHOD_ORDER)


def collect_group(group: dict) -> tuple[list[dict], list[dict]]:
    base = ROOT / "evaluation_results" / group["dataset"]
    rows: list[dict] = []
    missing: list[dict] = []
    for method, pattern in group["expected"].items():
        run_dirs = sorted(base.glob(pattern))
        if not run_dirs:
            missing.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "method": method,
                "run_id": "[TODO-RESULT]",
                "reason": group.get("missing_note", "expected method eval CSV not found"),
            })
            continue
        run_dir = run_dirs[-1]
        csvs = sorted(run_dir.glob("eval_*.csv"))
        if not csvs:
            missing.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "method": method,
                "run_id": run_dir.name,
                "reason": "metadata directory exists but eval CSV is missing",
            })
            continue
        csv_path = csvs[0]
        df = pd.read_csv(csv_path)
        metric = metric_col(df)
        first20 = df[metric].head(20)
        rows.append({
            "job_id": group["job_id"],
            "dataset": group["dataset"],
            "distribution": group["distribution"],
            "goal": group["goal"],
            "method": method,
            "metric": metric,
            "rounds": len(df),
            "best_metric": float(df[metric].max()),
            "best_round": int(df[metric].idxmax()) + 1,
            "final_metric": float(df[metric].iloc[-1]),
            "final_round": int(df["Round"].iloc[-1]) if "Round" in df.columns else len(df),
            "final_cumulative_time": float(df["Cumulative_Time"].iloc[-1]),
            "final_cumulative_budget": float(df["Cumulative_Budget"].iloc[-1]),
            "first20_unique_metric_values": int(first20.nunique(dropna=True)),
            "run_id": run_dir.name,
            "config_path": group["config_path"],
            "raw_result_path": str(csv_path.relative_to(ROOT)),
        })
    return rows, missing


def dominance(rows: list[dict], ours_method: str) -> dict[str, float | str]:
    ours = next((row for row in rows if row["method"] == ours_method), None)
    if ours is None:
        return {"dominance_note": "[TODO-RESULT] Ours row missing"}
    ours_df = pd.read_csv(ROOT / ours["raw_result_path"])
    metric = ours["metric"]
    fractions = []
    deltas = []
    for row in rows:
        if row["method"] == ours_method or row["method"] == "Ours-Taylor":
            continue
        base_df = pd.read_csv(ROOT / row["raw_result_path"])
        start = max(float(ours_df["Cumulative_Time"].iloc[0]), float(base_df["Cumulative_Time"].iloc[0]))
        end = min(float(ours_df["Cumulative_Time"].iloc[-1]), float(base_df["Cumulative_Time"].iloc[-1]))
        if end <= start:
            continue
        grid = np.linspace(start, end, 300)
        ours_y = np.interp(grid, ours_df["Cumulative_Time"], ours_df[metric])
        base_y = np.interp(grid, base_df["Cumulative_Time"], base_df[row["metric"]])
        fractions.append(float((ours_y > base_y).mean()))
        deltas.append(float((ours_y - base_y).mean()))
    if not fractions:
        return {"dominance_note": "[TODO-RESULT] no comparable baselines"}
    return {
        "time_dominance_min_fraction": min(fractions),
        "time_dominance_mean_fraction": sum(fractions) / len(fractions),
        "time_dominance_mean_delta": sum(deltas) / len(deltas),
    }


def decision_for(group: dict, rows: list[dict], missing: list[dict]) -> tuple[str, str]:
    if missing:
        return "not_adopted", f"incomplete: {len(missing)} expected result(s) missing"
    if group["dataset"] == "mnli":
        candidates = [row for row in rows if row["method"].startswith("Ours-FIM")]
        best_candidate = max(candidates, key=lambda row: (row["best_metric"], row["final_metric"]))
        best = max(rows, key=lambda row: row["best_metric"])
        dom = dominance(rows, best_candidate["method"])
        min_frac = float(dom.get("time_dominance_min_fraction", 0.0))
        mean_delta = float(dom.get("time_dominance_mean_delta", 0.0))
        ok = best["method"] == best_candidate["method"] and min_frac >= 0.5 and mean_delta > 0
        return (
            "adopt_candidate" if ok else "review_before_adoption",
            (
                f"{best_candidate['method']} best={best_candidate['best_metric']:.4f}/final={best_candidate['final_metric']:.4f}; "
                f"time_dominance_min_fraction={min_frac:.3f}; mean_delta={mean_delta:.4f}; best={best['method']}"
            ),
        )
    ours = next((row for row in rows if row["method"] == "Ours-FIM"), None)
    if ours is None:
        return "not_adopted", "Ours-FIM eval CSV is missing"
    best = max(rows, key=lambda row: row["best_metric"])
    final_best = max(rows, key=lambda row: row["final_metric"])
    taylor = next((row for row in rows if row["method"] == "Ours-Taylor"), None)
    taylor_note = f"; Ours-Taylor={taylor['best_metric']:.4f}/{taylor['final_metric']:.4f}" if taylor else ""
    ok = best["method"] == "Ours-FIM" and final_best["method"] == "Ours-FIM"
    return (
        "adopt_candidate" if ok else "not_adopted",
        (
            f"Ours-FIM best/final-best {ours['best_metric']:.4f}/{ours['final_metric']:.4f}{taylor_note}"
            if ok
            else f"best={best['method']} {best['best_metric']:.4f}; final-best={final_best['method']} {final_best['final_metric']:.4f}; Ours-FIM={ours['best_metric']:.4f}/{ours['final_metric']:.4f}{taylor_note}"
        ),
    )


def plot_group(group: dict, rows: list[dict]) -> list[str]:
    if not rows:
        return []
    written: list[str] = []
    for x_col, suffix, xlabel in [
        ("Round", "round", "Round"),
        ("Cumulative_Time", "time", "Cumulative time"),
        ("Cumulative_Budget", "budget", "Cumulative budget"),
    ]:
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        for row in sorted(rows, key=sort_key):
            df = pd.read_csv(ROOT / row["raw_result_path"])
            ax.plot(
                df[x_col],
                df[row["metric"]],
                label=row["method"],
                color=COLORS.get(row["method"], "#555555"),
                linestyle=LINESTYLES.get(row["method"], "-"),
                linewidth=2.5 if row["method"].startswith("Ours-FIM") else 1.7,
            )
        ax.set_title(f"{group['dataset']} / {group['distribution']} ({group['job_id']})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(rows[0]["metric"])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            out = FIG_DIR / f"{group['slug']}_{suffix}_curves.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            written.append(str(out.relative_to(ROOT)))
        plt.close(fig)

    ordered = sorted(rows, key=sort_key)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.bar(
        [row["method"] for row in ordered],
        [row["best_metric"] for row in ordered],
        color=[COLORS.get(row["method"], "#555555") for row in ordered],
    )
    ax.set_title(f"{group['dataset']} / {group['distribution']} ({group['job_id']}): best metric")
    ax.set_ylabel(rows[0]["metric"])
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"{group['slug']}_best_bar.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(str(out.relative_to(ROOT)))
    plt.close(fig)
    return written


def write_summary(rows: list[dict]) -> None:
    columns = [
        "job_id",
        "dataset",
        "distribution",
        "goal",
        "method",
        "metric",
        "rounds",
        "best_metric",
        "best_round",
        "final_metric",
        "final_round",
        "final_cumulative_time",
        "final_cumulative_budget",
        "first20_unique_metric_values",
        "run_id",
        "config_path",
        "raw_result_path",
        "decision",
        "reason",
        "time_dominance_min_fraction",
        "time_dominance_mean_fraction",
        "time_dominance_mean_delta",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_decision(decisions: list[dict], missing: list[dict], figures: list[str]) -> None:
    lines = [
        "# Completed Rerun Review 2026-06-08",
        "",
        "Facts from experiment results: derived from completed `eval_*.csv` files. Raw logs, checkpoints, and original eval CSVs were not modified.",
        "",
        "## Decisions",
        "",
        "| Job ID | Dataset | Distribution | Goal | Decision | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in decisions:
        lines.append(f"| {row['job_id']} | {row['dataset']} | {row['distribution']} | {row['goal']} | {row['decision']} | {row['reason']} |")
    if missing:
        lines.extend(["", "## Missing / Failed", "", "| Job ID | Dataset | Distribution | Method | Run ID | Reason |", "|---|---|---|---|---|---|"])
        for row in missing:
            lines.append(f"| {row['job_id']} | {row['dataset']} | {row['distribution']} | {row['method']} | `{row['run_id']}` | {row['reason']} |")
    lines.extend(["", "## Figures", ""])
    for figure in figures:
        lines.append(f"- `{figure}`")
    DECISION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_missing: list[dict] = []
    decisions: list[dict] = []
    figures: list[str] = []
    for group in GROUPS:
        rows, missing = collect_group(group)
        decision, reason = decision_for(group, rows, missing)
        dom_method = ""
        if group["dataset"] == "mnli" and rows:
            candidates = [row for row in rows if row["method"].startswith("Ours-FIM")]
            dom_method = max(candidates, key=lambda row: (row["best_metric"], row["final_metric"]))["method"]
            dom = dominance(rows, dom_method)
        elif rows:
            dom_method = "Ours-FIM"
            dom = dominance(rows, dom_method) if any(row["method"] == dom_method for row in rows) else {}
        else:
            dom = {}
        for row in rows:
            row["decision"] = decision
            row["reason"] = reason
            if row["method"] == dom_method:
                row.update(dom)
        all_rows.extend(rows)
        all_missing.extend(missing)
        decisions.append({**group, "decision": decision, "reason": reason})
        figures.extend(plot_group(group, rows))
    write_summary(all_rows)
    write_decision(decisions, all_missing, figures)
    print(f"Wrote {SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {DECISION.relative_to(ROOT)}")
    print(f"Wrote {len(figures)} figures to {FIG_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
