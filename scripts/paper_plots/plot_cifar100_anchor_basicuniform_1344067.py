#!/usr/bin/env python3
"""Plot and review CIFAR100 anchor-balanced Ours rerun job 1344067.

Reads completed eval CSVs and writes derived paper assets only.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_assets" / "05_figures" / "cifar100_anchor_basicuniform_1344067"
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "cifar100_anchor_basicuniform_1344067_summary.csv"
DECISION = ROOT / "paper_assets" / "02_experiments" / "cifar100_anchor_basicuniform_1344067_decision.md"

JOB_ID = "1344067"
CONFIG_PATH = "experiments/rerun_cifar100_anchor_basicuniform_b7200_r180_manifest.csv"
METRIC = "Accuracy"

METHODS = [
    ("FedAvg-LoRA basic random", "FedAvg_LoRA_homogeneous", "#777777"),
    ("Uniform matrix allocation", "Uniform_allocation", "#4c78a8"),
    ("Ours-FIM anchor-balanced", "Random_FIM", "#f58518"),
]

GROUPS = [
    {
        "slug": "cifar100_iid_b7200_r180",
        "dataset": "cifar100",
        "distribution": "iid",
        "setting": "B7200/R180",
        "patterns": {
            "FedAvg-LoRA basic random": "RERUN_CIFAR100_ANCHOR_BASICUNI_B7200_R180_iid_FedAvg_LoRA_homogeneous__Dist-iid_Bgt-7200_20260613-*",
            "Uniform matrix allocation": "RERUN_CIFAR100_ANCHOR_BASICUNI_B7200_R180_iid_Uniform_allocation__Dist-iid_Bgt-7200_20260613-*",
            "Ours-FIM anchor-balanced": "RERUN_CIFAR100_ANCHOR_BASICUNI_B7200_R180_iid_Random_FIM__Dist-iid_Bgt-7200_20260613-*",
        },
    },
    {
        "slug": "cifar100_dirichlet_1p0_b7200_r180",
        "dataset": "cifar100",
        "distribution": "dirichlet:1.0",
        "setting": "B7200/R180",
        "patterns": {
            "FedAvg-LoRA basic random": "RERUN_CIFAR100_ANCHOR_BASICUNI_B7200_R180_dirichlet_1p0_FedAvg_LoRA_homogeneous__Dist-dirichlet_1.0_Bgt-7200_20260613-*",
            "Uniform matrix allocation": "RERUN_CIFAR100_ANCHOR_BASICUNI_B7200_R180_dirichlet_1p0_Uniform_allocation__Dist-dirichlet_1.0_Bgt-7200_20260613-*",
            "Ours-FIM anchor-balanced": "RERUN_CIFAR100_ANCHOR_BASICUNI_B7200_R180_dirichlet_1p0_Random_FIM__Dist-dirichlet_1.0_Bgt-7200_20260613-*",
        },
    },
]


def method_order(label: str) -> int:
    return [label for label, _, _ in METHODS].index(label)


def method_color(label: str) -> str:
    return {label: color for label, _, color in METHODS}.get(label, "#555555")


def method_raw(label: str) -> str:
    return {label: raw for label, raw, _ in METHODS}[label]


def read_metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect_group(group: dict) -> tuple[list[dict], list[dict]]:
    base = ROOT / "evaluation_results" / group["dataset"]
    rows: list[dict] = []
    missing: list[dict] = []
    for method_label, pattern in group["patterns"].items():
        run_dirs = sorted(base.glob(pattern))
        if not run_dirs:
            missing.append({**group, "method": method_label, "run_id": "[TODO-RESULT]", "reason": "run directory missing"})
            continue
        run_dir = run_dirs[-1]
        raw_method = method_raw(method_label)
        csv_path = run_dir / f"eval_{raw_method}.csv"
        if not csv_path.exists():
            missing.append({**group, "method": method_label, "run_id": run_dir.name, "reason": f"{csv_path.name} missing"})
            continue
        df = pd.read_csv(csv_path)
        if METRIC not in df.columns:
            missing.append({**group, "method": method_label, "run_id": run_dir.name, "reason": "Accuracy column missing"})
            continue
        meta = read_metadata(run_dir)
        first20 = df[METRIC].head(20)
        rows.append({
            "job_id": JOB_ID,
            "dataset": group["dataset"],
            "distribution": group["distribution"],
            "setting": group["setting"],
            "method": method_label,
            "method_raw": raw_method,
            "seed": meta.get("seed", "[TODO-SEED]"),
            "metric": METRIC,
            "rounds": len(df),
            "best_metric": float(df[METRIC].max()),
            "best_round": int(df[METRIC].idxmax()) + 1,
            "final_metric": float(df[METRIC].iloc[-1]),
            "final_round": int(df["Round"].iloc[-1]),
            "final_cumulative_time": float(df["Cumulative_Time"].iloc[-1]),
            "final_cumulative_budget": float(df["Cumulative_Budget"].iloc[-1]),
            "first20_unique_metric_values": int(first20.nunique(dropna=True)),
            "run_id": run_dir.name,
            "config_path": CONFIG_PATH,
            "raw_result_path": str(csv_path.relative_to(ROOT)),
        })
    return rows, missing


def save_fig(fig, path: Path) -> list[str]:
    written = []
    for suffix in [".png", ".pdf"]:
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(str(out.relative_to(ROOT)))
    return written


def plot_group(group: dict, rows: list[dict]) -> list[str]:
    if not rows:
        return []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for x_col, suffix, xlabel in [
        ("Round", "round", "Round"),
        ("Cumulative_Time", "time", "Cumulative time"),
        ("Cumulative_Budget", "budget", "Cumulative budget"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for row in sorted(rows, key=lambda item: method_order(item["method"])):
            df = pd.read_csv(ROOT / row["raw_result_path"])
            ax.plot(
                df[x_col],
                df[METRIC],
                label=row["method"],
                color=method_color(row["method"]),
                linewidth=2.6 if row["method"].startswith("Ours") else 1.9,
            )
        ax.set_title(f"CIFAR100 {group['distribution']} {group['setting']}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        written.extend(save_fig(fig, FIG_DIR / f"{group['slug']}_{suffix}_curves"))
        plt.close(fig)

    ordered = sorted(rows, key=lambda item: method_order(item["method"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(
        [row["method"] for row in ordered],
        [row["best_metric"] for row in ordered],
        color=[method_color(row["method"]) for row in ordered],
    )
    ax.set_title(f"CIFAR100 {group['distribution']} {group['setting']}: best accuracy")
    ax.set_ylabel(METRIC)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    written.extend(save_fig(fig, FIG_DIR / f"{group['slug']}_best_bar"))
    plt.close(fig)
    return written


def decide(rows: list[dict], missing: list[dict]) -> tuple[str, str]:
    if missing:
        return "not_adopted", f"incomplete: {len(missing)} expected result(s) missing"
    ours = next((row for row in rows if row["method"].startswith("Ours-FIM")), None)
    if ours is None:
        return "not_adopted", "Ours-FIM anchor-balanced is missing"
    best = max(rows, key=lambda row: row["best_metric"])
    final_best = max(rows, key=lambda row: row["final_metric"])
    if best["method"] == ours["method"] and final_best["method"] == ours["method"]:
        return "adopt_candidate", f"Ours best/final-best = {ours['best_metric']:.4f}/{ours['final_metric']:.4f}"
    return (
        "not_adopted",
        f"best={best['method']} {best['best_metric']:.4f}; "
        f"final-best={final_best['method']} {final_best['final_metric']:.4f}; "
        f"Ours={ours['best_metric']:.4f}/{ours['final_metric']:.4f}",
    )


def write_summary(rows: list[dict]) -> None:
    fields = [
        "job_id", "dataset", "distribution", "setting", "method", "method_raw",
        "seed", "metric", "rounds", "best_metric", "best_round", "final_metric",
        "final_round", "final_cumulative_time", "final_cumulative_budget",
        "first20_unique_metric_values", "run_id", "config_path", "raw_result_path",
        "decision", "reason",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_decision(decisions: list[dict], missing: list[dict], figures: list[str]) -> None:
    lines = [
        "# CIFAR100 Anchor-Balanced Ours-FIM 1344067 Review",
        "",
        "Facts from experiment results: derived from completed `eval_*.csv` files. Raw logs, checkpoints, and original eval CSVs were not modified.",
        "",
        "## Decisions",
        "",
        "| Job ID | Distribution | Setting | Decision | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in decisions:
        lines.append(f"| {row['job_id']} | {row['distribution']} | {row['setting']} | {row['decision']} | {row['reason']} |")
    if missing:
        lines.extend(["", "## Missing", "", "| Distribution | Method | Run ID | Reason |", "|---|---|---|---|"])
        for row in missing:
            lines.append(f"| {row['distribution']} | {row['method']} | {row['run_id']} | {row['reason']} |")
    lines.extend(["", "## Generated Figures", ""])
    for fig in figures:
        lines.append(f"- `{fig}`")
    DECISION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_rows: list[dict] = []
    all_missing: list[dict] = []
    decisions: list[dict] = []
    figures: list[str] = []
    for group in GROUPS:
        rows, missing = collect_group(group)
        decision, reason = decide(rows, missing)
        for row in rows:
            row["decision"] = decision
            row["reason"] = reason
        all_rows.extend(rows)
        all_missing.extend(missing)
        decisions.append({**group, "job_id": JOB_ID, "decision": decision, "reason": reason})
        figures.extend(plot_group(group, rows))
    write_summary(all_rows)
    write_decision(decisions, all_missing, figures)
    print(f"Wrote {SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {DECISION.relative_to(ROOT)}")
    for fig in figures:
        print(f"Wrote {fig}")


if __name__ == "__main__":
    main()
