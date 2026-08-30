#!/usr/bin/env python3
"""Plot CIFAR100 roundscan job 1293671 and Fed-HeLLo supplement 1293294.

The script reads completed eval CSVs and writes only derived paper assets.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_assets" / "05_figures" / "cifar100_roundscan_1293671_1293294"
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "cifar100_roundscan_1293671_1293294_summary.csv"
DECISION = ROOT / "paper_assets" / "02_experiments" / "cifar100_roundscan_1293671_1293294_decision.md"

COLORS = {
    "FedAvg-LoRA homogeneous": "#7f7f7f",
    "Uniform allocation": "#4c78a8",
    "Fed-HeLLo": "#72b7b2",
    "Ours-FIM": "#f58518",
}
ORDER = ["FedAvg-LoRA homogeneous", "Uniform allocation", "Fed-HeLLo", "Ours-FIM"]

GROUPS = [
    {
        "slug": "cifar100_iid_b7200_r240",
        "job_id": "1293671",
        "dataset": "cifar100",
        "distribution": "iid",
        "setting": "B7200/R240",
        "config_path": "experiments/rerun_cifar100_budget_roundscan_fedavg_uniform_ours_manifest.csv",
        "expected": {
            "FedAvg-LoRA homogeneous": "RERUN_CIFAR100_ROUNDSCAN_B7200_R240_iid_FedAvg_LoRA_homogeneous__Dist-iid_Bgt-7200_20260608-*",
            "Uniform allocation": "RERUN_CIFAR100_ROUNDSCAN_B7200_R240_iid_Uniform_allocation__Dist-iid_Bgt-7200_20260608-*",
            "Ours-FIM": "RERUN_CIFAR100_ROUNDSCAN_B7200_R240_iid_Random_FIM__Dist-iid_Bgt-7200_20260608-*",
        },
    },
    {
        "slug": "cifar100_dirichlet_1p0_b7200_r240",
        "job_id": "1293671",
        "dataset": "cifar100",
        "distribution": "dirichlet:1.0",
        "setting": "B7200/R240",
        "config_path": "experiments/rerun_cifar100_budget_roundscan_fedavg_uniform_ours_manifest.csv",
        "expected": {
            "FedAvg-LoRA homogeneous": "RERUN_CIFAR100_ROUNDSCAN_B7200_R240_dirichlet_1p0_FedAvg_LoRA_homogeneous__Dist-dirichlet_1.0_Bgt-7200_20260608-*",
            "Uniform allocation": "RERUN_CIFAR100_ROUNDSCAN_B7200_R240_dirichlet_1p0_Uniform_allocation__Dist-dirichlet_1.0_Bgt-7200_20260608-*",
            "Ours-FIM": "RERUN_CIFAR100_ROUNDSCAN_B7200_R240_dirichlet_1p0_Random_FIM__Dist-dirichlet_1.0_Bgt-7200_20260608-*",
        },
    },
    {
        "slug": "cifar100_iid_b7200_r300",
        "job_id": "1293671",
        "dataset": "cifar100",
        "distribution": "iid",
        "setting": "B7200/R300",
        "config_path": "experiments/rerun_cifar100_budget_roundscan_fedavg_uniform_ours_manifest.csv",
        "expected": {
            "FedAvg-LoRA homogeneous": "RERUN_CIFAR100_ROUNDSCAN_B7200_R300_iid_FedAvg_LoRA_homogeneous__Dist-iid_Bgt-7200_20260608-*",
            "Uniform allocation": "RERUN_CIFAR100_ROUNDSCAN_B7200_R300_iid_Uniform_allocation__Dist-iid_Bgt-7200_20260608-*",
            "Ours-FIM": "RERUN_CIFAR100_ROUNDSCAN_B7200_R300_iid_Random_FIM__Dist-iid_Bgt-7200_20260608-*",
        },
    },
    {
        "slug": "cifar100_dirichlet_1p0_b7200_r300",
        "job_id": "1293671",
        "dataset": "cifar100",
        "distribution": "dirichlet:1.0",
        "setting": "B7200/R300",
        "config_path": "experiments/rerun_cifar100_budget_roundscan_fedavg_uniform_ours_manifest.csv",
        "expected": {
            "FedAvg-LoRA homogeneous": "RERUN_CIFAR100_ROUNDSCAN_B7200_R300_dirichlet_1p0_FedAvg_LoRA_homogeneous__Dist-dirichlet_1.0_Bgt-7200_20260608-*",
            "Uniform allocation": "RERUN_CIFAR100_ROUNDSCAN_B7200_R300_dirichlet_1p0_Uniform_allocation__Dist-dirichlet_1.0_Bgt-7200_20260608-*",
            "Ours-FIM": "RERUN_CIFAR100_ROUNDSCAN_B7200_R300_dirichlet_1p0_Random_FIM__Dist-dirichlet_1.0_Bgt-7200_20260608-*",
        },
    },
    {
        "slug": "cifar100_physbase_b7200_r180_fedhello_supplement",
        "job_id": "1272881+1293294",
        "dataset": "cifar100",
        "distribution": "iid+dirichlet:1.0",
        "setting": "B7200/R180 Fed-HeLLo supplement",
        "config_path": "experiments/rerun_cifar100_physical_baselines_b7200_fedhello_manifest.csv",
        "expected": {
            "Fed-HeLLo iid": "RERUN_CIFAR100_PHYSBASE_B7200_iid_Fed_HeLLo__Dist-iid_Bgt-7200_20260608-*",
            "Fed-HeLLo dirichlet:1.0": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_Fed_HeLLo__Dist-dirichlet_1.0_Bgt-7200_20260608-*",
        },
        "bar_only": True,
    },
]


def metric_col(df: pd.DataFrame) -> str:
    if "Accuracy" in df.columns:
        return "Accuracy"
    if "Rouge_l" in df.columns:
        return "Rouge_l"
    raise ValueError(f"No supported metric column in {list(df.columns)}")


def method_rank(method: str) -> int:
    base = method.replace(" iid", "").replace(" dirichlet:1.0", "")
    return ORDER.index(base) if base in ORDER else len(ORDER)


def collect_group(group: dict) -> tuple[list[dict], list[dict]]:
    base = ROOT / "evaluation_results" / group["dataset"]
    rows = []
    missing = []
    for method, pattern in group["expected"].items():
        run_dirs = sorted(base.glob(pattern))
        if not run_dirs:
            missing.append({**group, "method": method, "run_id": "[TODO-RESULT]", "reason": "eval CSV not found"})
            continue
        run_dir = run_dirs[-1]
        csvs = sorted(run_dir.glob("eval_*.csv"))
        if not csvs:
            missing.append({**group, "method": method, "run_id": run_dir.name, "reason": "metadata exists but eval CSV is missing"})
            continue
        csv_path = csvs[0]
        df = pd.read_csv(csv_path)
        metric = metric_col(df)
        first20 = df[metric].head(20)
        rows.append({
            "job_id": group["job_id"],
            "dataset": group["dataset"],
            "distribution": group["distribution"],
            "setting": group["setting"],
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


def decision_for(rows: list[dict], missing: list[dict]) -> tuple[str, str]:
    if missing:
        return "not_adopted", f"incomplete: {len(missing)} expected result(s) missing"
    if not rows or all(row["method"].startswith("Fed-HeLLo") for row in rows):
        return "diagnostic_only", "Fed-HeLLo supplement only; does not decide Ours-FIM adoption"
    ours = next((row for row in rows if row["method"] == "Ours-FIM"), None)
    if ours is None:
        return "not_adopted", "Ours-FIM is missing"
    best = max(rows, key=lambda row: row["best_metric"])
    final_best = max(rows, key=lambda row: row["final_metric"])
    ok = best["method"] == "Ours-FIM" and final_best["method"] == "Ours-FIM"
    return (
        "adopt_candidate" if ok else "not_adopted",
        (
            f"Ours-FIM best/final-best {ours['best_metric']:.4f}/{ours['final_metric']:.4f}"
            if ok
            else f"best={best['method']} {best['best_metric']:.4f}; final-best={final_best['method']} {final_best['final_metric']:.4f}; Ours-FIM={ours['best_metric']:.4f}/{ours['final_metric']:.4f}"
        ),
    )


def save_fig(fig, path: Path) -> list[str]:
    written = []
    for ext in ["png", "pdf"]:
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(str(out.relative_to(ROOT)))
    return written


def plot_group(group: dict, rows: list[dict]) -> list[str]:
    if not rows:
        return []
    written = []
    if not group.get("bar_only"):
        for x_col, suffix, xlabel in [
            ("Round", "round", "Round"),
            ("Cumulative_Time", "time", "Cumulative time"),
            ("Cumulative_Budget", "budget", "Cumulative budget"),
        ]:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for row in sorted(rows, key=lambda row: method_rank(row["method"])):
                df = pd.read_csv(ROOT / row["raw_result_path"])
                ax.plot(
                    df[x_col],
                    df[row["metric"]],
                    label=row["method"],
                    color=COLORS.get(row["method"], "#555555"),
                    linewidth=2.5 if row["method"] == "Ours-FIM" else 1.8,
                )
            ax.set_title(f"{group['dataset']} {group['distribution']} {group['setting']}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(rows[0]["metric"])
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            written.extend(save_fig(fig, FIG_DIR / f"{group['slug']}_{suffix}_curves"))
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ordered = sorted(rows, key=lambda row: method_rank(row["method"]))
    ax.bar(
        [row["method"] for row in ordered],
        [row["best_metric"] for row in ordered],
        color=[COLORS.get(row["method"].replace(" iid", "").replace(" dirichlet:1.0", ""), "#555555") for row in ordered],
    )
    ax.set_title(f"{group['dataset']} {group['distribution']} {group['setting']}: best")
    ax.set_ylabel(rows[0]["metric"])
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    written.extend(save_fig(fig, FIG_DIR / f"{group['slug']}_best_bar"))
    plt.close(fig)
    return written


def write_summary(rows: list[dict]) -> None:
    cols = [
        "job_id", "dataset", "distribution", "setting", "method", "metric",
        "rounds", "best_metric", "best_round", "final_metric", "final_round",
        "final_cumulative_time", "final_cumulative_budget", "first20_unique_metric_values",
        "run_id", "config_path", "raw_result_path", "decision", "reason",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in cols})


def write_decision(decisions: list[dict], missing: list[dict], figures: list[str]) -> None:
    lines = [
        "# CIFAR100 Roundscan 1293671 / Fed-HeLLo 1293294 Review",
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
        lines.extend(["", "## Missing", "", "| Job ID | Distribution | Setting | Method | Run ID | Reason |", "|---|---|---|---|---|---|"])
        for row in missing:
            lines.append(f"| {row['job_id']} | {row['distribution']} | {row['setting']} | {row['method']} | `{row['run_id']}` | {row['reason']} |")
    lines.extend([
        "",
        "## Runtime Note",
        "",
        "- No crash, timeout, CUDA OOM, DataLoader-worker, or disk-space error was found in logs for jobs `1293671` and `1293294`.",
        "- Logs report `Workers: 4 on A800`. The long wall time is mainly due to 240/300 FL rounds, 10 clients trained in worker batches, frequent evaluation, trace logging, and selection-analysis logging.",
        "",
        "## Figures",
        "",
    ])
    for fig in figures:
        lines.append(f"- `{fig}`")
    DECISION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_missing = []
    decisions = []
    figures = []
    for group in GROUPS:
        rows, missing = collect_group(group)
        decision, reason = decision_for(rows, missing)
        for row in rows:
            row["decision"] = decision
            row["reason"] = reason
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
