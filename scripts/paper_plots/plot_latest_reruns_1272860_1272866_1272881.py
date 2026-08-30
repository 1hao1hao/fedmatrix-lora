#!/usr/bin/env python3
"""Review latest reruns 1272860, 1272866, and 1272881.

The script reads completed eval CSVs and writes only derived paper assets.
Missing or still-running CIFAR100 rows are recorded as TODO rather than
inferred from partial logs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_assets" / "05_figures" / "latest_rerun_review"
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "latest_rerun_review_summary.csv"
DECISION = ROOT / "paper_assets" / "02_experiments" / "latest_rerun_review_decision.md"

METHOD_LABELS = {
    "FedAvg_LoRA_homogeneous": "FedAvg-LoRA homogeneous",
    "Uniform_allocation": "Uniform allocation",
    "Fed-HeLLo": "Fed-HeLLo",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Random_FIM": "Ours-FIM",
    "Random_Taylor": "Ours-Taylor",
}
METHOD_ORDER = [
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Ours-FIM",
    "Ours-Taylor",
]
COLORS = {
    "FedAvg-LoRA homogeneous": "#7f7f7f",
    "Uniform allocation": "#4c78a8",
    "Fed-HeLLo": "#72b7b2",
    "LEGEND": "#54a24b",
    "MS": "#b279a2",
    "Ours-FIM": "#f58518",
    "Ours-Taylor": "#e45756",
}
RUN_DIR_METHOD_TOKENS = {
    "Fed-HeLLo": ["_Fed_HeLLo__"],
    "Random_FIM": ["_Random_FIM__", "_Ours_FIM__"],
    "Random_Taylor": ["_Random_Taylor__", "_Ours_Taylor__"],
}

GROUPS = [
    {
        "slug": "mnli_iid_1272860",
        "job_id": "1272860",
        "dataset": "mnli",
        "distribution": "iid",
        "goal": "improve time-accuracy tradeoff after lowering per-round budget and extending rounds",
        "config_path": "experiments/rerun_mnli_iid_b6000_r90_manifest.csv",
        "expected_methods": [
            "FedAvg_LoRA_homogeneous",
            "Uniform_allocation",
            "Fed-HeLLo",
            "LEGEND",
            "MS",
            "Random_FIM",
        ],
        "glob": "RERUN_MNLI_IID_B6000_R90_*__Dist-iid_Bgt-6000_20260530-*",
    },
    {
        "slug": "sst2_iid_1272866",
        "job_id": "1272866",
        "dataset": "sst2",
        "distribution": "iid",
        "goal": "fix first-20-round plateau with stable loader",
        "config_path": "experiments/rerun_sst2_iid_lr5e5_stable_loader_manifest.csv",
        "expected_methods": [
            "FedAvg_LoRA_homogeneous",
            "Uniform_allocation",
            "Fed-HeLLo",
            "LEGEND",
            "MS",
            "Random_FIM",
        ],
        "glob": "RERUN_SST2_IID_LR5E5_STABLE_*__Dist-iid_Bgt-7200_20260530-*",
    },
    {
        "slug": "cifar100_iid_1272881",
        "job_id": "1272881",
        "dataset": "cifar100",
        "distribution": "iid",
        "goal": "make Ours-FIM beat baselines with physical-granularity FedAvg/Uniform and no LEGEND zero-training phase",
        "config_path": "experiments/rerun_cifar100_physical_baselines_b7200_manifest.csv",
        "expected_methods": [
            "FedAvg_LoRA_homogeneous",
            "Uniform_allocation",
            "Fed-HeLLo",
            "LEGEND",
            "MS",
            "Random_FIM",
            "Random_Taylor",
        ],
        "glob": "RERUN_CIFAR100_PHYSBASE_B7200_iid_*__Dist-iid_Bgt-7200_20260530-*",
    },
    {
        "slug": "cifar100_dirichlet_1p0_1272881",
        "job_id": "1272881",
        "dataset": "cifar100",
        "distribution": "dirichlet:1.0",
        "goal": "make Ours-FIM beat baselines with physical-granularity FedAvg/Uniform and no LEGEND zero-training phase",
        "config_path": "experiments/rerun_cifar100_physical_baselines_b7200_manifest.csv",
        "expected_methods": [
            "FedAvg_LoRA_homogeneous",
            "Uniform_allocation",
            "Fed-HeLLo",
            "LEGEND",
            "MS",
            "Random_FIM",
            "Random_Taylor",
        ],
        "glob": "RERUN_CIFAR100_PHYSBASE_B7200_dirichlet_1p0_*__Dist-dirichlet_1.0_Bgt-7200_20260530-*",
    },
]


def method_label(raw: str) -> str:
    return METHOD_LABELS.get(raw, raw)


def method_raw_from_csv(path: Path) -> str:
    raw = path.stem.replace("eval_", "")
    return "Fed-HeLLo" if raw == "Fed-HeLLo" else raw


def method_tokens(raw: str) -> list[str]:
    return RUN_DIR_METHOD_TOKENS.get(raw, [f"_{raw}__"])


def metric_col(df: pd.DataFrame) -> str:
    if "Accuracy" in df.columns:
        return "Accuracy"
    if "Rouge_l" in df.columns:
        return "Rouge_l"
    raise ValueError(f"No supported metric column in {list(df.columns)}")


def run_id_for(run_dir: Path) -> str:
    meta = run_dir / "metadata.json"
    if meta.exists():
        with meta.open(encoding="utf-8") as handle:
            return json.load(handle).get("run_id", run_dir.name)
    return run_dir.name


def sort_key(row: dict) -> int:
    try:
        return METHOD_ORDER.index(row["method"])
    except ValueError:
        return len(METHOD_ORDER)


def collect_group(group: dict) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    missing: list[dict] = []
    base = ROOT / "evaluation_results" / group["dataset"]
    found = set()
    seen_dirs = set()
    for run_dir in sorted(base.glob(group["glob"])):
        dir_method = ""
        for raw in group["expected_methods"]:
            if any(token in run_dir.name for token in method_tokens(raw)):
                dir_method = raw
                seen_dirs.add(raw)
                break
        csvs = sorted(run_dir.glob("eval_*.csv"))
        if not csvs:
            missing.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "method": method_label(dir_method) if dir_method else "[TODO-RESULT]",
                "run_id": run_dir.name,
                "reason": "metadata directory exists but eval CSV is missing",
            })
            continue
        for csv_path in csvs:
            raw = method_raw_from_csv(csv_path)
            found.add(raw)
            df = pd.read_csv(csv_path)
            metric = metric_col(df)
            first20 = df[metric].head(20)
            rows.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "goal": group["goal"],
                "method_raw": raw,
                "method": method_label(raw),
                "metric": metric,
                "rounds": len(df),
                "best_metric": float(df[metric].max()),
                "best_round": int(df[metric].idxmax()) + 1,
                "final_metric": float(df[metric].iloc[-1]),
                "final_round": int(df["Round"].iloc[-1]) if "Round" in df.columns else len(df),
                "final_cumulative_time": float(df["Cumulative_Time"].iloc[-1]),
                "final_cumulative_budget": float(df["Cumulative_Budget"].iloc[-1]),
                "first20_unique_metric_values": int(first20.nunique(dropna=True)),
                "first20_plateau_fixed": first20.nunique(dropna=True) > 1,
                "run_id": run_id_for(run_dir),
                "config_path": group["config_path"],
                "raw_result_path": str(csv_path.relative_to(ROOT)),
            })
    for raw in group["expected_methods"]:
        if raw not in found and raw not in seen_dirs:
            missing.append({
                "job_id": group["job_id"],
                "dataset": group["dataset"],
                "distribution": group["distribution"],
                "method": method_label(raw),
                "run_id": "[TODO-RESULT]",
                "reason": "expected method eval CSV not found",
            })
    return rows, missing


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
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        for row in sorted(rows, key=sort_key):
            df = pd.read_csv(ROOT / row["raw_result_path"])
            ax.plot(
                df[x_col],
                df[row["metric"]],
                label=row["method"],
                color=COLORS.get(row["method"], "#555555"),
                linewidth=2.4 if row["method"] == "Ours-FIM" else 1.7,
                linestyle="--" if row["method"] == "Ours-Taylor" else "-",
            )
        ax.set_title(f"{title}: metric vs {xlabel.lower()}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(rows[0]["metric"])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / f"{group['slug']}_{suffix}_curves.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(str(out.relative_to(ROOT)))

    ordered = sorted(rows, key=sort_key)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
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
    out = FIG_DIR / f"{group['slug']}_best_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.append(str(out.relative_to(ROOT)))
    return written


def decision_for(group: dict, rows: list[dict], missing: list[dict]) -> tuple[str, str]:
    if missing:
        return "not_adopted", f"incomplete: {len(missing)} expected result(s) missing"
    ours = next((row for row in rows if row["method"] == "Ours-FIM"), None)
    if ours is None:
        return "not_adopted", "Ours-FIM eval CSV is missing"
    best = max(rows, key=lambda row: row["best_metric"])
    final_best = max(rows, key=lambda row: row["final_metric"])
    if group["dataset"] == "sst2":
        ok = bool(ours["first20_plateau_fixed"]) and best["method"] == "Ours-FIM"
        return (
            "adopt_candidate" if ok else "not_adopted",
            (
                f"Ours-FIM plateau fixed and best metric {ours['best_metric']:.4f}; final-best={final_best['method']} {final_best['final_metric']:.4f}"
                if ok
                else f"plateau_fixed={ours['first20_plateau_fixed']}; best={best['method']} {best['best_metric']:.4f}"
            ),
        )
    if group["dataset"] == "mnli":
        ok = best["method"] == "Ours-FIM" and final_best["method"] == "Ours-FIM"
        return (
            "adopt_candidate" if ok else "review_before_adoption",
            (
                f"Ours-FIM has best and final-best metric {ours['best_metric']:.4f}/{ours['final_metric']:.4f}"
                if ok
                else f"best={best['method']} {best['best_metric']:.4f}; final-best={final_best['method']} {final_best['final_metric']:.4f}; Ours-FIM={ours['best_metric']:.4f}/{ours['final_metric']:.4f}"
            ),
        )
    ok = best["method"] == "Ours-FIM" and final_best["method"] == "Ours-FIM"
    taylor = next((row for row in rows if row["method"] == "Ours-Taylor"), None)
    taylor_note = ""
    if taylor is not None:
        taylor_note = f"; Ours-Taylor={taylor['best_metric']:.4f}/{taylor['final_metric']:.4f}"
    return (
        "adopt_candidate" if ok else "not_adopted",
        (
            f"Ours-FIM has best and final-best metric {ours['best_metric']:.4f}/{ours['final_metric']:.4f}{taylor_note}"
            if ok
            else f"best={best['method']} {best['best_metric']:.4f}; final-best={final_best['method']} {final_best['final_metric']:.4f}; Ours-FIM={ours['best_metric']:.4f}/{ours['final_metric']:.4f}{taylor_note}"
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
        "final_round",
        "final_cumulative_time",
        "final_cumulative_budget",
        "first20_unique_metric_values",
        "first20_plateau_fixed",
        "run_id",
        "config_path",
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


def write_decision(decisions: list[dict], missing: list[dict], figures: list[str]) -> None:
    lines = [
        "# Latest Rerun Review: 1272860 / 1272866 / 1272881",
        "",
        "Facts from experiment logs/results: derived from completed `eval_*.csv` files and run metadata. Raw result CSVs, logs, and checkpoints were not modified.",
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
        lines.extend([
            "",
            "## Missing / Incomplete",
            "",
            "| Job ID | Dataset | Distribution | Method | Run ID | Reason |",
            "|---|---|---|---|---|---|",
        ])
        for row in missing:
            lines.append(
                f"| {row['job_id']} | {row['dataset']} | {row['distribution']} | {row['method']} | `{row['run_id']}` | {row['reason']} |"
            )
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
