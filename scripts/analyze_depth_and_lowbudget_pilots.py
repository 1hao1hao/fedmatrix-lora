#!/usr/bin/env python3
"""Summarize and plot the completed depth-scaling and MNLI low-budget pilots."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEPTH_SUMMARY = ROOT / "paper_assets" / "02_experiments" / "depth_scaling_pilot_summary.csv"
LOWBUDGET_MANIFEST = ROOT / "experiments" / "rerun_mnli_iid_dir1_lowbudget_fim_manifest.csv"
LOWBUDGET_SUMMARY = ROOT / "paper_assets" / "02_experiments" / "mnli_lowbudget_fim_summary.csv"
OUT_DIR = ROOT / "paper_assets" / "05_figures" / "pilot_analysis"

METRIC_PRIORITY = ("Accuracy", "Macro_F1", "Rouge_l", "Rouge_L")
METHOD_ORDER_DEPTH = ["Pure_FIM", "Old_FIM_Depth", "Lite_FIM_WeakDepth", "MS"]
METHOD_ORDER_MNLI = [
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Ours-FIM",
]
COLORS = {
    "Pure_FIM": "#4C78A8",
    "Old_FIM_Depth": "#F58518",
    "Lite_FIM_WeakDepth": "#54A24B",
    "MS": "#B279A2",
    "FedAvg-LoRA homogeneous": "#9D755D",
    "Uniform allocation": "#BAB0AC",
    "Fed-HeLLo": "#E45756",
    "LEGEND": "#72B7B2",
    "Ours-FIM": "#4C78A8",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_column(df: pd.DataFrame) -> str:
    for col in METRIC_PRIORITY:
        if col in df.columns:
            return col
    raise ValueError(f"No known metric column in {list(df.columns)}")


def relative(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_depth_rows() -> list[dict[str, str]]:
    if not DEPTH_SUMMARY.exists():
        raise SystemExit(f"Missing {relative(DEPTH_SUMMARY)}; run scripts/summarize_depth_scaling_pilot.py first.")
    with DEPTH_SUMMARY.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_curve(path: Path | str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    metric = metric_column(df)
    return df, metric


def find_lowbudget_result(row: dict[str, str]) -> tuple[dict, Path, pd.DataFrame, str]:
    pattern = f"{row['experiment_id']}__Dist-*Bgt-{row['total_budget']}_*/metadata.json"
    matches = sorted((ROOT / "evaluation_results" / row["dataset"]).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No metadata found for {row['experiment_id']}")
    metadata_path = matches[-1]
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    raw_methods = metadata.get("raw_methods") or [row["test_method"]]
    raw_method = raw_methods[0]
    raw_path = Path(metadata.get("raw_result_paths", {}).get(raw_method, ""))
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    df, metric = load_curve(raw_path)
    return metadata, raw_path, df, metric


def summarize_lowbudget() -> list[dict[str, object]]:
    rows = []
    for row in read_manifest(LOWBUDGET_MANIFEST):
        metadata, raw_path, df, metric = find_lowbudget_result(row)
        method = metadata.get("paper_methods", {}).get(metadata["raw_methods"][0], row["methods_paper"])
        final_metric = float(df[metric].iloc[-1])
        best_metric = float(df[metric].max())
        rows.append(
            {
                "dataset": metadata["dataset"],
                "distribution": metadata["distribution"],
                "method": method,
                "metric": metric,
                "final_metric": final_metric,
                "best_metric": best_metric,
                "final_best_gap": best_metric - final_metric,
                "cumulative_budget": float(df["Cumulative_Budget"].iloc[-1]) if "Cumulative_Budget" in df else "",
                "average_latency": float(df["T_Actual"].mean()) if "T_Actual" in df else "",
                "average_memory": float(df["Avg_Memory_GB"].mean()) if "Avg_Memory_GB" in df else "",
                "run_id": metadata["run_id"],
                "seed": metadata.get("seed", row["seed"]),
                "config_path": relative(LOWBUDGET_MANIFEST),
                "raw_result_path": relative(raw_path),
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["distribution"])].append(row)
    for group_rows in grouped.values():
        best = max(float(r["best_metric"]) for r in group_rows)
        final_best = max(float(r["final_metric"]) for r in group_rows)
        for row in group_rows:
            row["best_rank"] = 1 + sum(float(other["best_metric"]) > float(row["best_metric"]) for other in group_rows)
            row["final_rank"] = 1 + sum(float(other["final_metric"]) > float(row["final_metric"]) for other in group_rows)
            row["gap_to_best_metric"] = best - float(row["best_metric"])
            row["gap_to_best_final"] = final_best - float(row["final_metric"])
    return rows


def write_lowbudget_summary(rows: list[dict[str, object]]) -> None:
    LOWBUDGET_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "distribution",
        "method",
        "metric",
        "final_metric",
        "best_metric",
        "final_best_gap",
        "cumulative_budget",
        "average_latency",
        "average_memory",
        "best_rank",
        "final_rank",
        "gap_to_best_metric",
        "gap_to_best_final",
        "run_id",
        "seed",
        "config_path",
        "raw_result_path",
    ]
    with LOWBUDGET_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_depth_bars(rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows)
    for col in ("final_metric", "best_metric", "final_best_gap", "selected_depth_mean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    groups = list(df[["dataset", "distribution"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(1, len(groups), figsize=(6.8 * len(groups), 4.2), sharey=False)
    if len(groups) == 1:
        axes = [axes]
    for ax, (dataset, distribution) in zip(axes, groups):
        sub = df[(df["dataset"] == dataset) & (df["distribution"] == distribution)].copy()
        sub["method"] = pd.Categorical(sub["method"], METHOD_ORDER_DEPTH, ordered=True)
        sub = sub.sort_values("method")
        x = range(len(sub))
        ax.bar([i - 0.18 for i in x], sub["best_metric"], width=0.36, label="Best", color="#4C78A8")
        ax.bar([i + 0.18 for i in x], sub["final_metric"], width=0.36, label="Final", color="#F58518")
        ax.set_title(f"{dataset} / {distribution}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["method"], rotation=25, ha="right")
        ax.set_ylabel("Metric")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Depth-scaling pilot: best vs final metric")
    savefig(fig, "depth_scaling_best_final")


def plot_depth_curves(rows: list[dict[str, str]]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["distribution"])].append(row)
    fig, axes = plt.subplots(1, len(groups), figsize=(7.0 * len(groups), 4.4), sharey=False)
    if len(groups) == 1:
        axes = [axes]
    for ax, ((dataset, distribution), group_rows) in zip(axes, groups.items()):
        for row in sorted(group_rows, key=lambda r: METHOD_ORDER_DEPTH.index(r["method"])):
            curve, metric = load_curve(ROOT / row["raw_result_path"])
            ax.plot(curve["Round"], curve[metric], label=row["method"], linewidth=2, color=COLORS.get(row["method"]))
        ax.set_title(f"{dataset} / {distribution}")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Depth-scaling pilot: metric curves")
    savefig(fig, "depth_scaling_round_curves")


def plot_depth_selection(rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows)
    df["selected_depth_mean"] = pd.to_numeric(df["selected_depth_mean"], errors="coerce")
    df["selected_depth_std"] = pd.to_numeric(df["selected_depth_std"], errors="coerce")
    groups = list(df[["dataset", "distribution"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(1, len(groups), figsize=(6.8 * len(groups), 4.0), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (dataset, distribution) in zip(axes, groups):
        sub = df[(df["dataset"] == dataset) & (df["distribution"] == distribution)].copy()
        sub["method"] = pd.Categorical(sub["method"], METHOD_ORDER_DEPTH, ordered=True)
        sub = sub.sort_values("method")
        x = range(len(sub))
        ax.bar(x, sub["selected_depth_mean"], yerr=sub["selected_depth_std"], color="#72B7B2", capsize=3)
        ax.set_title(f"{dataset} / {distribution}")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["method"], rotation=25, ha="right")
        ax.set_ylabel("Selected depth, normalized")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Depth-scaling pilot: selected matrix depth")
    savefig(fig, "depth_scaling_selected_depth")


def plot_lowbudget_bars(rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    groups = list(df[["dataset", "distribution"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(1, len(groups), figsize=(7.2 * len(groups), 4.4), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (dataset, distribution) in zip(axes, groups):
        sub = df[(df["dataset"] == dataset) & (df["distribution"] == distribution)].copy()
        sub["method"] = pd.Categorical(sub["method"], METHOD_ORDER_MNLI, ordered=True)
        sub = sub.sort_values("method")
        x = range(len(sub))
        colors = [COLORS.get(m, "#999999") for m in sub["method"].astype(str)]
        ax.bar(x, sub["best_metric"], color=colors)
        ax.scatter(x, sub["final_metric"], marker="D", color="black", s=26, label="Final")
        ax.set_title(f"{dataset} / {distribution}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["method"], rotation=25, ha="right")
        ax.set_ylabel("Best Accuracy")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(frameon=False)
    fig.suptitle("MNLI low-budget FIM-only rerun: best metric, final marker")
    savefig(fig, "mnli_lowbudget_best_final")


def plot_lowbudget_curves(rows: list[dict[str, object]]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["distribution"])].append(row)
    fig, axes = plt.subplots(1, len(grouped), figsize=(7.2 * len(grouped), 4.4), sharey=True)
    if len(grouped) == 1:
        axes = [axes]
    for ax, ((dataset, distribution), group_rows) in zip(axes, grouped.items()):
        for row in sorted(group_rows, key=lambda r: METHOD_ORDER_MNLI.index(str(r["method"]))):
            curve, metric = load_curve(ROOT / str(row["raw_result_path"]))
            ax.plot(curve["Round"], curve[metric], label=str(row["method"]), linewidth=2, color=COLORS.get(str(row["method"])))
        ax.set_title(f"{dataset} / {distribution}")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("MNLI low-budget FIM-only rerun: round curves")
    savefig(fig, "mnli_lowbudget_round_curves")


def print_depth_conclusion(rows: list[dict[str, str]]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["distribution"])].append(row)
    print("Depth-scaling pilot:")
    for key, group_rows in grouped.items():
        by_method = {row["method"]: row for row in group_rows}
        best_winner = max(group_rows, key=lambda r: float(r["best_metric"]))
        final_winner = max(group_rows, key=lambda r: float(r["final_metric"]))
        pure = by_method.get("Pure_FIM")
        old = by_method.get("Old_FIM_Depth")
        lite = by_method.get("Lite_FIM_WeakDepth")
        print(
            f"  {key[0]} / {key[1]}: best winner={best_winner['method']} ({float(best_winner['best_metric']):.4f}), "
            f"final winner={final_winner['method']} ({float(final_winner['final_metric']):.4f})"
        )
        if pure and old and lite:
            print(
                "    Pure vs Old best delta="
                f"{float(pure['best_metric']) - float(old['best_metric']):+.4f}; "
                "Lite vs Old best delta="
                f"{float(lite['best_metric']) - float(old['best_metric']):+.4f}"
            )


def print_lowbudget_conclusion(rows: list[dict[str, object]]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["distribution"])].append(row)
    print("MNLI low-budget FIM-only rerun:")
    for key, group_rows in grouped.items():
        ours = next((r for r in group_rows if r["method"] == "Ours-FIM"), None)
        best = max(group_rows, key=lambda r: float(r["best_metric"]))
        final = max(group_rows, key=lambda r: float(r["final_metric"]))
        print(
            f"  {key[0]} / {key[1]}: best winner={best['method']} ({float(best['best_metric']):.4f}), "
            f"final winner={final['method']} ({float(final['final_metric']):.4f})"
        )
        if ours:
            print(
                f"    Ours-FIM best={float(ours['best_metric']):.4f}, rank={ours['best_rank']}; "
                f"final={float(ours['final_metric']):.4f}, rank={ours['final_rank']}; "
                f"gap_to_best={float(ours['gap_to_best_metric']):.4f}"
            )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    depth_rows = load_depth_rows()
    lowbudget_rows = summarize_lowbudget()
    write_lowbudget_summary(lowbudget_rows)

    plot_depth_bars(depth_rows)
    plot_depth_curves(depth_rows)
    plot_depth_selection(depth_rows)
    plot_lowbudget_bars(lowbudget_rows)
    plot_lowbudget_curves(lowbudget_rows)

    print_depth_conclusion(depth_rows)
    print_lowbudget_conclusion(lowbudget_rows)
    print(f"Wrote {relative(LOWBUDGET_SUMMARY)}")
    print(f"Wrote figures to {relative(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
