#!/usr/bin/env python3
"""Summarize and plot the final low-budget reruns and clean C5 ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_assets" / "05_figures" / "rerun_1260098_1260123"
EXP_DIR = ROOT / "paper_assets" / "02_experiments"

FINAL_MANIFEST = ROOT / "experiments" / "final_lowbudget_stable_rerun_manifest.csv"
C5_MANIFEST = ROOT / "experiments" / "ablation_clean_c5_lowbudget_manifest.csv"

FINAL_SUMMARY = EXP_DIR / "final_lowbudget_stable_rerun_summary.csv"
C5_SUMMARY = EXP_DIR / "ablation_clean_c5_summary.csv"
FINAL_DECISION = EXP_DIR / "final_lowbudget_stable_rerun_decision.md"

METHOD_LABELS = {
    "FedAvg_LoRA_homogeneous": "FedAvg-LoRA",
    "Uniform_allocation": "Uniform",
    "Fed-HeLLo": "Fed-HeLLo",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Random_FIM": "Ours-FIM",
    "Greedy_ValueCost": "Greedy",
}

COLORS = {
    "FedAvg-LoRA": "#7f7f7f",
    "Uniform": "#bcbd22",
    "Fed-HeLLo": "#9467bd",
    "LEGEND": "#8c564b",
    "MS": "#ff7f0e",
    "Ours-FIM": "#1f77b4",
    "Greedy": "#d62728",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_run_dir(dataset: str, experiment_id: str) -> Path | None:
    base = ROOT / "evaluation_results" / dataset
    matches = sorted(base.glob(f"{experiment_id}__*"))
    if not matches:
        return None
    return matches[-1]


def metric_col(df: pd.DataFrame) -> str:
    if "Accuracy" in df.columns:
        return "Accuracy"
    if "Rouge_l" in df.columns:
        return "Rouge_l"
    raise ValueError(f"No supported metric column in {list(df.columns)}")


def summarize_eval(path: Path) -> dict[str, float | int | str]:
    df = pd.read_csv(path)
    if df.empty:
        return {
            "status": "empty",
            "rounds": 0,
            "metric": "Accuracy",
            "best_metric": float("nan"),
            "final_metric": float("nan"),
            "final_best_gap": float("nan"),
            "final_cumulative_budget": float("nan"),
            "final_cumulative_time": float("nan"),
            "average_latency": float("nan"),
            "average_memory": float("nan"),
            "average_communication": float("nan"),
        }
    mcol = metric_col(df)
    final_metric = float(df[mcol].iloc[-1])
    best_metric = float(df[mcol].max())
    return {
        "status": "complete",
        "rounds": int(len(df)),
        "metric": mcol,
        "best_metric": best_metric,
        "final_metric": final_metric,
        "final_best_gap": best_metric - final_metric,
        "final_cumulative_budget": float(df.get("Cumulative_Budget", pd.Series([float("nan")])).iloc[-1]),
        "final_cumulative_time": float(df.get("Cumulative_Time", pd.Series([float("nan")])).iloc[-1]),
        "average_latency": float(df.get("T_Actual", pd.Series(dtype=float)).mean()),
        "average_memory": float(df.get("Avg_Memory_GB", pd.Series(dtype=float)).mean()),
        "average_communication": float(df.get("Avg_Comm_MB", pd.Series(dtype=float)).mean()),
    }


def summarize_manifest(path: Path, multi_method: bool = False) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(read_manifest(path), start=1):
        run_dir = find_run_dir(row["dataset"], row["experiment_id"])
        methods = [m.strip() for m in row["test_method"].split(",") if m.strip()]
        if run_dir is None:
            for method in methods:
                rows.append({
                    "manifest": str(path.relative_to(ROOT)),
                    "manifest_row": idx,
                    "experiment_id": row["experiment_id"],
                    "dataset": row["dataset"],
                    "distribution": row["distribution_arg"],
                    "method_raw": method,
                    "method": METHOD_LABELS.get(method, method),
                    "seed": row["seed"],
                    "status": "[TODO-RESULT]",
                    "run_id": "[TODO-RESULT]",
                    "config_path": str(path.relative_to(ROOT)),
                    "raw_result_path": "[TODO-RESULT]",
                })
            continue
        metadata_path = run_dir / "metadata.json"
        run_id = run_dir.name
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as handle:
                run_id = json.load(handle).get("run_id", run_id)
        for method in methods:
            raw_path = run_dir / f"eval_{method}.csv"
            summary = summarize_eval(raw_path) if raw_path.exists() else {"status": "[TODO-RESULT]"}
            rows.append({
                "manifest": str(path.relative_to(ROOT)),
                "manifest_row": idx,
                "experiment_id": row["experiment_id"],
                "dataset": row["dataset"],
                "distribution": row["distribution_arg"],
                "method_raw": method,
                "method": METHOD_LABELS.get(method, method),
                "seed": row["seed"],
                "run_id": run_id,
                "config_path": str(path.relative_to(ROOT)),
                "raw_result_path": str(raw_path.relative_to(ROOT)),
                **summary,
            })
    df = pd.DataFrame(rows)
    if not df.empty and "best_metric" in df.columns:
        df["best_rank"] = df.groupby(["dataset", "distribution"])["best_metric"].rank(ascending=False, method="min")
        df["final_rank"] = df.groupby(["dataset", "distribution"])["final_metric"].rank(ascending=False, method="min")
    return df


def savefig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_best(df: pd.DataFrame) -> None:
    complete = df[df["status"].eq("complete")].copy()
    groups = list(complete.groupby(["dataset", "distribution"], sort=True))
    fig, axes = plt.subplots(len(groups), 1, figsize=(9.5, max(3.0, 2.7 * len(groups))), squeeze=False)
    for ax, ((dataset, dist), group) in zip(axes[:, 0], groups):
        group = group.sort_values("best_metric", ascending=False)
        labels = group["method"].tolist()
        values = group["best_metric"].tolist()
        ax.bar(labels, values, color=[COLORS.get(x, "#999999") for x in labels])
        ax.set_title(f"{dataset} / {dist} best metric")
        ax.set_ylabel(group["metric"].iloc[0])
        ax.set_ylim(max(0, min(values) - 0.05), min(1.0, max(values) + 0.05))
        ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(values):
            ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    savefig(fig, "final_lowbudget_stable_grouped_best")


def plot_round_curves(df: pd.DataFrame, name: str, title: str) -> None:
    complete = df[df["status"].eq("complete")].copy()
    groups = list(complete.groupby(["dataset", "distribution"], sort=True))
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, max(3.2, 3.0 * len(groups))), squeeze=False)
    for ax, ((dataset, dist), group) in zip(axes[:, 0], groups):
        for _, row in group.sort_values("method").iterrows():
            raw_path = ROOT / row["raw_result_path"]
            curve = pd.read_csv(raw_path)
            mcol = metric_col(curve)
            ax.plot(curve["Round"], curve[mcol], label=row["method"], linewidth=2.0 if row["method"] == "Ours-FIM" else 1.4, color=COLORS.get(row["method"]))
        ax.set_title(f"{dataset} / {dist}")
        ax.set_xlabel("Round")
        ax.set_ylabel(group["metric"].iloc[0])
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    savefig(fig, name)


def write_decision(final_df: pd.DataFrame, c5_df: pd.DataFrame) -> None:
    lines = [
        "# Final Rerun Decision Notes",
        "",
        "Sources:",
        "",
        f"- Final rerun summary: `{FINAL_SUMMARY.relative_to(ROOT)}`",
        f"- Clean C5 summary: `{C5_SUMMARY.relative_to(ROOT)}`",
        f"- Final rerun manifest: `{FINAL_MANIFEST.relative_to(ROOT)}`",
        f"- Clean C5 manifest: `{C5_MANIFEST.relative_to(ROOT)}`",
        "",
        "## 1260098 final low-budget rerun",
        "",
    ]
    for (dataset, dist), group in final_df[final_df["status"].eq("complete")].groupby(["dataset", "distribution"], sort=True):
        ours = group[group["method"].eq("Ours-FIM")]
        best = group.sort_values("best_metric", ascending=False).iloc[0]
        final = group.sort_values("final_metric", ascending=False).iloc[0]
        if ours.empty:
            verdict = "[TODO-RESULT] Ours-FIM missing."
        else:
            o = ours.iloc[0]
            verdict = (
                f"Ours-FIM best={o['best_metric']:.4f} rank={int(o['best_rank'])}; "
                f"final={o['final_metric']:.4f} rank={int(o['final_rank'])}. "
                f"Best method={best['method']} ({best['best_metric']:.4f}); "
                f"final-best method={final['method']} ({final['final_metric']:.4f})."
            )
        lines.append(f"- `{dataset} / {dist}`: {verdict}")
    lines.extend(["", "## 1260123 clean C5 scheduling ablation", ""])
    for (dataset, dist), group in c5_df[c5_df["status"].eq("complete")].groupby(["dataset", "distribution"], sort=True):
        vals = {row["method"]: row for _, row in group.iterrows()}
        ours = vals.get("Ours-FIM")
        greedy = vals.get("Greedy")
        if ours is None or greedy is None:
            lines.append(f"- `{dataset} / {dist}`: [TODO-RESULT] paired C5 rows are incomplete.")
            continue
        lines.append(
            f"- `{dataset} / {dist}`: Ours-FIM best/final={ours['best_metric']:.4f}/{ours['final_metric']:.4f}; "
            f"Greedy best/final={greedy['best_metric']:.4f}/{greedy['final_metric']:.4f}; "
            f"budget={ours['final_cumulative_budget']:.0f} vs {greedy['final_cumulative_budget']:.0f}."
        )
    FINAL_DECISION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    final_df = summarize_manifest(FINAL_MANIFEST)
    c5_df = summarize_manifest(C5_MANIFEST, multi_method=True)
    final_df.to_csv(FINAL_SUMMARY, index=False)
    c5_df.to_csv(C5_SUMMARY, index=False)
    plot_grouped_best(final_df)
    plot_round_curves(final_df, "final_lowbudget_stable_round_curves", "1260098 stable low-budget rerun curves")
    plot_round_curves(c5_df, "c5_clean_mckp_vs_greedy_curves", "1260123 clean C5: MCKP vs greedy")
    write_decision(final_df, c5_df)
    print(f"Wrote {FINAL_SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {C5_SUMMARY.relative_to(ROOT)}")
    print(f"Wrote figures to {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {FINAL_DECISION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
