#!/usr/bin/env python3
"""Register MNLI IID low-budget runs as final evidence and diagnose MNLI Dirichlet-1.0 selection."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "paper_assets" / "02_experiments" / "mnli_lowbudget_fim_summary.csv"
CURATED = ROOT / "paper_assets" / "02_experiments" / "curated_paper_data"
OVERRIDE_LEDGER = ROOT / "paper_assets" / "02_experiments" / "final_result_overrides.csv"
TRACE_DIR = ROOT / "paper_assets" / "02_experiments" / "traces" / "mnli_iid_dir1_lowbudget_fim"
DIAG_CSV = ROOT / "paper_assets" / "02_experiments" / "mnli_dirichlet_1p0_selection_diagnosis.csv"
DIAG_MD = ROOT / "paper_assets" / "02_experiments" / "mnli_dirichlet_1p0_selection_diagnosis.md"
FIG_DIR = ROOT / "paper_assets" / "05_figures" / "pilot_analysis"

METHOD_TO_TRACE = {
    "Ours-FIM": "selection_trace_Random_FIM.csv",
    "MS": "selection_trace_MS.csv",
}


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace(":", "_")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def register_iid_final() -> list[dict[str, str]]:
    df = pd.read_csv(SUMMARY)
    iid = df[(df["dataset"] == "mnli") & (df["distribution"] == "iid")].copy()
    rows: list[dict[str, str]] = []
    for _, row in iid.iterrows():
        method = str(row["method"])
        raw_path = ROOT / str(row["raw_result_path"])
        dst_dir = CURATED / "main" / "mnli" / "iid" / safe_name(method)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{safe_name(str(row['run_id']))}.csv"
        shutil.copy2(raw_path, dst)
        rows.append(
            {
                "placement": "main",
                "dataset": "mnli",
                "distribution": "iid",
                "method": method,
                "seed": str(row["seed"]),
                "metric": str(row["metric"]),
                "best_metric": f"{float(row['best_metric']):.4f}",
                "final_metric": f"{float(row['final_metric']):.4f}",
                "run_id": str(row["run_id"]),
                "config_path": str(row["config_path"]),
                "source_raw_path": str(row["raw_result_path"]),
                "curated_path": rel(dst),
                "source": "mnli_lowbudget_fim_final_override",
                "status": "complete",
                "notes": "User-selected final MNLI IID evidence from low-budget FIM-only rerun; original raw CSV unchanged.",
            }
        )
    OVERRIDE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with OVERRIDE_LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_trace(method: str) -> pd.DataFrame:
    path = TRACE_DIR / METHOD_TO_TRACE[method]
    df = pd.read_csv(path)
    df = df[(df["distribution"] == "dirichlet:1.0")].copy()
    df["block_id"] = df["module_name"].str.extract(r"layer\.(\d+)").astype(int)
    df["module_type"] = (
        df["module_name"]
        .str.extract(r"(key_proj|query_proj|value_proj)")[0]
        .map({"key_proj": "k", "query_proj": "q", "value_proj": "v"})
        .fillna("unknown")
    )
    df["method_paper"] = method
    return df


def summarize_trace(method: str, df: pd.DataFrame) -> dict[str, object]:
    per_block = df.groupby(["round", "client_id", "block_id"])["module_type"].nunique()
    per_client = df.groupby(["round", "client_id"]).agg(
        unique_blocks=("block_id", "nunique"),
        selected_matrices=("block_id", "size"),
        earliest_layer=("block_id", "min"),
        latest_layer=("block_id", "max"),
    )
    per_client["layer_span"] = per_client["latest_layer"] - per_client["earliest_layer"] + 1
    per_client["block_density"] = per_client["unique_blocks"] / per_client["layer_span"]
    depth_bins = pd.cut(df["depth_norm"], [0, 0.33, 0.66, 1.01], labels=["low", "mid", "high"])
    module_share = df["module_type"].value_counts(normalize=True)
    layer_counts = df["block_id"].value_counts().sort_index()
    return {
        "method": method,
        "run_id": df["run_id"].iloc[0],
        "rows": len(df),
        "rounds": df["round"].nunique(),
        "clients": df["client_id"].nunique(),
        "selected_per_round_mean": df.groupby("round").size().mean(),
        "depth_mean": df["depth_norm"].mean(),
        "depth_std": df["depth_norm"].std(),
        "layer_mean": df["block_id"].mean(),
        "layer_std": df["block_id"].std(),
        "min_layer": int(df["block_id"].min()),
        "max_layer": int(df["block_id"].max()),
        "depth_low_share": depth_bins.value_counts(normalize=True).get("low", 0.0),
        "depth_mid_share": depth_bins.value_counts(normalize=True).get("mid", 0.0),
        "depth_high_share": depth_bins.value_counts(normalize=True).get("high", 0.0),
        "q_share": module_share.get("q", 0.0),
        "k_share": module_share.get("k", 0.0),
        "v_share": module_share.get("v", 0.0),
        "avg_modules_per_selected_block": per_block.mean(),
        "complete_qkv_block_fraction": (per_block == 3).mean(),
        "single_matrix_block_fraction": (per_block == 1).mean(),
        "unique_blocks_per_client_mean": per_client["unique_blocks"].mean(),
        "layer_span_per_client_mean": per_client["layer_span"].mean(),
        "block_density_mean": per_client["block_density"].mean(),
        "top_layers": " ".join(f"{k}:{v}" for k, v in layer_counts.tail(8).items()),
    }


def write_diagnosis(rows: list[dict[str, object]]) -> None:
    DIAG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DIAG_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_method = {row["method"]: row for row in rows}
    ours = by_method["Ours-FIM"]
    ms = by_method["MS"]
    DIAG_MD.write_text(
        "\n".join(
            [
                "# MNLI Dirichlet-1.0 Selection Diagnosis",
                "",
                "Facts from trace CSVs:",
                "",
                f"- Ours-FIM trace: `{rel(TRACE_DIR / METHOD_TO_TRACE['Ours-FIM'])}`.",
                f"- MS trace: `{rel(TRACE_DIR / METHOD_TO_TRACE['MS'])}`.",
                f"- Both traces contain `distribution=dirichlet:1.0` rows for the low-budget MNLI rerun.",
                f"- Diagnosis table: `{rel(DIAG_CSV)}`.",
                "",
                "Key observations:",
                "",
                f"- Ours-FIM selected depth mean is {ours['depth_mean']:.3f}, with selected layers spanning {ours['min_layer']}-{ours['max_layer']}.",
                f"- MS selected depth mean is {ms['depth_mean']:.3f}, with selected layers spanning {ms['min_layer']}-{ms['max_layer']}.",
                f"- Ours-FIM complete q/k/v block fraction is {ours['complete_qkv_block_fraction']:.3f}; MS is {ms['complete_qkv_block_fraction']:.3f}.",
                f"- Ours-FIM single-matrix block fraction is {ours['single_matrix_block_fraction']:.3f}; MS is {ms['single_matrix_block_fraction']:.3f}.",
                f"- Ours-FIM module shares are q={ours['q_share']:.3f}, k={ours['k_share']:.3f}, v={ours['v_share']:.3f}; MS is exactly balanced at q={ms['q_share']:.3f}, k={ms['k_share']:.3f}, v={ms['v_share']:.3f}.",
                "",
                "Interpretation:",
                "",
                "The failure mode is not only insufficient depth preference. Ours-FIM already selects mostly high-depth matrices, but it is fragmented across more layers and often selects only one matrix inside a q/k/v block. MS is both deeper and block-complete, repeatedly selecting full q/k/v groups in the last layers.",
                "",
                "Decision:",
                "",
                "Do not keep lowering MNLI Dirichlet-1.0 budget as the first fix. The evidence points to adding a task-conditional block-coherence bonus or q/k/v completeness prior for MNLI Dirichlet-1.0, optionally with a mild extra depth prior. This preserves matrix-level granularity while preventing isolated matrix picks from losing to MS.",
                "",
                "TODO:",
                "",
                "- [TODO-RESULT] Rerun MNLI Dirichlet-1.0 with Ours-FIM + block coherence and unique trace_dir per distribution.",
                "- [TODO-CODE-CHECK] Confirm whether block-coherence flags are fully wired into current DP transitions before launching the rerun.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def plot_layers(traces: dict[str, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (method, df) in zip(axes, traces.items()):
        counts = df["block_id"].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color="#4C78A8" if method == "Ours-FIM" else "#F58518")
        ax.set_title(method)
        ax.set_xlabel("DeBERTa layer")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Selected matrices")
    fig.suptitle("MNLI Dirichlet-1.0 low-budget selected layers")
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mnli_dir1_selection_layer_hist.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)

    metrics = pd.DataFrame(
        [
            {
                "method": method,
                "depth_mean": df["depth_norm"].mean(),
                "complete_qkv": df.groupby(["round", "client_id", "block_id"])["module_type"].nunique().eq(3).mean(),
                "single_matrix": df.groupby(["round", "client_id", "block_id"])["module_type"].nunique().eq(1).mean(),
            }
            for method, df in traces.items()
        ]
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = range(len(metrics))
    width = 0.25
    ax.bar([i - width for i in x], metrics["depth_mean"], width, label="Depth mean")
    ax.bar(x, metrics["complete_qkv"], width, label="Complete q/k/v")
    ax.bar([i + width for i in x], metrics["single_matrix"], width, label="Single-matrix block")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics["method"])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.suptitle("MNLI Dirichlet-1.0 selection structure")
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mnli_dir1_selection_structure.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> int:
    registered = register_iid_final()
    traces = {method: load_trace(method) for method in METHOD_TO_TRACE}
    diagnosis_rows = [summarize_trace(method, df) for method, df in traces.items()]
    write_diagnosis(diagnosis_rows)
    plot_layers(traces)
    print(f"Registered {len(registered)} MNLI IID final override rows at {rel(OVERRIDE_LEDGER)}")
    print(f"Wrote diagnosis table to {rel(DIAG_CSV)}")
    print(f"Wrote diagnosis note to {rel(DIAG_MD)}")
    print(f"Wrote figures to {rel(FIG_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
