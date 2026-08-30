#!/usr/bin/env python3
"""Build paper-oriented figures and tables from aligned experiment evidence."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "paper_assets/02_experiments/evidence_ledger.csv"
MAIN_TABLE = ROOT / "paper_assets/02_experiments/extended_main_results_table.csv"
ABLATION_TABLE = ROOT / "paper_assets/02_experiments/ablation_results_table.csv"
FIG_DIR = ROOT / "paper_assets/05_figures/final"
TABLE_DIR = ROOT / "paper_assets/06_tables/final"
MANIFEST = ROOT / "paper_assets/05_figures/final_figure_manifest.md"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

METHOD_ORDER = [
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
    "Fed-HeLLo",
    "LEGEND",
    "MS",
    "Ours-FIM",
    "Ours-Taylor",
]
METHOD_COLORS = {
    "FedAvg-LoRA homogeneous": "#8c8c8c",
    "Uniform allocation": "#4c78a8",
    "Fed-HeLLo": "#72b7b2",
    "LEGEND": "#54a24b",
    "MS": "#b279a2",
    "Ours-FIM": "#f58518",
    "Ours-Taylor": "#e45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final paper figures.")
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def figure_grid(n_items: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, n_items))
    rows = math.ceil(n_items / cols)
    return rows, cols


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({col: row.get(col, "") for col in columns} for row in rows)


def f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float) -> str:
    return f"{value:.4f}"


def method_rank(name: str) -> int:
    return METHOD_ORDER.index(name) if name in METHOD_ORDER else len(METHOD_ORDER)


def short_method(name: str) -> str:
    return {
        "FedAvg-LoRA homogeneous": "FedAvg",
        "Uniform allocation": "Uniform",
        "Physical-layer FIM": "Physical",
        "random/no-importance scores": "Random score",
        "Ours-FIM w/o depth prior": "No depth",
        "Ours-FIM additive-memory DP": "Additive mem",
        "Ours-FIM lambda_prime=0": "No latency",
        "Ours-FIM MCKP": "Ours-FIM",
        "greedy value/cost scheduler": "Greedy",
    }.get(name, name)


def metric_col_for(row: dict) -> str:
    return "Rouge_l" if row["dataset"] == "samsum" else "Accuracy"


def load_result(row: dict) -> list[dict]:
    path = ROOT / row["raw_result_path"]
    return read_csv(path) if path.exists() else []


def group_key(row: dict) -> str:
    return f"{row['dataset']} / {row['distribution']}"


def plot_main_result_matrix(rows: list[dict], out_dir: Path, fmt_ext: str) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    groups = sorted({group_key(row) for row in rows})
    methods = [m for m in METHOD_ORDER if any(row["method"] == m for row in rows)]
    values = np.full((len(groups), len(methods)), np.nan)
    labels = [["" for _ in methods] for _ in groups]
    by_pair = {(group_key(row), row["method"]): f(row["best_metric"]) for row in rows}
    for i, group in enumerate(groups):
        for j, method in enumerate(methods):
            if (group, method) in by_pair:
                values[i, j] = by_pair[(group, method)]
                labels[i][j] = fmt(values[i, j])

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1.0, np.nanmax(values)))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([short_method(m) for m in methods], rotation=35, ha="right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_title("Main-grid best performance")
    for i in range(len(groups)):
        for j in range(len(methods)):
            if labels[i][j]:
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Best Accuracy / Rouge-L")
    fig.tight_layout()
    path = out_dir / f"fig1_main_result_matrix.{fmt_ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grouped_best(rows: list[dict], out_dir: Path, fmt_ext: str) -> Path:
    import matplotlib.pyplot as plt

    groups = sorted(defaultdict(list, {}).keys())
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    groups = sorted(grouped)

    n_rows, n_cols = figure_grid(len(groups))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.2 * n_rows), squeeze=False)
    for ax, group in zip(axes.ravel(), groups):
        data = sorted(grouped[group], key=lambda row: method_rank(row["method"]))
        labels = [short_method(row["method"]) for row in data]
        vals = [f(row["best_metric"]) for row in data]
        colors = [METHOD_COLORS.get(row["method"], "#999999") for row in data]
        ax.bar(range(len(vals)), vals, color=colors, width=0.72)
        ax.set_title(group, fontsize=10)
        ax.set_ylim(0, max(1.0, max(vals) * 1.12 if vals else 1.0))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.suptitle("Best performance by dataset, heterogeneity, and method", y=1.01, fontsize=13)
    fig.tight_layout()
    path = out_dir / f"fig2_main_grouped_best.{fmt_ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_curves(rows: list[dict], out_dir: Path, fmt_ext: str, x_col: str, suffix: str, title: str) -> Path:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    groups = sorted(grouped)
    n_rows, n_cols = figure_grid(len(groups))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax, group in zip(axes.ravel(), groups):
        for row in sorted(grouped[group], key=lambda r: method_rank(r["method"])):
            result_rows = load_result(row)
            metric = metric_col_for(row)
            xs = [f(r.get(x_col)) for r in result_rows]
            ys = [f(r.get(metric)) for r in result_rows]
            if xs and ys:
                ax.plot(xs, ys, linewidth=1.6, color=METHOD_COLORS.get(row["method"], "#555555"), label=short_method(row["method"]))
        ax.set_title(f"{group} ({metric_col_for(grouped[group][0])})", fontsize=10)
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, linewidth=0.6)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.06, fontsize=13)
    fig.tight_layout()
    path = out_dir / f"app_full_{suffix}_curves.{fmt_ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_representative_curves(rows: list[dict], out_dir: Path, fmt_ext: str, x_col: str, suffix: str) -> Path:
    import matplotlib.pyplot as plt

    preferred = [
        ("qqp", "dirichlet:0.1"),
        ("mnli", "iid"),
        ("samsum", "length_skew:mild"),
        ("samsum", "length_skew:severe"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), squeeze=False)
    for ax, (dataset, distribution) in zip(axes.ravel(), preferred):
        subset = [
            row for row in rows
            if row["dataset"] == dataset and row["distribution"] == distribution
        ]
        for row in sorted(subset, key=lambda r: method_rank(r["method"])):
            result_rows = load_result(row)
            metric = metric_col_for(row)
            xs = [f(r.get(x_col)) for r in result_rows]
            ys = [f(r.get(metric)) for r in result_rows]
            if xs and ys:
                ax.plot(xs, ys, linewidth=1.9, color=METHOD_COLORS.get(row["method"], "#555555"), label=short_method(row["method"]))
        ax.set_title(f"{dataset} / {distribution}")
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel("Accuracy" if dataset != "samsum" else "Rouge-L")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, linewidth=0.6)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    path = out_dir / f"fig3_representative_{suffix}_curves.{fmt_ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation_delta(rows: list[dict], out_dir: Path, fmt_ext: str) -> Path:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["manifest_row"]].append(row)

    labels, best_delta, final_delta = [], [], []
    for manifest_row in sorted(grouped, key=lambda value: int(value)):
        data = grouped[manifest_row]
        if len(data) == 1:
            row = data[0]
            labels.append(f"C{manifest_row}\n{short_method(row['method'])}")
            best_delta.append(0.0)
            final_delta.append(0.0)
            continue
        base = next((row for row in data if "Ours-FIM" in row["method"] and "w/o" not in row["method"] and "lambda" not in row["method"] and "additive" not in row["method"]), data[0])
        for row in data:
            if row is base:
                continue
            labels.append(f"C{manifest_row}\n{short_method(row['method'])}")
            best_delta.append(f(base["best_metric"]) - f(row["best_metric"]))
            final_delta.append(f(base["final_metric"]) - f(row["final_metric"]))

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar([i - 0.18 for i in x], best_delta, width=0.36, label="Best metric delta", color="#f58518")
    ax.bar([i + 0.18 for i in x], final_delta, width=0.36, label="Final metric delta", color="#4c78a8")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Full method minus ablation")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"fig4_ablation_delta.{fmt_ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def trace_files(evidence_rows: list[dict], kind: str) -> list[Path]:
    paths = []
    seen = set()
    for row in evidence_rows:
        run_dir = ROOT / "evaluation_results" / row["dataset"] / row["run_id"] / "traces"
        method = row["method_raw"]
        path = run_dir / f"{kind}_trace_{method}.csv"
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def build_selection_heatmap(evidence_rows: list[dict], out_dir: Path, fmt_ext: str) -> tuple[Path | None, Path | None]:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for path in trace_files(evidence_rows, "selection"):
        data = read_csv(path)
        if not data:
            continue
        first = data[0]
        if first["method"] not in {"Random_FIM", "Random_Taylor"}:
            continue
        label = f"{first['dataset']}/{first['distribution']}/{first['method']}"
        counts: dict[int, int] = defaultdict(int)
        for row in data:
            counts[int(float(row["matrix_index"]))] += 1
        rows.append((label, counts))
    if not rows:
        return None, None

    max_index = max(max(counts) for _, counts in rows)
    matrix = np.zeros((len(rows), max_index + 1))
    for i, (_, counts) in enumerate(rows):
        total = max(1, sum(counts.values()))
        for idx, count in counts.items():
            matrix[i, idx] = count / total

    fig, ax = plt.subplots(figsize=(13, max(4, len(rows) * 0.42)))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=8)
    ax.set_xlabel("LoRA candidate matrix index")
    ax.set_title("Selection frequency over matrix-level LoRA candidates")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative selection frequency")
    fig.tight_layout()
    figure_path = out_dir / f"fig5_selection_heatmap.{fmt_ext}"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    table_rows = []
    for label, counts in rows:
        total = max(1, sum(counts.values()))
        top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
        table_rows.append({
            "trace": label,
            "total_selected_matrix_records": total,
            "top_matrix_indices": " ".join(str(idx) for idx, _ in top),
            "top_matrix_frequencies": " ".join(fmt(count / total) for _, count in top),
        })
    table_path = TABLE_DIR / "selection_heatmap_top_matrices.csv"
    write_csv(table_path, table_rows, ["trace", "total_selected_matrix_records", "top_matrix_indices", "top_matrix_frequencies"])
    return figure_path, table_path


def build_feasibility_table(evidence_rows: list[dict], out_dir: Path, fmt_ext: str) -> tuple[Path | None, Path | None]:
    import matplotlib.pyplot as plt

    aggregate: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in trace_files(evidence_rows, "memory"):
        data = read_csv(path)
        for row in data:
            aggregate[(row["dataset"], row["distribution"], row["method"])].append(row)
    if not aggregate:
        return None, None

    rows = []
    for key, data in sorted(aggregate.items()):
        generated = [f(row["generated_configs"]) for row in data]
        retained = [f(row["pruned_configs"]) for row in data]
        rejects = [f(row["memory_rejects"]) for row in data]
        seen = [f(row["configs_seen_by_server"]) for row in data]
        rows.append({
            "dataset": key[0],
            "distribution": key[1],
            "method": key[2],
            "records": len(data),
            "avg_generated_configs": fmt(mean(generated)),
            "avg_feasible_configs": fmt(mean(retained)),
            "avg_memory_rejects": fmt(mean(rejects)),
            "avg_configs_seen_by_server": fmt(mean(seen)),
            "feasible_rate": fmt(mean(retained) / mean(generated) if mean(generated) else 0.0),
        })
    table_path = TABLE_DIR / "feasibility_summary.csv"
    columns = [
        "dataset", "distribution", "method", "records", "avg_generated_configs",
        "avg_feasible_configs", "avg_memory_rejects", "avg_configs_seen_by_server",
        "feasible_rate",
    ]
    write_csv(table_path, rows, columns)

    labels = [f"{r['dataset']}/{r['distribution']}/{short_method(r['method'])}" for r in rows]
    rates = [f(r["feasible_rate"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(rows) * 0.32)))
    ax.barh(range(len(rows)), rates, color="#4c78a8")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Feasible configuration rate")
    ax.set_xlim(0, max(1.0, max(rates) * 1.1 if rates else 1.0))
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig_path = out_dir / f"fig6_feasibility_rates.{fmt_ext}"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path, table_path


def build_overhead_figure(evidence_rows: list[dict], out_dir: Path, fmt_ext: str) -> Path | None:
    import matplotlib.pyplot as plt

    rows = []
    for row in evidence_rows:
        path_text = row.get("overhead_result_path", "")
        if not path_text or path_text.startswith("[TODO"):
            continue
        path = ROOT / path_text
        if not path.exists():
            continue
        data = read_csv(path)
        if not data:
            continue
        stages = [
            "profiling_time_sec",
            "config_generation_time_sec",
            "server_scheduling_time_sec",
            "client_training_time_sec",
            "aggregation_time_sec",
            "evaluation_time_sec",
        ]
        rows.append({
            "label": f"{row['dataset']}/{row['distribution']}/{short_method(row['method_paper'])}",
            **{stage: mean([f(r.get(stage)) for r in data]) for stage in stages},
        })
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["label"])[:24]
    stages = [
        "profiling_time_sec",
        "config_generation_time_sec",
        "server_scheduling_time_sec",
        "client_training_time_sec",
        "aggregation_time_sec",
        "evaluation_time_sec",
    ]
    labels = [r["label"] for r in rows]
    left = [0.0] * len(rows)
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2"]
    fig, ax = plt.subplots(figsize=(12, max(5, len(rows) * 0.32)))
    for stage, color in zip(stages, colors):
        values = [r[stage] for r in rows]
        ax.barh(range(len(rows)), values, left=left, label=stage.replace("_time_sec", ""), color=color)
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Average stage time per round (s)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    path = out_dir / f"app_overhead_breakdown.{fmt_ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_tables(main_rows: list[dict], ablation_rows: list[dict]) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    main_cols = [
        "dataset", "distribution", "method", "metric", "best_metric", "final_metric",
        "final_cumulative_time", "final_cumulative_budget", "run_id", "raw_result_path",
    ]
    ablation_cols = [
        "manifest_row", "dataset", "distribution", "method", "metric", "best_metric",
        "final_metric", "run_id", "raw_result_path",
    ]
    main_path = TABLE_DIR / "table1_main_results.csv"
    ablation_path = TABLE_DIR / "table2_ablation_results.csv"
    write_csv(main_path, main_rows, main_cols)
    write_csv(ablation_path, ablation_rows, ablation_cols)
    return [main_path, ablation_path]


def write_manifest(paths: list[tuple[str, Path | None, str]], table_paths: list[Path]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as handle:
        handle.write("# Final Figure Manifest\n\n")
        handle.write("Generated from `paper_assets/02_experiments/extended_main_results_table.csv`, `paper_assets/02_experiments/evidence_ledger.csv`, and raw CSVs under `evaluation_results/`.\n\n")
        handle.write("| Asset | Path | Meaning |\n")
        handle.write("|---|---|---|\n")
        for name, path, meaning in paths:
            path_text = rel(path) if path else "[TODO-RESULT]"
            handle.write(f"| {name} | `{path_text}` | {meaning} |\n")
        for path in table_paths:
            handle.write(f"| Table | `{rel(path)}` | Derived CSV table for paper reporting. |\n")


def main() -> int:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = read_csv(MAIN_TABLE)
    main_rows = [row for row in main_rows if row.get("metric") != "Loss"]
    ablation_rows = read_csv(ABLATION_TABLE)
    evidence_rows = read_csv(EVIDENCE)

    figure_paths: list[tuple[str, Path | None, str]] = []
    figure_paths.append(("Fig. 1 Main result matrix", plot_main_result_matrix(main_rows, FIG_DIR, args.format), "Compact heatmap of best Accuracy/Rouge-L for each dataset, distribution, and method."))
    figure_paths.append(("Fig. 2 Main grouped bars", plot_grouped_best(main_rows, FIG_DIR, args.format), "Grouped best-performance bars for all main-grid conditions."))
    figure_paths.append(("Fig. 3a Budget curves", plot_representative_curves(main_rows, FIG_DIR, args.format, "Cumulative_Budget", "budget"), "Representative performance-versus-budget curves."))
    figure_paths.append(("Fig. 3b Time curves", plot_representative_curves(main_rows, FIG_DIR, args.format, "Cumulative_Time", "time"), "Representative performance-versus-estimated-time curves."))
    figure_paths.append(("Fig. 4 Ablation deltas", plot_ablation_delta(ablation_rows, FIG_DIR, args.format), "Metric drop from replacing/removing one component in representative ablations."))
    selection_fig, selection_table = build_selection_heatmap(evidence_rows, FIG_DIR, args.format)
    figure_paths.append(("Fig. 5 Selection heatmap", selection_fig, "Matrix-level LoRA candidate selection frequency from trace outputs."))
    feasibility_fig, feasibility_table = build_feasibility_table(evidence_rows, FIG_DIR, args.format)
    figure_paths.append(("Fig. 6 Feasibility rates", feasibility_fig, "Configuration feasibility under memory constraints from memory traces."))
    figure_paths.append(("Appendix A1 Round curves", plot_curves(main_rows, FIG_DIR, args.format, "Round", "round", "Full performance-vs-round curves"), "All main-grid per-round performance curves."))
    figure_paths.append(("Appendix A2 Budget curves", plot_curves(main_rows, FIG_DIR, args.format, "Cumulative_Budget", "budget", "Full performance-vs-budget curves"), "All main-grid performance-vs-budget curves."))
    figure_paths.append(("Appendix A3 Time curves", plot_curves(main_rows, FIG_DIR, args.format, "Cumulative_Time", "time", "Full performance-vs-time curves"), "All main-grid performance-vs-estimated-time curves."))
    figure_paths.append(("Appendix A4 Overhead", build_overhead_figure(evidence_rows, FIG_DIR, args.format), "Average instrumentation overhead breakdown when overhead CSVs are available."))

    table_paths = copy_tables(main_rows, ablation_rows)
    if selection_table:
        table_paths.append(selection_table)
    if feasibility_table:
        table_paths.append(feasibility_table)
    write_manifest(figure_paths, table_paths)
    for _, path, _ in figure_paths:
        if path:
            print(rel(path))
    for path in table_paths:
        print(rel(path))
    print(rel(MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
