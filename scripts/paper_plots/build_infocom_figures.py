#!/usr/bin/env python3
"""Build compact INFOCOM figures from the accepted result ledger.

Raw evaluation and overhead CSVs are read-only inputs. Outputs are written only
under the INFOCOM manuscript directory.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper_assets/06_tables/final/table1_main_results.csv"
ABLATIONS = ROOT / "paper_assets/06_tables/final/table2_ablation_pairs.csv"
OUT = ROOT / "paper_assets/04_manuscript/infocom_2027/figures"
TABLE_OUT = ROOT / "paper_assets/04_manuscript/infocom_2027/tables"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

METHODS = [
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
    "Fed-HeLLo",
    "HETLoRA-style",
    "FAH-QLoRA-style",
    "LEGEND",
    "MS",
    "Ours-FIM",
]
SHORT = {
    "FedAvg-LoRA homogeneous": "FedAvg",
    "Uniform allocation": "Uniform",
    "Fed-HeLLo": "Fed-HeLLo",
    "HETLoRA-style": "HET-style",
    "FAH-QLoRA-style": "FAH-style",
    "LEGEND": "LEGEND",
    "MS": "MS",
    "Ours-FIM": "FedMATRIX",
}
COLORS = {
    "FedAvg-LoRA homogeneous": "#777777",
    "Uniform allocation": "#4c78a8",
    "Fed-HeLLo": "#72b7b2",
    "HETLoRA-style": "#9d755d",
    "FAH-QLoRA-style": "#e3b51a",
    "LEGEND": "#54a24b",
    "MS": "#b279a2",
    "Ours-FIM": "#e66101",
}
LINESTYLES = {
    "FedAvg-LoRA homogeneous": "-",
    "Uniform allocation": "--",
    "Fed-HeLLo": ":",
    "HETLoRA-style": "-.",
    "FAH-QLoRA-style": (0, (3, 1, 1, 1)),
    "LEGEND": (0, (5, 1)),
    "MS": (0, (1, 1)),
    "Ours-FIM": "-",
}
MARKERS = {
    "FedAvg-LoRA homogeneous": "o",
    "Uniform allocation": "s",
    "Fed-HeLLo": "^",
    "HETLoRA-style": "v",
    "FAH-QLoRA-style": "D",
    "LEGEND": "P",
    "MS": "X",
    "Ours-FIM": "o",
}
GROUPS = [
    ("cifar100", "dirichlet:0.1", "CIFAR/Non-IID 0.1"),
    ("cifar100", "dirichlet:1.0", "CIFAR/Non-IID 1.0"),
    ("cifar100", "iid", "CIFAR/IID"),
    ("mnli", "dirichlet:1.0", "MNLI/Non-IID 1.0"),
    ("mnli", "iid", "MNLI/IID"),
    ("qnli", "dirichlet:1.0", "QNLI/Non-IID 1.0"),
    ("qnli", "iid", "QNLI/IID"),
    ("qqp", "dirichlet:1.0", "QQP/Non-IID 1.0"),
    ("qqp", "iid", "QQP/IID"),
    ("sst2", "dirichlet:1.0", "SST-2/Non-IID 1.0"),
    ("sst2", "iid", "SST-2/IID"),
    ("samsum", "iid", "SAMSum/IID"),
    ("samsum", "length_skew:mild", "SAMSum/Len-M"),
    ("samsum", "length_skew:severe", "SAMSum/Len-S"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_matrix(rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    lookup = {(r["dataset"], r["distribution"], r["method"]): number(r["best_metric"]) for r in rows}
    values = np.full((len(GROUPS), len(METHODS)), np.nan)
    for i, (dataset, distribution, _) in enumerate(GROUPS):
        for j, method in enumerate(METHODS):
            values[i, j] = lookup.get((dataset, distribution, method), np.nan)

    fig, ax = plt.subplots(figsize=(6.20, 3.25))
    norm = Normalize(vmin=0.25, vmax=0.95)
    image = ax.pcolormesh(
        np.arange(len(METHODS) + 1) - 0.5,
        np.arange(len(GROUPS) + 1) - 0.5,
        np.ma.masked_invalid(values),
        cmap="YlGnBu",
        norm=norm,
        shading="flat",
    )
    ax.set_xlim(-0.5, len(METHODS) - 0.5)
    ax.set_ylim(len(GROUPS) - 0.5, -0.5)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([SHORT[m] for m in METHODS], rotation=28, ha="right", fontsize=8.4)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([g[2] for g in GROUPS], fontsize=8.2)
    ax.tick_params(length=2, pad=1.5)
    for i in range(values.shape[0]):
        best = np.nanmax(values[i])
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                continue
            is_best = np.isclose(value, best)
            color = "white" if value >= 0.68 else "#17202a"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    fontsize=7.4, color=color, fontweight="bold" if is_best else "normal")
            if is_best:
                ax.add_patch(plt.Rectangle((j - .47, i - .43), .94, .86, fill=False,
                                           edgecolor="#d94801", linewidth=1.0))
    cbar = fig.colorbar(image, ax=ax, fraction=.018, pad=.012)
    cbar.solids.set_rasterized(False)
    cbar.ax.tick_params(labelsize=7.6, length=2)
    cbar.set_label("Best Accuracy / Rouge-L", fontsize=8.0)
    fig.tight_layout(pad=.25)
    save(fig, "fig2_compact_main_matrix")
    plt.close(fig)


def build_resource_curves(rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    classification_groups = GROUPS[:11]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["distribution"])].append(row)

    axis_specs = [
        ("Cumulative_Budget", "Budget", "fig4_budget_curves"),
        ("Cumulative_Time", "Time (s)", "fig5_time_curves"),
    ]
    for xcol, xlabel, stem in axis_specs:
        fig, axes = plt.subplots(2, 6, figsize=(8.20, 2.02), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)

        handles = {}
        for index, (dataset, distribution, title) in enumerate(classification_groups):
            ax = axes[index // 6][index % 6]
            ax.set_visible(True)
            all_y = []
            setting_rows = sorted(
                grouped[(dataset, distribution)], key=lambda r: METHODS.index(r["method"])
            )
            for row in setting_rows:
                path = ROOT / row["raw_result_path"]
                if not path.exists():
                    continue
                trace = read_csv(path)
                xs = [number(item.get(xcol)) for item in trace]
                ys = [number(item.get("Accuracy")) for item in trace]
                if not xs:
                    continue
                is_ours = row["method"] == "Ours-FIM"
                line, = ax.plot(
                    xs,
                    ys,
                    color=COLORS[row["method"]],
                    linestyle=LINESTYLES[row["method"]],
                    marker=MARKERS[row["method"]],
                    markevery=0.28,
                    markersize=2.1 if is_ours else 1.6,
                    linewidth=1.45 if is_ours else .62,
                    alpha=1 if is_ours else .76,
                    zorder=5 if is_ours else 2,
                    label=SHORT[row["method"]],
                )
                handles[row["method"]] = line
                all_y.extend(ys)
            if all_y:
                ymin, ymax = min(all_y), max(all_y)
                pad = max(.018, .08 * (ymax - ymin))
                ax.set_ylim(max(0, ymin - pad), min(1, ymax + pad))
            ax.set_title(title, fontsize=6.4, pad=1.0)
            ax.set_xlabel(xlabel, fontsize=5.8, labelpad=.8)
            ax.set_ylabel("Accuracy", fontsize=5.8, labelpad=.8)
            ax.tick_params(labelsize=5.1, length=1.2, pad=.8)
            ax.grid(alpha=.20, linewidth=.28)

        ordered = [handles[m] for m in METHODS if m in handles]
        labels = [SHORT[m] for m in METHODS if m in handles]
        legend_ax = axes[1][5]
        legend_ax.set_visible(True)
        legend_ax.axis("off")
        legend = legend_ax.legend(
            ordered,
            labels,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=6.3,
            columnspacing=.55,
            handlelength=1.25,
            handletextpad=.30,
        )
        for line in legend.get_lines():
            line.set_linewidth(1.5)
        fig.subplots_adjust(
            left=.055,
            right=.995,
            bottom=.11,
            top=.965,
            wspace=.62,
            hspace=.72,
        )
        save(fig, stem)
        plt.close(fig)


def build_combined_resource_curves(rows: list[dict[str, str]]) -> None:
    """Build the paper's compact budget/time panel with one shared legend."""
    import matplotlib.pyplot as plt

    classification_groups = GROUPS[:11]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["distribution"])].append(row)

    fig = plt.figure(figsize=(7.15, 3.75))
    grid = fig.add_gridspec(
        5,
        6,
        height_ratios=[1, 1, .14, 1, 1],
        left=.055,
        right=.995,
        bottom=.075,
        top=.965,
        wspace=.34,
        hspace=.42,
    )

    handles = {}
    axis_specs = [
        ("Cumulative_Budget", "Cumulative matrix budget", 0),
        ("Cumulative_Time", "Modeled elapsed time (s)", 3),
    ]
    for xcol, _, row_offset in axis_specs:
        for index, (dataset, distribution, title) in enumerate(classification_groups):
            ax = fig.add_subplot(grid[row_offset + index // 6, index % 6])
            all_y = []
            setting_rows = sorted(
                grouped[(dataset, distribution)], key=lambda r: METHODS.index(r["method"])
            )
            for row in setting_rows:
                path = ROOT / row["raw_result_path"]
                if not path.exists():
                    continue
                trace = read_csv(path)
                xs = [number(item.get(xcol)) for item in trace]
                ys = [number(item.get("Accuracy")) for item in trace]
                if not xs:
                    continue
                is_ours = row["method"] == "Ours-FIM"
                line, = ax.plot(
                    xs,
                    ys,
                    color=COLORS[row["method"]],
                    linewidth=1.45 if is_ours else .62,
                    alpha=1 if is_ours else .76,
                    zorder=5 if is_ours else 2,
                    label=SHORT[row["method"]],
                )
                handles[row["method"]] = line
                all_y.extend(ys)
            if all_y:
                ymin, ymax = min(all_y), max(all_y)
                pad = max(.018, .08 * (ymax - ymin))
                ax.set_ylim(max(0, ymin - pad), min(1, ymax + pad))
            if index % 6 == 0:
                ax.set_ylabel("Accuracy", fontsize=4.8, labelpad=.8)
            ax.set_title(title, fontsize=5.1, pad=1.0)
            ax.tick_params(labelsize=3.9, length=1.2, pad=.8)
            ax.grid(alpha=.20, linewidth=.28)

    budget_label_ax = fig.add_subplot(grid[2, :])
    budget_label_ax.axis("off")
    budget_label_ax.text(
        .5,
        .5,
        "Cumulative matrix budget",
        ha="center",
        va="center",
        fontsize=5.7,
    )
    fig.text(.50, .012, "Modeled elapsed time (s)", ha="center", va="bottom", fontsize=5.7)

    legend_ax = fig.add_subplot(grid[4, 5])
    legend_ax.axis("off")
    ordered = [handles[m] for m in METHODS if m in handles]
    labels = [SHORT[m] for m in METHODS if m in handles]
    legend = legend_ax.legend(
        ordered,
        labels,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=6.3,
        columnspacing=.55,
        handlelength=1.25,
        handletextpad=.30,
    )
    for line in legend.get_lines():
        line.set_linewidth(1.5)
    save(fig, "fig4_compact_resource_curves")
    plt.close(fig)


def overhead_path(row: dict[str, str]) -> Path:
    raw = ROOT / row["raw_result_path"]
    return raw.with_name(raw.name.replace("eval_", "overhead_"))


def build_ablation_overhead(rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ablations = read_csv(ABLATIONS)
    metric_rows = [r for r in ablations if r["component"] in {"C1", "C2", "C3", "C4"}]
    labels = [r["component"] for r in metric_rows]
    deltas = [number(r["metric_delta"]) for r in metric_rows]

    overhead_stage_columns = [
        ("profiling_time_sec", "Profile", "#4c78a8"),
        ("config_generation_time_sec", "Config", "#f58518"),
        ("server_scheduling_time_sec", "Schedule", "#e45756"),
        ("client_training_time_sec", "Train", "#72b7b2"),
    ]
    wanted = [("cifar100", "iid", "CIFAR"), ("mnli", "iid", "MNLI"), ("qnli", "iid", "QNLI")]
    mean_stage_times = []
    stage_time_labels = []
    overhead_rows = []
    for dataset, distribution, label in wanted:
        candidates = [r for r in rows if r["dataset"] == dataset and r["distribution"] == distribution and r["method"] == "Ours-FIM"]
        if not candidates:
            continue
        path = overhead_path(candidates[0])
        if not path.exists():
            continue
        trace = read_csv(path)
        means = [sum(number(r.get(col)) for r in trace) / len(trace) for col, _, _ in overhead_stage_columns]
        mean_stage_times.append(means)
        stage_time_labels.append(label)
        scheduling_total = sum(means[:3])
        overhead_rows.append({
            "dataset": dataset,
            "distribution": distribution,
            "method": candidates[0]["method"],
            "seed": candidates[0]["seed"],
            "metric": candidates[0]["metric"],
            "run_id": candidates[0]["run_id"],
            "config_path": candidates[0]["config_path"],
            "raw_result_path": candidates[0]["raw_result_path"],
            "overhead_result_path": str(path.relative_to(ROOT)),
            "rounds": len(trace),
            "profile_mean_sec": f"{means[0]:.6f}",
            "config_mean_sec": f"{means[1]:.6f}",
            "schedule_mean_sec": f"{means[2]:.6f}",
            "training_mean_sec": f"{means[3]:.6f}",
            "scheduling_pipeline_mean_sec": f"{scheduling_total:.6f}",
            "training_over_pipeline_ratio": f"{means[3] / scheduling_total:.6f}",
        })

    if overhead_rows:
        write_csv(
            TABLE_OUT / "overhead_per_round.csv",
            overhead_rows,
            list(overhead_rows[0]),
        )

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.05))
    bars = axes[0].bar(labels, deltas, color="#e66101", width=.58)
    axes[0].set_ylabel("Metric loss (full - ablated)", fontsize=7.2)
    axes[0].set_title("(a) Component ablations", fontsize=8.2)
    axes[0].set_ylim(0, max(deltas) * 1.28)
    axes[0].tick_params(labelsize=7.5, length=2)
    axes[0].grid(axis="y", alpha=.22, linewidth=.4)
    for bar, value in zip(bars, deltas):
        axes[0].text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=6.7)

    if mean_stage_times:
        x = np.arange(len(mean_stage_times))
        width = .18
        for idx, (_, stage, color) in enumerate(overhead_stage_columns):
            vals = [row[idx] for row in mean_stage_times]
            bars = axes[1].bar(x + (idx - 1.5) * width, vals, width=width,
                               color=color, label=stage)
            for bar, value in zip(bars, vals):
                axes[1].text(bar.get_x() + bar.get_width() / 2, value,
                             f"{value:.1f}", ha="center", va="bottom",
                             fontsize=5.2, rotation=90 if value < 12 else 0)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(stage_time_labels, fontsize=7.2)
        axes[1].set_ylabel("Mean time per round (s)", fontsize=7.2)
        axes[1].set_ylim(0, max(max(row) for row in mean_stage_times) * 1.16)
        axes[1].tick_params(axis="y", labelsize=7, length=2)
        axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=4,
                       frameon=False, fontsize=5.8, columnspacing=.65, handlelength=.9)
    axes[1].set_title("(b) Per-round stage time", fontsize=8.2)
    axes[1].grid(axis="y", alpha=.2, linewidth=.4)
    fig.tight_layout(rect=(0, .05, 1, 1), pad=.45, h_pad=.75)
    save(fig, "fig4_compact_ablation_overhead")
    plt.close(fig)


def build_time_to_quality(rows: list[dict[str, str]]) -> None:
    """Derive first-hit resource efficiency with full provenance.

    The target for each classification setting is the strongest baseline's
    best accuracy. The competing reference is selected independently as the
    baseline with the smallest first-hit budget/time at that target.
    """
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] != "samsum":
            grouped[(row["dataset"], row["distribution"])].append(row)

    output = []
    for (dataset, distribution), group in grouped.items():
        ours_row = next((r for r in group if r["method"] == "Ours-FIM"), None)
        if ours_row is None:
            continue
        baseline_rows = [r for r in group if r["method"] != "Ours-FIM"]
        if not baseline_rows:
            continue
        target = max(number(r["best_metric"]) for r in baseline_rows)
        hits = []
        for row in group:
            path = ROOT / row["raw_result_path"]
            if not path.exists():
                continue
            hit = next(
                (trace for trace in read_csv(path) if number(trace.get("Accuracy")) >= target),
                None,
            )
            if hit is not None:
                hits.append({
                    "row": row,
                    "budget": number(hit.get("Cumulative_Budget")),
                    "time": number(hit.get("Cumulative_Time")),
                    "round": int(number(hit.get("Round"))),
                })
        ours = next((hit for hit in hits if hit["row"]["method"] == "Ours-FIM"), None)
        baselines = [hit for hit in hits if hit["row"]["method"] != "Ours-FIM"]
        if ours is None or not baselines:
            continue
        budget_ref = min(baselines, key=lambda hit: hit["budget"])
        time_ref = min(baselines, key=lambda hit: hit["time"])
        output.append({
            "dataset": dataset,
            "distribution": distribution,
            "metric": ours_row["metric"],
            "target_accuracy": f"{target:.4f}",
            "ours_method": ours_row["method"],
            "ours_seed": ours_row["seed"],
            "ours_budget": f"{ours['budget']:.4f}",
            "ours_time": f"{ours['time']:.4f}",
            "ours_round": ours["round"],
            "ours_run_id": ours_row["run_id"],
            "ours_config_path": ours_row["config_path"],
            "ours_raw_result_path": ours_row["raw_result_path"],
            "best_budget_baseline": budget_ref["row"]["method"],
            "budget_baseline_seed": budget_ref["row"]["seed"],
            "baseline_budget": f"{budget_ref['budget']:.4f}",
            "budget_saving_pct": f"{100 * (budget_ref['budget'] - ours['budget']) / budget_ref['budget']:.2f}",
            "budget_baseline_run_id": budget_ref["row"]["run_id"],
            "budget_baseline_config_path": budget_ref["row"]["config_path"],
            "budget_baseline_raw_result_path": budget_ref["row"]["raw_result_path"],
            "best_time_baseline": time_ref["row"]["method"],
            "time_baseline_seed": time_ref["row"]["seed"],
            "baseline_time": f"{time_ref['time']:.4f}",
            "time_saving_pct": f"{100 * (time_ref['time'] - ours['time']) / time_ref['time']:.2f}",
            "time_baseline_run_id": time_ref["row"]["run_id"],
            "time_baseline_config_path": time_ref["row"]["config_path"],
            "time_baseline_raw_result_path": time_ref["row"]["raw_result_path"],
        })
    order = {(dataset, distribution): i for i, (dataset, distribution, _) in enumerate(GROUPS)}
    output.sort(key=lambda row: order.get((row["dataset"], row["distribution"]), 999))
    columns = list(output[0]) if output else []
    write_csv(TABLE_OUT / "time_to_quality_baseline_peak.csv", output, columns)


def main() -> None:
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    rows = read_csv(RESULTS)
    build_matrix(rows)
    build_resource_curves(rows)
    build_combined_resource_curves(rows)
    build_ablation_overhead(rows)
    build_time_to_quality(rows)
    print(f"Wrote compact INFOCOM figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
