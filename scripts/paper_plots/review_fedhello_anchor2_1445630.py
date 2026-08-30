#!/usr/bin/env python3
"""Render paper-style review figures for Fed-HeLLo anchor2 job 1445630."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper_assets/06_tables/final/table1_main_results.csv"
OUT = ROOT / "paper_assets/05_figures/fedhello_anchor2_1445630_review"
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
]

RERUNS = {
    ("qqp", "dirichlet:1.0"): (
        "evaluation_results/qqp/"
        "RERUN_FIG45_FEDHELLO_ANCHOR2_QQP_D1__"
        "Dist-dirichlet_1.0_Bgt-9500_20260729-130630/eval_Fed-HeLLo.csv"
    ),
    ("mnli", "dirichlet:1.0"): (
        "evaluation_results/mnli/"
        "RERUN_FIG45_FEDHELLO_ANCHOR2_MNLI_D1__"
        "Dist-dirichlet_1.0_Bgt-9500_20260729-130630/eval_Fed-HeLLo.csv"
    ),
    ("qnli", "iid"): (
        "evaluation_results/qnli/"
        "RERUN_FIG45_FEDHELLO_ANCHOR2_QNLI_IID__"
        "Dist-iid_Bgt-4800_20260729-130638/eval_Fed-HeLLo.csv"
    ),
}


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


def review_rows() -> tuple[list[dict[str, str]], dict[tuple[str, str], str]]:
    rows = read_csv(RESULTS)
    old_paths: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["dataset"], row["distribution"])
        if row["method"] == "Fed-HeLLo" and key in RERUNS:
            old_paths[key] = row["raw_result_path"]
            row["raw_result_path"] = RERUNS[key]
    return rows, old_paths


def build_full_curves(rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["distribution"])].append(row)

    specs = [
        ("Cumulative_Budget", "Cumulative matrix budget", "fig4_budget_curves_anchor2"),
        ("Cumulative_Time", "Modeled elapsed time (s)", "fig5_time_curves_anchor2"),
    ]
    for xcol, xlabel, stem in specs:
        fig, axes = plt.subplots(2, 6, figsize=(7.15, 1.72), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)

        handles = {}
        for index, (dataset, distribution, title) in enumerate(GROUPS):
            ax = axes[index // 6][index % 6]
            ax.set_visible(True)
            all_y: list[float] = []
            setting_rows = sorted(
                grouped[(dataset, distribution)], key=lambda row: METHODS.index(row["method"])
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

        fig.text(.50, .012, xlabel, ha="center", va="bottom", fontsize=5.7)
        legend_ax = axes[1][5]
        legend_ax.set_visible(True)
        legend_ax.axis("off")
        ordered = [handles[method] for method in METHODS if method in handles]
        labels = [SHORT[method] for method in METHODS if method in handles]
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
            left=.055, right=.995, bottom=.14, top=.955, wspace=.34, hspace=.38
        )
        save(fig, stem)
        plt.close(fig)


def build_local_comparison(
    rows: list[dict[str, str]], old_paths: dict[tuple[str, str], str]
) -> None:
    import matplotlib.pyplot as plt

    row_lookup = {
        (row["dataset"], row["distribution"], row["method"]): row for row in rows
    }
    order = [
        ("mnli", "dirichlet:1.0", "MNLI/Non-IID 1.0"),
        ("qqp", "dirichlet:1.0", "QQP/Non-IID 1.0"),
        ("qnli", "iid", "QNLI/IID"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 1.72))
    handles = []
    labels = []
    for ax, (dataset, distribution, title) in zip(axes, order):
        old_trace = read_csv(ROOT / old_paths[(dataset, distribution)])
        new_trace = read_csv(ROOT / RERUNS[(dataset, distribution)])
        ours_row = row_lookup[(dataset, distribution, "Ours-FIM")]
        ours_trace = read_csv(ROOT / ours_row["raw_result_path"])
        series = [
            ("Original Fed-HeLLo", old_trace, "#777777", "--", .9),
            ("Fed-HeLLo-anchor2", new_trace, COLORS["Fed-HeLLo"], "-", 1.25),
            ("FedMATRIX", ours_trace, COLORS["Ours-FIM"], "-", 1.55),
        ]
        all_y = []
        for label, trace, color, linestyle, linewidth in series:
            line, = ax.plot(
                [number(item["Round"]) for item in trace],
                [number(item["Accuracy"]) for item in trace],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
            all_y.extend(number(item["Accuracy"]) for item in trace)
            if len(handles) < 3:
                handles.append(line)
                labels.append(label)
        pad = max(.018, .08 * (max(all_y) - min(all_y)))
        ax.set_ylim(max(0, min(all_y) - pad), min(1, max(all_y) + pad))
        ax.set_title(title, fontsize=7.2, pad=2)
        ax.set_xlabel("Communication round", fontsize=6.4)
        ax.tick_params(labelsize=5.7, length=1.8, pad=1)
        ax.grid(alpha=.20, linewidth=.35)
    axes[0].set_ylabel("Accuracy", fontsize=6.4)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(.5, 1.035),
        ncol=3,
        frameon=False,
        fontsize=6.7,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=.065, right=.995, bottom=.22, top=.82, wspace=.24)
    save(fig, "fedhello_original_anchor2_fedmatrix_round_comparison")
    plt.close(fig)


def main() -> None:
    rows, old_paths = review_rows()
    missing = set(RERUNS) - set(old_paths)
    if missing:
        raise RuntimeError(f"Missing final-ledger Fed-HeLLo rows: {sorted(missing)}")
    build_full_curves(rows)
    build_local_comparison(rows, old_paths)
    print(f"Wrote review figures to {OUT}")


if __name__ == "__main__":
    main()
