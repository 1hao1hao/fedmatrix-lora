#!/usr/bin/env python3
"""Plot scoring-v2 pilot curves and expectation checks."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "paper_assets" / "02_experiments" / "scoring_v2_pilot_summary.csv"
OUT_DIR = ROOT / "paper_assets" / "05_figures" / "scoring_v2_pilot"
CHECK_PATH = ROOT / "paper_assets" / "02_experiments" / "scoring_v2_pilot_expectation_check.csv"

METHOD_ORDER = ["Old_Random_FIM", "Ours_FIM", "MS", "Fed-HeLLo"]
METHOD_LABELS = {
    "Old_Random_FIM": "Old Random-FIM",
    "Ours_FIM": "New Ours-FIM",
    "MS": "MS",
    "Fed-HeLLo": "Fed-HeLLo",
}
METHOD_COLORS = {
    "Old_Random_FIM": "#6b7280",
    "Ours_FIM": "#1f77b4",
    "MS": "#d62728",
    "Fed-HeLLo": "#2ca02c",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=None):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_summary():
    rows = read_csv(SUMMARY_PATH)
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["dataset"], row["distribution"])][row["method"]] = row
    return rows, groups


def metric_series(raw_result_path: str, metric: str):
    path = ROOT / raw_result_path
    rows = read_csv(path)
    xs, ys = [], []
    for idx, row in enumerate(rows, start=1):
        xs.append(to_float(row.get("Round"), idx))
        ys.append(to_float(row.get(metric)))
    return xs, ys


def savefig(name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        plt.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)


def make_expectation_check(groups):
    rows = []
    for (dataset, distribution), methods in sorted(groups.items()):
        old = methods.get("Old_Random_FIM")
        ours = methods.get("Ours_FIM")
        ms = methods.get("MS")
        if not old or not ours:
            continue
        old_best = to_float(old.get("best_metric"))
        old_final = to_float(old.get("final_metric"))
        old_gap = to_float(old.get("final_best_gap"))
        ours_best = to_float(ours.get("best_metric"))
        ours_final = to_float(ours.get("final_metric"))
        ours_gap = to_float(ours.get("final_best_gap"))
        ms_best = to_float(ms.get("best_metric")) if ms else None
        ms_final = to_float(ms.get("final_metric")) if ms else None

        rows.append({
            "dataset": dataset,
            "distribution": distribution,
            "old_final": old_final,
            "ours_final": ours_final,
            "old_best": old_best,
            "ours_best": ours_best,
            "old_gap": old_gap,
            "ours_gap": ours_gap,
            "ms_final": ms_final if ms_final is not None else "",
            "ms_best": ms_best if ms_best is not None else "",
            "ours_final_minus_old": (ours_final - old_final) if None not in (ours_final, old_final) else "",
            "ours_best_minus_old": (ours_best - old_best) if None not in (ours_best, old_best) else "",
            "ours_final_minus_ms": (ours_final - ms_final) if None not in (ours_final, ms_final) else "",
            "ours_best_minus_ms": (ours_best - ms_best) if None not in (ours_best, ms_best) else "",
            "gap_reduction_vs_old": (old_gap - ours_gap) if None not in (old_gap, ours_gap) else "",
            "meets_new_gt_old": bool(ours_best is not None and old_best is not None and ours_best > old_best),
            "meets_ge_ms_or_gap_shrinks": bool(
                ours_best is not None
                and ms_best is not None
                and (ours_best >= ms_best or abs(ours_best - ms_best) < abs(old_best - ms_best))
            ),
            "meets_gap_smaller": bool(ours_gap is not None and old_gap is not None and ours_gap < old_gap),
        })
    write_csv(
        CHECK_PATH,
        rows,
        [
            "dataset", "distribution", "old_final", "ours_final", "old_best", "ours_best",
            "old_gap", "ours_gap", "ms_final", "ms_best", "ours_final_minus_old",
            "ours_best_minus_old", "ours_final_minus_ms", "ours_best_minus_ms",
            "gap_reduction_vs_old", "meets_new_gt_old", "meets_ge_ms_or_gap_shrinks",
            "meets_gap_smaller",
        ],
    )
    return rows


def plot_curves(groups):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, ((dataset, distribution), methods) in zip(axes, sorted(groups.items())):
        for method in METHOD_ORDER:
            row = methods.get(method)
            if not row:
                continue
            xs, ys = metric_series(row["raw_result_path"], row["metric"])
            ax.plot(
                xs,
                ys,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                linewidth=2.0 if method == "Ours_FIM" else 1.6,
            )
        ax.set_title(f"{dataset} / {distribution}", fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig("scoring_v2_pilot_accuracy_curves")
    plt.close(fig)


def plot_bars(groups):
    labels = [f"{dataset}\n{distribution}" for dataset, distribution in sorted(groups)]
    x = list(range(len(labels)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, method in enumerate(METHOD_ORDER):
        vals = []
        for key in sorted(groups):
            vals.append(to_float(groups[key].get(method, {}).get("best_metric"), 0.0))
        offsets = [pos + (idx - 1.5) * width for pos in x]
        ax.bar(offsets, vals, width=width, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best accuracy")
    ax.set_title("Scoring-v2 pilot: best metric comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout()
    savefig("scoring_v2_pilot_best_bars")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    old_gaps, ours_gaps = [], []
    for key in sorted(groups):
        old_gaps.append(to_float(groups[key].get("Old_Random_FIM", {}).get("final_best_gap"), 0.0))
        ours_gaps.append(to_float(groups[key].get("Ours_FIM", {}).get("final_best_gap"), 0.0))
    ax.bar([pos - width / 2 for pos in x], old_gaps, width=width, label="Old Random-FIM", color=METHOD_COLORS["Old_Random_FIM"])
    ax.bar([pos + width / 2 for pos in x], ours_gaps, width=width, label="New Ours-FIM", color=METHOD_COLORS["Ours_FIM"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best - final gap")
    ax.set_title("Scoring-v2 pilot: stability gap")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig("scoring_v2_pilot_final_best_gap")
    plt.close(fig)


def main():
    _, groups = load_summary()
    checks = make_expectation_check(groups)
    plot_curves(groups)
    plot_bars(groups)
    print(f"Wrote expectation check: {CHECK_PATH.relative_to(ROOT)}")
    print(f"Wrote figures to: {OUT_DIR.relative_to(ROOT)}")
    print(f"Checked groups: {len(checks)}")


if __name__ == "__main__":
    main()
