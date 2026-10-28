#!/usr/bin/env python3
"""Plot the client-local P2 matrix-minus-layer validation-loss gains."""

import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "paper_assets/05_figures/experiment_p2_20261027/paired_summary.csv"
SEEDS = ROOT / "paper_assets/05_figures/experiment_p2_20261027/paired_seed_differences.csv"
OUTPUT = ROOT / "paper_assets/04_manuscript/figures/fig1_p2_local_only.pdf"

DEEP_BLUE = "#21496b"
LIGHT_BLUE = "#8ab9d4"
LIGHT_GRAY = "#aeb9c0"
PANELS = (("iid", "(a) IID"), ("dirichlet:1.0", "(b) Dirichlet 1.0"),
          ("dirichlet:0.1", "(c) Dirichlet 0.1"))
CHECKPOINTS = (("early", "Initial"), ("middle", "Warm-up"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    seed_rows = [r for r in read_csv(SEEDS) if r["source"] == "Local"]
    summary_rows = {(r["distribution"], r["checkpoint"]): r
                    for r in read_csv(SUMMARY) if r["source"] == "Local"}
    assert len(seed_rows) == 18 and len(summary_rows) == 6
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#83949e", "xtick.color": "#304451",
        "ytick.color": "#304451", "text.color": "#263846",
    })
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 2.5), sharex=True)
    for ax, (distribution, title) in zip(axes, PANELS):
        low, high = 0.0, 0.0
        for x, (checkpoint, _) in enumerate(CHECKPOINTS):
            matches = sorted((r for r in seed_rows if r["distribution"] == distribution
                              and r["checkpoint"] == checkpoint), key=lambda r: int(r["seed"]))
            assert len(matches) == 3 and {int(r["seed"]) for r in matches} == {2024, 2025, 2026}
            values = [float(r["mean_matrix_minus_layer"]) for r in matches]
            mean, std = statistics.mean(values), statistics.stdev(values)
            summary = summary_rows[(distribution, checkpoint)]
            assert math.isclose(mean, float(summary["mean_matrix_minus_layer"]), abs_tol=1e-12)
            assert math.isclose(std, float(summary["std_across_seeds"]), abs_tol=1e-12)
            for offset, value in zip((-0.12, -0.04, 0.04), values):
                ax.plot(x + offset, value, marker="o", linestyle="none", ms=3.2,
                        markerfacecolor="white", markeredgecolor=LIGHT_BLUE,
                        markeredgewidth=1.0, zorder=3)
            ax.errorbar(x + 0.13, mean, yerr=std, fmt="D", ms=3.6,
                        color=DEEP_BLUE, ecolor=DEEP_BLUE, elinewidth=0.85,
                        capsize=1.9, capthick=0.85, zorder=4)
            low = min(low, min(values), mean - std)
            high = max(high, max(values), mean + std)
        span = max(high - low, 0.002)
        ax.set_ylim(low - 0.12 * span, high + 0.16 * span)
        ax.axhline(0, color=LIGHT_GRAY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        ax.set_title(title, loc="left", fontsize=7.5, fontweight="semibold", pad=1)
        ax.set_xlim(-0.32, 1.32)
        ax.set_xticks([0, 1], [label for _, label in CHECKPOINTS])
        ax.tick_params(axis="both", labelsize=6.5, length=2, pad=1)
        ax.tick_params(axis="x", labelbottom=True)
    fig.text(0.015, 0.5, "Matrix–Layer validation-loss gain", rotation=90,
             va="center", fontsize=7)
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.09, top=0.94, hspace=0.44)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
