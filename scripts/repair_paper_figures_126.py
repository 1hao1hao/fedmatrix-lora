#!/usr/bin/env python3
"""Rebuild the three paper figures from existing summaries and artwork conventions."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_assets/04_manuscript/figures"
P2 = ROOT / "paper_assets/05_figures/experiment_p2_20261027/paired_summary.csv"
B = ROOT / "paper_assets/05_figures/experiment_20261025/P_A_B_review/B_three_seed_summary.csv"

BLUE = "#21496b"
ORANGE = "#ca772e"
TEAL = "#168078"
GREEN = "#43865a"
RED = "#b65855"
GRAY = "#aeb8bd"
INK = "#263846"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#82929d",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "savefig.facecolor": "white",
})


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fig1():
    data = {(r["distribution"], r["checkpoint"], r["source"]): r for r in rows(P2)}
    panels = [("iid", "(a) IID"), ("dirichlet:1.0", "(b) Dirichlet 1.0"),
              ("dirichlet:0.1", "(c) Dirichlet 0.1")]
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 3.65), sharex=True)
    for ax, (dist, name) in zip(axes, panels):
        for source, offset, color in [("Local", -0.095, BLUE), ("Proxy", 0.095, ORANGE)]:
            matches = [data[(dist, cp, source)] for cp in ("early", "middle")]
            y = [float(r["mean_matrix_minus_layer"]) for r in matches]
            err = [float(r["std_across_seeds"]) for r in matches]
            ax.errorbar([offset, 1 + offset], y, yerr=err, fmt="o", ms=4.1,
                        capsize=2.0, capthick=0.9, elinewidth=1, color=color,
                        label=source, zorder=3)
        ax.axhline(0, lw=0.8, color=GRAY, ls=(0, (3, 2)), zorder=0)
        ax.set_title(name, loc="left", fontsize=8, fontweight="semibold", pad=2)
        ax.set_xlim(-0.35, 1.35)
        ax.grid(axis="y", color="#e7ebed", lw=0.55)
        ax.tick_params(axis="both", labelsize=7, length=2, pad=2)
        ax.set_xticks([0, 1], ["Early", "Middle"])
        ax.tick_params(axis="x", labelbottom=True)
    axes[0].legend(loc="upper right", frameon=False, fontsize=7, ncol=2,
                   handletextpad=0.2, columnspacing=0.65, borderaxespad=0)
    axes[1].set_ylabel("Matrix–Layer loss-reduction gain", fontsize=7, labelpad=5)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.94, bottom=0.07, hspace=0.38)
    fig.savefig(OUT / "fig1_p2_granularity_repaired.pdf")
    plt.close(fig)


def box(ax, xy, wh, color, title, details=(), title_size=8.5):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
                                fc=color[0], ec=color[1], lw=1.2))
    ax.text(x + 0.022, y + h - 0.07, title, va="top", ha="left",
            fontsize=title_size, fontweight="bold", color=color[2], linespacing=1.05)
    for n, line in enumerate(details):
        ax.text(x + 0.022, y + h - 0.245 - 0.072 * n, line, va="top", ha="left",
                fontsize=7.3, color=INK)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def fig2():
    # The layout follows the older four-stage, color-coded client/server artwork.
    fig, ax = plt.subplots(figsize=(7.0, 2.87))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.012, 0.79), 0.976, 0.17,
                                boxstyle="round,pad=0.008,rounding_size=0.025",
                                fc="#f2f5f6", ec="#cbd3d8", lw=0.8))
    ax.text(0.04, 0.875, "Frozen backbone  +  global LoRA adapter", fontsize=9,
            fontweight="bold", va="center", color=INK)
    ax.text(0.72, 0.875, "Candidate targets  Q · K · V", fontsize=8.2,
            va="center", color=INK)
    blue = ("#e9f2f8", "#89b2c9", BLUE)
    orange = ("#fff1e5", "#e2ad78", ORANGE)
    teal = ("#e8f5f2", "#87bcb4", TEAL)
    box(ax, (0.015, 0.24), (0.223, 0.49), blue,
        "1  Client-conditioned\n    matrix profiling", ("Local mini-batch", "Matrix utility scores"))
    box(ax, (0.270, 0.24), (0.223, 0.49), blue,
        "2  Feasible configs", ("Memory + latency check", "Client configuration menu"))
    box(ax, (0.525, 0.24), (0.223, 0.49), orange,
        "3  Server coordination", ("Budget + latency aware", "Joint configuration choice"))
    box(ax, (0.780, 0.24), (0.205, 0.49), teal,
        "4  Local training\n    & aggregation", ("Assigned LoRA matrices", "Matrix-wise updates"))
    for n, label in enumerate(("Q", "K", "V")):
        ax.add_patch(FancyBboxPatch((0.043 + 0.057*n, 0.285), 0.046, 0.063,
                                    boxstyle="round,pad=0.002,rounding_size=0.008",
                                    fc="#bed9e8", ec="#6f9db7", lw=0.7))
        ax.text(0.066 + 0.057*n, 0.316, label, ha="center", va="center", fontsize=7.5, color=BLUE)
    for n, width in enumerate((0.11, 0.15, 0.09)):
        ax.add_patch(FancyBboxPatch((0.297, 0.305 + 0.024*n), width, 0.012,
                                    boxstyle="round,pad=0,rounding_size=0.004",
                                    fc="#8fb9cf", ec="none"))
    ax.add_patch(FancyBboxPatch((0.552, 0.30), 0.17, 0.038,
                                boxstyle="round,pad=0.002,rounding_size=0.008",
                                fc="#f7d9b8", ec="none"))
    ax.add_patch(FancyBboxPatch((0.552, 0.30), 0.105, 0.038,
                                boxstyle="round,pad=0.002,rounding_size=0.008",
                                fc=ORANGE, ec="none"))
    for n in range(3):
        ax.plot(0.813 + 0.05*n, 0.318, "o", ms=5, color=TEAL, mec="white", mew=0.4)
    arrow(ax, 0.92, 0.318, 0.955, 0.318, TEAL, 1.0)
    for x in (0.238, 0.493, 0.748):
        arrow(ax, x + 0.003, 0.47, x + 0.029, 0.47, "#657d89")
    ax.add_patch(Rectangle((0.015, 0.15), 0.478, 0.055, fc="#e9f2f8", ec="none"))
    ax.add_patch(Rectangle((0.525, 0.15), 0.223, 0.055, fc="#fff1e5", ec="none"))
    ax.add_patch(Rectangle((0.780, 0.15), 0.205, 0.055, fc="#e8f5f2", ec="none"))
    ax.text(0.254, 0.177, "CLIENT SIDE", color=BLUE, fontsize=8, fontweight="bold", ha="center", va="center")
    ax.text(0.637, 0.177, "SERVER SIDE", color=ORANGE, fontsize=8, fontweight="bold", ha="center", va="center")
    ax.text(0.882, 0.177, "AGGREGATION", color=TEAL, fontsize=8, fontweight="bold", ha="center", va="center")
    arrow(ax, 0.887, 0.11, 0.11, 0.11, "#b5c0c5", 1.0)
    ax.text(0.50, 0.035, "next communication round", fontsize=7.5,
            color="#70818a", ha="center", va="center")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.03)
    fig.savefig(OUT / "fig2_method_overview_repaired.pdf")
    plt.close(fig)


def fig6():
    data = {r["method"]: r for r in rows(B) if r["dataset"] == "sst2" and r["distribution"] == "dirichlet:1.0"}
    methods = ["Full", "Server-Proxy", "Layer-Level", "No-Depth"]
    assert len(data) == 4 and len({float(r["used_budget_mean"]) for r in data.values()}) == 1
    colors = [BLUE, ORANGE, GREEN, RED]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15), sharey=True)
    for ax, (key, title) in zip(axes, [("final", "(a) Final accuracy"), ("best", "(b) Best accuracy")]):
        for x, method, color in zip(range(4), methods, colors):
            r = data[method]
            mean = 100 * float(r[f"{key}_accuracy_mean"])
            std = 100 * float(r[f"{key}_accuracy_std"])
            ax.errorbar(x, mean, yerr=std, fmt="o", ms=6, color=color,
                        ecolor=color, capsize=2.5, elinewidth=1.2, capthick=1.2)
        ax.set_title(title, fontsize=9, fontweight="semibold", loc="left", pad=4)
        ax.set_xticks(range(4), ["Full", "Server-\nProxy", "Layer-\nLevel", "No-Depth"])
        ax.set_xlim(-0.45, 3.45)
        ax.set_ylim(88.2, 92.1)
        ax.set_yticks([89, 90, 91, 92])
        ax.set_xlabel("Variant", fontsize=8)
        ax.grid(axis="y", color="#e7ebed", lw=0.7)
        ax.tick_params(axis="both", labelsize=7.5, length=2)
    axes[0].set_ylabel("Validation accuracy (%)", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.26, wspace=0.16)
    fig.savefig(OUT / "fig6_ablation_accuracy_repaired.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig6()
