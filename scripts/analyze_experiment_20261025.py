#!/usr/bin/env python3
"""Build objective CSV/figures for the October 25 validation experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = ROOT / "paper_assets/02_experiments/matrix_score_20261025/full"
FIG_ROOT = ROOT / "paper_assets/05_figures/experiment_20261025"
METHODS = ("Local", "Local-Mean", "Server-Proxy")
VARIANTS = ("Full", "Server-Proxy", "Layer-Level", "No-Depth")
COLORS = {"Local": "#087e8b", "Local-Mean": "#dd8538", "Server-Proxy": "#914c91",
          "Full": "#087e8b", "Layer-Level": "#d57b29", "No-Depth": "#64748b"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_stats(values):
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else float("nan")


def score_study(allow_partial):
    out = FIG_ROOT / "matrix_score"
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    selections = []
    correlations = []
    for folder in sorted(SCORE_ROOT.glob("cifar100_*_seed*")):
        score_path = folder / "scores.json"
        if not score_path.exists():
            continue
        data = json.loads(score_path.read_text())
        dist, seed = data["distribution"], int(data["seed"])
        for checkpoint, stage in data["stages"].items():
            factor = np.asarray(stage["depth_factor"])
            top = {
                "Local-Mean": set(np.argsort(-np.asarray(stage["local_mean_raw"]) * factor)[:12]),
                "Server-Proxy": set(np.argsort(-np.asarray(stage["proxy_raw"]) * factor)[:12]),
            }
            for client in stage["clients"]:
                cid = int(client["client_id"])
                top[f"Local-{cid}"] = set(np.argsort(-np.asarray(client["raw"]) * factor)[:12])
            clients = [top[f"Local-{c['client_id']}"] for c in stage["clients"]]
            for i, left in enumerate(clients):
                for j, right in enumerate(clients[i + 1:], i + 1):
                    selections.append({"run_id": folder.name, "dataset": "cifar100", "method": "Local_vs_Local",
                                       "seed": seed, "metric": "top12_jaccard",
                                       "config_path": "scripts/run_matrix_score_validation_20261025.py",
                                       "raw_result_path": str(score_path.relative_to(ROOT)),
                                       "distribution": dist, "checkpoint": checkpoint, "client_a": i,
                                       "client_b": j, "jaccard": len(left & right) / len(left | right)})
                for method in ("Local-Mean", "Server-Proxy"):
                    right = top[method]
                    selections.append({"run_id": folder.name, "dataset": "cifar100", "method": "Local_vs_" + method,
                                       "seed": seed, "metric": "top12_jaccard",
                                       "config_path": "scripts/run_matrix_score_validation_20261025.py",
                                       "raw_result_path": str(score_path.relative_to(ROOT)),
                                       "distribution": dist, "checkpoint": checkpoint, "client_a": i,
                                       "client_b": method, "jaccard": len(left & right) / len(left | right)})
            for client in stage["clients"]:
                cid = int(client["client_id"])
                path = folder / "short_train" / f"{checkpoint}_client{cid}.csv"
                if not path.exists():
                    continue
                rows = read_csv(path)
                all_rows.extend(rows)
                gains = np.array([float(r["val_loss_reduction"]) for r in rows])
                for name, column in (("raw", "pred_local_raw"), ("depth", "pred_local_depth")):
                    result = spearmanr([float(r[column]) for r in rows], gains)
                    correlations.append({"run_id": folder.name, "dataset": "cifar100", "method": name,
                                         "seed": seed, "metric": "spearman_vs_validation_loss_reduction",
                                         "config_path": "scripts/run_matrix_score_validation_20261025.py",
                                         "raw_result_path": str(path.relative_to(ROOT)),
                                         "distribution": dist, "checkpoint": checkpoint,
                                         "client_id": cid, "n_configs": len(rows), "rho": float(result.statistic),
                                         "p_value": float(result.pvalue)})
    write_csv(out / "config_results.csv", all_rows)
    write_csv(out / "top12_selection_jaccard.csv", selections)
    write_csv(out / "client_checkpoint_spearman.csv", correlations)
    expected = 3 * 3 * 2 * 5
    if not allow_partial and len({(r["distribution"], r["seed"], r["checkpoint"], r["client_id"]) for r in all_rows}) != expected:
        raise RuntimeError(f"Score study is incomplete; expected {expected} client/checkpoint cells")
    if not all_rows:
        print("No completed score-validation results yet")
        return
    method_rows = []
    for key in sorted({(r["distribution"], r["checkpoint"], m) for r in all_rows for m in METHODS}):
        dist, checkpoint, method = key
        selected = [r for r in all_rows if r["distribution"] == dist
                    and r["checkpoint"] == checkpoint and method in r["method"].split("+")]
        if not selected:
            continue
        seed_means = [np.mean([float(r["val_loss_reduction"]) for r in selected if int(r["seed"]) == seed])
                      for seed in sorted({int(r["seed"]) for r in selected})]
        mean, std = summary_stats(seed_means)
        method_rows.append({"dataset": "cifar100", "distribution": dist, "checkpoint": checkpoint,
                            "method": method, "metric": "validation_loss_reduction",
                            "n_seeds": len(seed_means), "n_client_cells": len(selected),
                            "mean": mean, "std": std})
    write_csv(out / "method_loss_reduction_summary.csv", method_rows)
    corr_summary = []
    for key in sorted({(r["distribution"], r["checkpoint"], r["method"]) for r in correlations}):
        selected = [r for r in correlations if (r["distribution"], r["checkpoint"], r["method"]) == key]
        seed_means = [np.mean([float(r["rho"]) for r in selected if int(r["seed"]) == seed])
                      for seed in sorted({int(r["seed"]) for r in selected})]
        mean, std = summary_stats(seed_means)
        corr_summary.append({"distribution": key[0], "checkpoint": key[1], "method": key[2],
                             "metric": "spearman", "n_seeds": len(seed_means),
                             "n_client_cells": len(selected), "mean": mean, "std": std})
    write_csv(out / "spearman_summary.csv", corr_summary)
    jaccard_summary = []
    distance_summary = []
    for key in sorted({(r["distribution"], r["checkpoint"], r["method"]) for r in selections}):
        selected = [r for r in selections if (r["distribution"], r["checkpoint"], r["method"]) == key]
        seed_means = [np.mean([float(r["jaccard"]) for r in selected if int(r["seed"]) == seed])
                      for seed in sorted({int(r["seed"]) for r in selected})]
        mean, std = summary_stats(seed_means)
        jaccard_summary.append({"distribution": key[0], "checkpoint": key[1], "method": key[2],
                                "metric": "top12_jaccard", "n_seeds": len(seed_means),
                                "mean": mean, "std": std})
        distance_summary.append({"distribution": key[0], "checkpoint": key[1], "method": key[2],
                                 "metric": "top12_jaccard_distance", "n_seeds": len(seed_means),
                                 "mean": 1.0 - mean, "std": std})
    write_csv(out / "top12_jaccard_summary.csv", jaccard_summary)
    write_csv(out / "top12_jaccard_distance_summary.csv", distance_summary)
    paired_rows = []
    for key in sorted({(r["distribution"], r["checkpoint"], int(r["seed"]), int(r["client_id"]))
                       for r in all_rows}):
        dist, checkpoint, seed, client_id = key
        selected = [r for r in all_rows if (r["distribution"], r["checkpoint"],
                    int(r["seed"]), int(r["client_id"])) == key]
        gains = {method: float(row["val_loss_reduction"])
                 for row in selected for method in METHODS if method in row["method"].split("+")}
        if set(gains) != set(METHODS):
            raise RuntimeError(f"Missing paired scoring-source result for {key}: {set(METHODS) - set(gains)}")
        for comparison in ("Server-Proxy", "Local-Mean"):
            paired_rows.append({"distribution": dist, "checkpoint": checkpoint, "seed": seed,
                                "client_id": client_id, "comparison": f"Local_minus_{comparison}",
                                "metric": "paired_validation_loss_reduction_difference",
                                "difference": gains["Local"] - gains[comparison]})
    write_csv(out / "paired_loss_reduction.csv", paired_rows)
    paired_summary = []
    for key in sorted({(r["distribution"], r["checkpoint"], r["comparison"]) for r in paired_rows}):
        selected = [r for r in paired_rows if (r["distribution"], r["checkpoint"], r["comparison"]) == key]
        seed_means = [np.mean([float(r["difference"]) for r in selected if int(r["seed"]) == seed])
                      for seed in sorted({int(r["seed"]) for r in selected})]
        mean, std = summary_stats(seed_means)
        paired_summary.append({"distribution": key[0], "checkpoint": key[1],
                               "comparison": key[2], "metric": "paired_validation_loss_reduction_difference",
                               "n_seeds": len(seed_means), "n_client_cells": len(selected),
                               "mean": mean, "std": std})
    write_csv(out / "paired_loss_reduction_summary.csv", paired_summary)
    profile_rows = []
    for folder in sorted(SCORE_ROOT.glob("cifar100_*_seed*")):
        score_path = folder / "scores.json"
        if not score_path.exists():
            continue
        data = json.loads(score_path.read_text())
        for checkpoint, stage in data["stages"].items():
            profile_rows.append({"run_id": folder.name, "dataset": "cifar100",
                                 "method": "Local+Local-Mean+Server-Proxy", "seed": int(data["seed"]),
                                 "metric": "profiling_wall_clock_seconds",
                                 "config_path": "scripts/run_matrix_score_validation_20261025.py",
                                 "raw_result_path": str(score_path.relative_to(ROOT)),
                                 "distribution": data["distribution"], "checkpoint": checkpoint,
                                 "samples_per_local_client": int(data["profile_size"]),
                                 "proxy_samples": int(data["profile_size"]) * len(stage["clients"]),
                                 "local_profile_sec_sum": sum(float(c["profile_sec"]) for c in stage["clients"]),
                                 "local_profile_sec_max": max(float(c["profile_sec"]) for c in stage["clients"]),
                                 "proxy_profile_sec_sum": float(stage["proxy_profile_sec"])})
    write_csv(out / "profiling_overhead.csv", profile_rows)
    run_ledger = []
    for folder in sorted(SCORE_ROOT.glob("cifar100_*_seed*")):
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        run_ledger.append({"run_id": metadata["run_id"], "dataset": "cifar100",
                           "method": "Local+Local-Mean+Server-Proxy", "seed": metadata["seed"],
                           "metric": "top12_jaccard_distance+validation_loss_reduction+spearman",
                           "config_path": metadata["config_path"],
                           "raw_result_path": metadata["raw_result_path"],
                           "short_train_result_dir": str((folder / "short_train").relative_to(ROOT))})
    write_csv(out / "run_ledger.csv", run_ledger)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), constrained_layout=True)
    settings = [(dist, checkpoint) for checkpoint in ("early", "middle")
                for dist in ("iid", "dirichlet:1.0", "dirichlet:0.1")]
    labels = [f"{'IID' if dist == 'iid' else 'Dirichlet ' + dist.split(':')[1]}\n{checkpoint}"
              for dist, checkpoint in settings]
    positions = np.arange(len(settings))
    distance_rows = [next(r for r in distance_summary if
                     (r["distribution"], r["checkpoint"], r["method"]) ==
                     (dist, checkpoint, "Local_vs_Server-Proxy")) for dist, checkpoint in settings]
    axes[0].bar(positions, [r["mean"] for r in distance_rows],
                yerr=[r["std"] for r in distance_rows], capsize=2, color=COLORS["Local"])
    distance_max = max(r["mean"] + r["std"] for r in distance_rows)
    axes[0].set(ylabel="Top-12 Jaccard distance", ylim=(0, min(1, distance_max * 1.25)),
                title="(a) Selection disagreement")
    width = 0.25
    for offset, method in enumerate(METHODS):
        rows = [next(r for r in method_rows if (r["distribution"], r["checkpoint"], r["method"]) ==
                (dist, checkpoint, method)) for dist, checkpoint in settings]
        axes[1].bar(positions + (offset - 1) * width, [r["mean"] for r in rows], width,
                    yerr=[r["std"] for r in rows], capsize=2, label=method, color=COLORS[method])
    axes[1].set(ylabel="20-step validation loss reduction", title="(b) Short-term gain")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(positions, labels)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "scoring_source_p_comparison.pdf")
    fig.savefig(out / "scoring_source_p_comparison.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=False, constrained_layout=True)
    for row, checkpoint in enumerate(("early", "middle")):
        for col, dist in enumerate(("iid", "dirichlet:1.0", "dirichlet:0.1")):
            ax = axes[row, col]
            subset = [r for r in all_rows if r["distribution"] == dist and r["checkpoint"] == checkpoint]
            if subset:
                ax.scatter([float(r["pred_local_depth"]) for r in subset],
                           [float(r["val_loss_reduction"]) for r in subset],
                           s=16, alpha=0.5, color=COLORS["Local"])
            ax.set_title(f"{dist}, {checkpoint}")
            ax.set_xlabel("Predicted calibrated utility")
            ax.set_ylabel("Validation loss reduction")
            ax.grid(alpha=0.2)
    fig.savefig(out / "predicted_utility_vs_gain.pdf")
    fig.savefig(out / "predicted_utility_vs_gain.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    comparisons = ("Local_vs_Local", "Local_vs_Local-Mean", "Local_vs_Server-Proxy")
    distribution_order = ("iid", "dirichlet:1.0", "dirichlet:0.1")
    for idx, checkpoint in enumerate(("early", "middle")):
        ax = axes[idx]
        width = 0.23
        for offset, comparison in enumerate(comparisons):
            values = [next((r["mean"] for r in jaccard_summary
                            if (r["distribution"], r["checkpoint"], r["method"])
                            == (dist, checkpoint, comparison)), np.nan)
                      for dist in distribution_order]
            ax.bar(np.arange(3) + (offset - 1) * width, values, width, label=comparison.replace("Local_vs_", "vs "))
        ax.set_xticks(range(3), ("IID", "D1.0", "D0.1"))
        ax.set_ylim(0, 1)
        ax.set_title(checkpoint.capitalize())
        ax.set_ylabel("Top-12 Jaccard overlap")
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "top12_selection_overlap.pdf")
    fig.savefig(out / "top12_selection_overlap.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for idx, checkpoint in enumerate(("early", "middle")):
        ax = axes[idx]
        width = 0.23
        for offset, method in enumerate(METHODS):
            values = [next((r for r in method_rows if
                            (r["distribution"], r["checkpoint"], r["method"])
                            == (dist, checkpoint, method)), None)
                      for dist in distribution_order]
            means = [r["mean"] if r is not None else np.nan for r in values]
            stds = [r["std"] if r is not None and np.isfinite(r["std"]) else 0.0 for r in values]
            ax.bar(np.arange(3) + (offset - 1) * width, means, width,
                   yerr=stds, capsize=2, label=method, color=COLORS[method])
        ax.set_xticks(range(3), ("IID", "D1.0", "D0.1"))
        ax.set_title(checkpoint.capitalize())
        ax.set_ylabel("20-step validation loss reduction")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "scoring_source_short_train_gain.pdf")
    fig.savefig(out / "scoring_source_short_train_gain.png", dpi=220)
    plt.close(fig)
    positive = sum(float(r["rho"]) > 0 for r in correlations if r["method"] == "depth")
    checked = sum(r["method"] == "depth" for r in correlations)
    lines = ["# 矩阵评分实验结果", "",
             f"已完成 {checked} 个客户端×检查点相关性单元；深度校准预测效用与验证损失下降的 Spearman 相关为正的单元数：{positive}/{checked}。",
             "", "以下均值和标准差先在同一种子内对客户端求均值，再跨种子计算；这只是描述性统计，不代表因果证明。", "",
             "| 划分 | 检查点 | 评分 | Spearman 均值±标准差 |", "|---|---|---|---:|"]
    for row in corr_summary:
        lines.append(f"| {row['distribution']} | {row['checkpoint']} | {row['method']} | {row['mean']:.3f} ± {row['std']:.3f} |")
    (out / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Score study written to", out, "with", len(all_rows), "config rows")


def ablation_study(allow_partial):
    out = FIG_ROOT / "sst2_ablation"
    out.mkdir(parents=True, exist_ok=True)
    run_rows = []
    curves = defaultdict(list)
    for path in sorted((ROOT / "evaluation_results/sst2").glob("SCORE_ABL_20261025_SST2_D1_*/metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("pilot"):
            continue
        raw = ROOT / metadata["raw_result_path"]
        if not raw.exists():
            continue
        records = read_csv(raw)
        if not records:
            continue
        method = metadata["method"]
        seed = int(metadata["seed"])
        budget = np.array([float(r["Cumulative_Budget"]) for r in records])
        accuracy = np.array([float(r["Accuracy"]) for r in records])
        consumed = np.array([float(r["Consumed_Budget"]) for r in records])
        if not np.isclose(consumed.sum(), budget[-1], atol=1e-3):
            raise RuntimeError(f"Inconsistent cumulative budget: {raw}")
        run_rows.append({"run_id": metadata["run_id"], "dataset": "sst2", "method": method,
                         "seed": seed, "metric": "accuracy", "config_path": metadata["config_path"],
                         "raw_result_path": metadata["raw_result_path"],
                         "final_accuracy": float(accuracy[-1]), "best_accuracy": float(accuracy.max()),
                         "final_budget": float(budget[-1]), "rounds": len(records)})
        curves[method].append((seed, budget, accuracy))
    write_csv(out / "run_summary.csv", run_rows)
    expected = {(variant, seed) for variant in VARIANTS for seed in (2024, 2025, 2026)}
    observed = {(r["method"], r["seed"]) for r in run_rows}
    if not allow_partial and observed != expected:
        raise RuntimeError(f"Ablation incomplete; missing {sorted(expected - observed)}")
    if not run_rows:
        print("No completed SST-2 ablation results yet")
        return
    summary = []
    for method in VARIANTS:
        selected = [r for r in run_rows if r["method"] == method]
        if not selected:
            continue
        final_mean, final_std = summary_stats([r["final_accuracy"] for r in selected])
        best_mean, best_std = summary_stats([r["best_accuracy"] for r in selected])
        budget_mean, budget_std = summary_stats([r["final_budget"] for r in selected])
        summary.append({"dataset": "sst2", "distribution": "dirichlet:1.0", "method": method,
                        "n_seeds": len(selected), "final_accuracy_mean": final_mean,
                        "final_accuracy_std": final_std, "best_accuracy_mean": best_mean,
                        "best_accuracy_std": best_std, "used_budget_mean": budget_mean,
                        "used_budget_std": budget_std})
    write_csv(out / "three_seed_mean_std.csv", summary)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    max_budget = min(max(budget) for runs in curves.values() for _, budget, _ in runs)
    grid = np.linspace(0, max_budget, 161)
    for method in VARIANTS:
        runs = curves.get(method, [])
        if not runs:
            continue
        vals = []
        for seed, budget, accuracy in runs:
            ax.plot(budget, accuracy, color=COLORS[method], alpha=0.16, linewidth=0.9)
            unique_budget, unique_idx = np.unique(budget[::-1], return_index=True)
            unique_acc = accuracy[::-1][unique_idx]
            vals.append(np.interp(grid, unique_budget, unique_acc))
        stack = np.vstack(vals)
        mean = stack.mean(axis=0)
        ax.plot(grid, mean, label=method, color=COLORS[method], linewidth=2.0,
                linestyle="--" if method in ("Layer-Level", "No-Depth") else "-")
        if len(vals) > 1:
            ax.fill_between(grid, mean - stack.std(axis=0, ddof=1),
                            mean + stack.std(axis=0, ddof=1), color=COLORS[method], alpha=0.09)
    ax.set(xlabel="Cumulative matrix-equivalent budget", ylabel="SST-2 validation accuracy",
           title="SST-2 / Dirichlet 1.0: controlled ablation")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out / "accuracy_vs_equal_budget.pdf")
    fig.savefig(out / "accuracy_vs_equal_budget.png", dpi=220)
    plt.close(fig)
    lines = ["# SST-2 消融实验结果", "",
             "三种子结果均使用同一预算、训练参数和代理集预留规则；下表报告运行轨迹中的最终与最佳验证准确率。", "",
             "| 方法 | 种子数 | 最终准确率均值±标准差 | 最佳准确率均值±标准差 | 实际预算均值±标准差 |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['n_seeds']} | "
                     f"{row['final_accuracy_mean']:.4f} ± {row['final_accuracy_std']:.4f} | "
                     f"{row['best_accuracy_mean']:.4f} ± {row['best_accuracy_std']:.4f} | "
                     f"{row['used_budget_mean']:.1f} ± {row['used_budget_std']:.1f} |")
    lines.extend(["", "图中横轴为实际累计矩阵等价成本；对照组可能提前停止或消耗不同预算，故只在共同覆盖的预算区间比较均值曲线。"])
    (out / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Ablation study written to", out, "with", len(run_rows), "runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=["score", "ablation", "all"])
    parser.add_argument("--allow-partial", action="store_true")
    arg = parser.parse_args()
    if arg.study in ("score", "all"):
        score_study(arg.allow_partial)
    if arg.study in ("ablation", "all"):
        ablation_study(arg.allow_partial)
