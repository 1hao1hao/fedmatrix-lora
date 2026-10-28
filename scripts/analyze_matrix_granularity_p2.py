#!/usr/bin/env python3
"""Strict paired analysis of the fixed-protocol CIFAR100 P2 experiment."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_assets/02_experiments/matrix_granularity_p2_20261027"
P_SOURCE = ROOT / "paper_assets/02_experiments/matrix_score_20261025/full"
OUT = ROOT / "paper_assets/05_figures/experiment_p2_20261027"
SEEDS = (2024, 2025, 2026)
DISTRIBUTIONS = ("iid", "dirichlet:1.0", "dirichlet:0.1")
CHECKPOINTS = ("early", "middle")
METHODS = ("Local-Matrix", "Local-Layer", "Proxy-Matrix", "Proxy-Layer")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    if not rows and not fieldnames:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else fieldnames,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def average_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask_rows = []
    all_results = []
    ledger = []
    score_rows = []
    for seed in SEEDS:
        for distribution in DISTRIBUTIONS:
            folder = SOURCE / f"p2_cifar100_{distribution.replace(':', '_')}_seed{seed}"
            mask_file = folder / "masks.csv"
            if not mask_file.exists():
                raise RuntimeError(f"Missing pre-generated P2 masks: {mask_file}")
            masks = read_csv(mask_file)
            if len(masks) != 40:
                raise RuntimeError(f"Expected 40 method masks per group: {mask_file}")
            mask_rows.extend(masks)
            score_path = P_SOURCE / f"cifar100_{distribution.replace(':', '_')}_seed{seed}" / "scores.json"
            score_data = json.loads(score_path.read_text(encoding="utf-8"))
            targets = score_data["candidate_modules"]
            for checkpoint in CHECKPOINTS:
                stage = score_data["stages"][checkpoint]
                factor = np.asarray(stage["depth_factor"], dtype=float)
                for client_id in range(5):
                    sources = {"Local": stage["clients"][client_id]["raw"],
                               "Proxy": stage["proxy_raw"]}
                    for source_name, raw_values in sources.items():
                        raw = np.asarray(raw_values, dtype=float)
                        calibrated = raw * factor
                        divisor = max(float(calibrated.max()), 1e-12)
                        for matrix_index, target in enumerate(targets):
                            score_rows.append({
                                "run_id": folder.name, "dataset": "cifar100",
                                "method": source_name, "seed": seed,
                                "metric": "normalized_profile_score",
                                "config_path": "scripts/run_matrix_score_validation_20261025.py",
                                "raw_result_path": str(score_path.relative_to(ROOT)),
                                "distribution": distribution, "checkpoint": checkpoint,
                                "client_id": client_id, "matrix_index": matrix_index,
                                "candidate_module": target, "raw_score": raw[matrix_index],
                                "depth_factor": factor[matrix_index],
                                "calibrated_score": calibrated[matrix_index],
                                "normalization_divisor": divisor,
                                "normalized_score": calibrated[matrix_index] / divisor,
                            })
            for checkpoint in CHECKPOINTS:
                for client_id in range(5):
                    path = folder / "results" / f"{checkpoint}_client{client_id}.csv"
                    if not path.exists():
                        raise RuntimeError(f"P2 training incomplete: {path}")
                    records = read_csv(path)
                    if [r["method"] for r in records] != list(METHODS):
                        raise RuntimeError(f"Missing or reordered method results: {path}")
                    all_results.extend(records)
            ledger.append({"run_id": folder.name, "dataset": "cifar100",
                           "method": "+".join(METHODS), "seed": seed,
                           "metric": "paired_validation_loss_reduction_difference",
                           "config_path": "scripts/run_matrix_granularity_p2.py",
                           "raw_result_path": str((folder / "results").relative_to(ROOT)),
                           "mask_path": str(mask_file.relative_to(ROOT)),
                           "distribution": distribution})
    if len(mask_rows) != 360 or len(all_results) != 360:
        raise RuntimeError("P2 must retain all 360 masks and method records")
    write_csv(OUT / "run_ledger.csv", ledger)
    write_csv(OUT / "all_masks.csv", mask_rows)
    write_csv(OUT / "all_results.csv", all_results)
    if len(score_rows) != 6480:
        raise RuntimeError("Expected 6480 per-matrix score records")
    write_csv(OUT / "score_inputs.csv", score_rows)
    infeasible = [r for r in all_results if r["status"] != "feasible"]
    write_csv(OUT / "infeasible_cells.csv", infeasible, fieldnames=list(all_results[0]))
    paired = []
    for seed in SEEDS:
        for distribution in DISTRIBUTIONS:
            for checkpoint in CHECKPOINTS:
                for client_id in range(5):
                    rows = [r for r in all_results if (int(r["seed"]), r["distribution"],
                            r["checkpoint"], int(r["client_id"])) ==
                            (seed, distribution, checkpoint, client_id)]
                    values = {r["method"]: r for r in rows}
                    for source in ("Local", "Proxy"):
                        matrix = values[f"{source}-Matrix"]
                        layer = values[f"{source}-Layer"]
                        good = matrix["status"] == "feasible" and layer["status"] == "feasible"
                        paired.append({"run_id": matrix["run_id"], "dataset": "cifar100",
                                       "method": source, "seed": seed,
                                       "metric": "matrix_minus_layer_validation_loss_reduction",
                                       "config_path": "scripts/run_matrix_granularity_p2.py",
                                       "raw_result_path": matrix["raw_result_path"],
                                       "distribution": distribution, "checkpoint": checkpoint,
                                       "client_id": client_id, "status": "paired" if good else "infeasible",
                                       "matrix_mask": matrix["mask"], "layer_mask": layer["mask"],
                                       "matrix_gain": matrix.get("val_loss_reduction", ""),
                                       "layer_gain": layer.get("val_loss_reduction", ""),
                                       "matrix_minus_layer": (float(matrix["val_loss_reduction"]) -
                                                              float(layer["val_loss_reduction"])) if good else ""})
    write_csv(OUT / "paired_client_differences.csv", paired)
    seed_rows = []
    summary = []
    for distribution in DISTRIBUTIONS:
        for checkpoint in CHECKPOINTS:
            for source in ("Local", "Proxy"):
                for seed in SEEDS:
                    group = [r for r in paired if (r["distribution"], r["checkpoint"],
                             r["method"], int(r["seed"])) ==
                             (distribution, checkpoint, source, seed)]
                    valid = [float(r["matrix_minus_layer"]) for r in group if r["status"] == "paired"]
                    seed_rows.append({"distribution": distribution, "checkpoint": checkpoint,
                                      "source": source, "seed": seed, "n_clients": len(valid),
                                      "n_infeasible_clients": 5 - len(valid),
                                      "mean_matrix_minus_layer": float(np.mean(valid)) if valid else ""})
                selected = [r for r in seed_rows if (r["distribution"], r["checkpoint"], r["source"]) ==
                            (distribution, checkpoint, source)]
                complete = all(r["n_clients"] == 5 for r in selected)
                means = [float(r["mean_matrix_minus_layer"]) for r in selected if r["n_clients"] == 5]
                mean, std = average_std(means) if means else (float("nan"), float("nan"))
                summary.append({"distribution": distribution, "checkpoint": checkpoint,
                                "source": source, "n_complete_seeds": len(means),
                                "status": "complete" if complete else "incomplete_due_to_infeasible",
                                "mean_matrix_minus_layer": mean if complete else "",
                                "std_across_seeds": std if complete else ""})
    write_csv(OUT / "paired_seed_differences.csv", seed_rows)
    write_csv(OUT / "paired_summary.csv", summary)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)
    names = {"iid": "IID", "dirichlet:1.0": "Dirichlet 1.0", "dirichlet:0.1": "Dirichlet 0.1"}
    for ax, distribution in zip(axes, DISTRIBUTIONS):
        for offset, source, color, marker in ((-0.08, "Local", "#087e8b", "o"),
                                               (0.08, "Proxy", "#914c91", "s")):
            xs, ys, errors = [], [], []
            for index, checkpoint in enumerate(CHECKPOINTS):
                row = next(r for r in summary if (r["distribution"], r["checkpoint"], r["source"]) ==
                           (distribution, checkpoint, source))
                if row["status"] == "complete":
                    xs.append(index + offset)
                    ys.append(float(row["mean_matrix_minus_layer"]))
                    errors.append(float(row["std_across_seeds"]))
            ax.errorbar(xs, ys, yerr=errors, fmt=marker, color=color, capsize=3,
                        markersize=6, linewidth=1.4, label=source)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(names[distribution])
        ax.set_xticks(range(2), ["Early", "Middle"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Matrix - Layer loss reduction")
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("P2: paired 20-step validation gain (independent y-scales)")
    fig.savefig(OUT / "experiment_P2.pdf")
    fig.savefig(OUT / "experiment_P2.png", dpi=220)
    plt.close(fig)
    reused = sum(r.get("result_source") == "reused_original_p_nonfirst" for r in all_results)
    unique_new = {tuple((r["run_id"], r["checkpoint"], r["client_id"], r["mask"]))
                  for r in all_results if r.get("result_source") == "p2_repaired_rng_or_new_layer"}
    lines = ["# P2 固定协议结果", "", f"共保留 {len(all_results)} 条方法记录；不可行记录 {len(infeasible)} 条。",
             f"复用原 P 非首配置的方法记录 {reused} 条；本次新训练去重后 {len(unique_new)} 个配置。", "",
             "正值表示矩阵级选择的 20 步验证损失下降大于完整层绑定；误差为先在每个种子内跨 5 客户端求均值、再跨 3 种子计算的样本标准差。", "",
             "| 划分 | 检查点 | 来源 | Matrix-Layer 均值±标准差 |", "|---|---|---|---:|"]
    for r in summary:
        value = (f"{r['mean_matrix_minus_layer']:+.5f} ± {r['std_across_seeds']:.5f}"
                 if r["status"] == "complete" else "不可行，见 infeasible_cells.csv")
        lines.append(f"| {r['distribution']} | {r['checkpoint']} | {r['source']} | {value} |")
    positive_seed_rows = sum(float(r["mean_matrix_minus_layer"]) > 0 for r in seed_rows)
    positive_client_rows = sum(float(r["matrix_minus_layer"]) > 0 for r in paired)
    lines += ["", f"逐种子配对均值：{positive_seed_rows}/{len(seed_rows)} 为正；"
              f"逐客户端配对：{positive_client_rows}/{len(paired)} 为正。"
              "负值也完整保存在 paired_seed_differences.csv 和 paired_client_differences.csv。",
              "图中各划分采用独立纵轴刻度，以便同时看清 IID 与 Dirichlet 0.1 的差值。"]
    lines += ["", "本实验保持 rank=8、12 个矩阵、相同评分来源/深度校准/归一化、原 P 的训练及验证划分；没有调参或删选负结果。",
              "跨划分比较不能单独归因于数据异构性，因为原 P 深度系数也随划分变化。"]
    (OUT / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P2 ANALYZED", len(all_results), "method records", len(infeasible), "infeasible")


if __name__ == "__main__":
    main()
