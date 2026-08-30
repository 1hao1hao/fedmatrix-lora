#!/usr/bin/env python3
"""Review completed CIFAR100 generic-path budget scan job 1384551."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "evaluation_results/cifar100"
MANIFEST = ROOT / "experiments/rerun_cifar100_generic_budget_scan_manifest.csv"
SUMMARY = ROOT / "paper_assets/02_experiments/cifar100_generic_budget_scan_1384551_summary.csv"
DECISION = ROOT / "paper_assets/02_experiments/cifar100_generic_budget_scan_1384551_decision.md"
FIG_DIR = ROOT / "paper_assets/05_figures/cifar100_generic_budget_scan_1384551"

METHODS = ["FedAvg", "Uniform", "FedMATRIX"]
RAW_TO_METHOD = {
    "FedAvg_LoRA_homogeneous": "FedAvg",
    "Uniform_allocation": "Uniform",
    "Random_FIM": "FedMATRIX",
}
COLORS = {"FedAvg": "#777777", "Uniform": "#4c78a8", "FedMATRIX": "#e66101"}
DIST_LABEL = {"iid": "IID", "dirichlet:1.0": "Dirichlet-1.0", "dirichlet:0.1": "Dirichlet-0.1"}
BUDGETS = {
    "iid": [4800, 6000, 7200],
    "dirichlet:1.0": [4800, 6000, 7200],
    "dirichlet:0.1": [7200, 9600, 12000],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rolling_mean(values: list[float], width: int = 5) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - width + 1)
        result.append(sum(values[start:index + 1]) / (index - start + 1))
    return result


def discover() -> list[dict[str, object]]:
    manifest_rows = read_csv(MANIFEST)
    records = []
    for spec in manifest_rows:
        candidates = sorted(RESULT_ROOT.glob(f"{spec['experiment_id']}__Dist-*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one result directory for {spec['experiment_id']}, found {len(candidates)}")
        run_dir = candidates[0]
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        raw_method = spec["method_raw"]
        eval_path = run_dir / f"eval_{raw_method}.csv"
        trace = read_csv(eval_path)
        accuracy = [float(row["Accuracy"]) for row in trace]
        loss = [float(row["Loss"]) for row in trace]
        best = max(accuracy)
        best_round = accuracy.index(best) + 1
        first20 = sum(accuracy[:20]) / 20
        last20 = sum(accuracy[-20:]) / 20
        records.append({
            "job_id": "1384551",
            "dataset": "cifar100",
            "distribution": metadata["distribution"],
            "total_budget": int(metadata["total_budget"]),
            "method": RAW_TO_METHOD[raw_method],
            "method_raw": raw_method,
            "seed": int(metadata["seed"]),
            "metric": "Accuracy",
            "rounds": len(trace),
            "best_metric": best,
            "best_round": best_round,
            "final_metric": accuracy[-1],
            "first20_mean": first20,
            "last20_mean": last20,
            "peak_to_final_drop": best - accuracy[-1],
            "initial_loss": loss[0],
            "final_loss": loss[-1],
            "final_cumulative_time": float(trace[-1]["Cumulative_Time"]),
            "final_cumulative_budget": float(trace[-1]["Cumulative_Budget"]),
            "run_id": metadata["run_id"],
            "config_path": str(MANIFEST.relative_to(ROOT)),
            "raw_result_path": str(eval_path.relative_to(ROOT)),
            "status": "complete" if len(trace) == int(metadata["max_rounds"]) else "incomplete",
            "trace": trace,
        })
    return records


def write_summary(records: list[dict[str, object]]) -> None:
    fields = [
        "job_id", "dataset", "distribution", "total_budget", "method", "method_raw", "seed",
        "metric", "rounds", "best_metric", "best_round", "final_metric", "first20_mean",
        "last20_mean", "peak_to_final_drop", "initial_loss", "final_loss",
        "final_cumulative_time", "final_cumulative_budget", "run_id", "config_path",
        "raw_result_path", "status", "decision", "reason",
    ]
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["distribution"], record["total_budget"])].append(record)

    for group in grouped.values():
        ours = next(row for row in group if row["method"] == "FedMATRIX")
        baseline_best = max(float(row["best_metric"]) for row in group if row["method"] != "FedMATRIX")
        baseline_final = max(float(row["final_metric"]) for row in group if row["method"] != "FedMATRIX")
        metric_win = float(ours["best_metric"]) > baseline_best and float(ours["final_metric"]) >= baseline_final
        for row in group:
            row["decision"] = "diagnostic_only"
            if metric_win:
                row["reason"] = "FedMATRIX wins this grid point, but the full generic-path scan fails Dirichlet-1.0 and does not replace stronger current final evidence."
            else:
                row["reason"] = "FedMATRIX does not beat both baselines in best and final Accuracy at this grid point."

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda row: (
            list(BUDGETS).index(row["distribution"]), row["total_budget"], METHODS.index(row["method"])
        )):
            writer.writerow({key: record[key] for key in fields})


def plot_round_curves(records: list[dict[str, object]]) -> None:
    lookup = {(row["distribution"], row["total_budget"], row["method"]): row for row in records}
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 7.1), sharex=True)
    for row_index, distribution in enumerate(BUDGETS):
        for col_index, budget in enumerate(BUDGETS[distribution]):
            ax = axes[row_index, col_index]
            all_values = []
            for method in METHODS:
                record = lookup[(distribution, budget, method)]
                values = [float(item["Accuracy"]) for item in record["trace"]]
                rounds = list(range(1, len(values) + 1))
                ax.plot(rounds, values, color=COLORS[method], alpha=.20, linewidth=.55)
                ax.plot(rounds, rolling_mean(values), color=COLORS[method], linewidth=1.45, label=method)
                all_values.extend(values)
            lower = max(0.0, min(all_values) - .035)
            upper = min(1.0, max(all_values) + .025)
            ax.set_ylim(lower, upper)
            ax.set_title(f"{DIST_LABEL[distribution]}, budget={budget}", fontsize=9)
            ax.grid(alpha=.22, linewidth=.45)
            ax.tick_params(labelsize=7)
            if col_index == 0:
                ax.set_ylabel("Accuracy", fontsize=8)
            if row_index == 2:
                ax.set_xlabel("Round", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(.5, .995))
    fig.tight_layout(rect=(0, 0, 1, .955), pad=.8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"cifar100_generic_budget_scan_1384551_round_curves.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_budget_summary(records: list[dict[str, object]]) -> None:
    lookup = {(row["distribution"], row["total_budget"], row["method"]): row for row in records}
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.9))
    for col_index, distribution in enumerate(BUDGETS):
        budgets = BUDGETS[distribution]
        for row_index, metric_key in enumerate(("best_metric", "final_metric")):
            ax = axes[row_index, col_index]
            for method in METHODS:
                values = [float(lookup[(distribution, budget, method)][metric_key]) for budget in budgets]
                ax.plot(budgets, values, marker="o", markersize=4, linewidth=1.5,
                        color=COLORS[method], label=method)
            ax.set_title(DIST_LABEL[distribution], fontsize=9)
            ax.set_ylabel("Best Accuracy" if row_index == 0 else "Final Accuracy", fontsize=8)
            ax.set_xlabel("Total budget", fontsize=8)
            ax.grid(alpha=.22, linewidth=.45)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94), pad=.8)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"cifar100_generic_budget_scan_1384551_budget_summary.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_decision(records: list[dict[str, object]]) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["distribution"], record["total_budget"])].append(record)

    lines = [
        "# CIFAR100 通用路径预算扫描 1384551 审查结论",
        "",
        "## 总结",
        "",
        "- Slurm 数组任务 `1384551` 的 27 个子任务均正常结束，均生成 180 轮 CSV、metadata 和轨迹文件，未检出 traceback、OOM 或非零退出。",
        "- 整批实验未达到预注册采纳条件：FedMATRIX 在 Dirichlet-1.0 的三个预算点均未取得最高 best Accuracy。",
        "- 本批结果仅作为通用路径和预算敏感性诊断，不写入 `final_result_overrides.csv`，也不替换当前最终 CIFAR100 数据。",
        "",
        "## 逐设置结果",
        "",
        "| 分布 | 总预算 | FedAvg best/final | Uniform best/final | FedMATRIX best/final | Ours 相对最强基线 best/final | 决定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for distribution in BUDGETS:
        for budget in BUDGETS[distribution]:
            group = grouped[(distribution, budget)]
            by_method = {row["method"]: row for row in group}
            ours = by_method["FedMATRIX"]
            base_best = max(float(by_method[name]["best_metric"]) for name in ("FedAvg", "Uniform"))
            base_final = max(float(by_method[name]["final_metric"]) for name in ("FedAvg", "Uniform"))
            metric_win = float(ours["best_metric"]) > base_best and float(ours["final_metric"]) >= base_final
            lines.append(
                f"| {DIST_LABEL[distribution]} | {budget} | "
                f"{by_method['FedAvg']['best_metric']:.3f}/{by_method['FedAvg']['final_metric']:.3f} | "
                f"{by_method['Uniform']['best_metric']:.3f}/{by_method['Uniform']['final_metric']:.3f} | "
                f"{ours['best_metric']:.3f}/{ours['final_metric']:.3f} | "
                f"{float(ours['best_metric']) - base_best:+.3f}/{float(ours['final_metric']) - base_final:+.3f} | "
                f"{'局部胜出但不采纳' if metric_win else '未达预期'} |"
            )

    lines.extend([
        "",
        "## 与当前最终数据比较",
        "",
        "| 分布 | 当前最终 Ours best/final | 本次通用路径最强 Ours best/final | 结论 |",
        "|---|---:|---:|---|",
        "| IID | 0.900/0.894 | 0.894/0.888 (B7200) | 新结果更低，不替换 |",
        "| Dirichlet-1.0 | 0.896/0.892 | 0.872/0.866 (B7200) | 低于当前结果且 best 低于 Uniform 0.878 |",
        "| Dirichlet-0.1 | 0.810/0.780 | 0.804/0.788 (B9600) 或 0.790/0.772 (B12000) | best 低于当前结果；不替换 |",
        "",
        "## 失败原因",
        "",
        "通用路径的 FIM 排名在 CIFAR100 上几乎退化为 V-only 选择。累计选择统计如下：",
        "",
        "- IID/B7200：FedMATRIX 的 7,176 次选择全部为 V；Uniform 同时覆盖 K 和 V。",
        "- Dirichlet-1.0/B7200：FedMATRIX 选择 7,172 次 V、4 次 K，且平均深度为 0.595；Uniform 的平均深度为 0.493，并稳定覆盖浅、中、深层。",
        "- Dirichlet-0.1/B12000：FedMATRIX 选择 11,915 次 V、32 次 K，平均深度升至 0.753。高预算下它能依靠更多 V 矩阵胜出，但并未形成通用的 Q/K/V 覆盖。",
        "",
        "这说明仅调整预算不能解决 Dirichlet-1.0：预算增加只会让相同的 V 偏好扩展到更多中深层矩阵。当前最终方案中的 CIFAR100 结构先验恰好补充了浅层覆盖与 Q/K/V 平衡，因此现阶段不能删除该先验而保持全部 CIFAR100 结论。",
        "",
        "## 可追溯文件",
        "",
        f"- 配置：`{MANIFEST.relative_to(ROOT)}`",
        f"- 完整 27 行汇总（含 run_id、seed、metric、config 和 raw path）：`{SUMMARY.relative_to(ROOT)}`",
        f"- 轮次曲线：`{(FIG_DIR / 'cifar100_generic_budget_scan_1384551_round_curves.pdf').relative_to(ROOT)}`",
        f"- 预算扫描摘要：`{(FIG_DIR / 'cifar100_generic_budget_scan_1384551_budget_summary.pdf').relative_to(ROOT)}`",
    ])
    DECISION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = discover()
    if len(records) != 27:
        raise RuntimeError(f"Expected 27 records, found {len(records)}")
    write_summary(records)
    plot_round_curves(records)
    plot_budget_summary(records)
    write_decision(records)
    print(SUMMARY.relative_to(ROOT))
    print(DECISION.relative_to(ROOT))
    print(FIG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
