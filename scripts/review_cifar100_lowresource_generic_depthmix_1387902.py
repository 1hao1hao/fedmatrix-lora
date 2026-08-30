#!/usr/bin/env python3
"""Review CIFAR100 low-resource generic depth-mix Slurm job 1387902."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "evaluation_results/cifar100"
MANIFEST = ROOT / "experiments/rerun_cifar100_lowresource_generic_depthmix_manifest.csv"
SUMMARY = ROOT / "paper_assets/02_experiments/cifar100_lowresource_generic_depthmix_1387902_summary.csv"
DECISION = ROOT / "paper_assets/02_experiments/cifar100_lowresource_generic_depthmix_1387902_decision.md"
FIG_DIR = ROOT / "paper_assets/05_figures/cifar100_lowresource_generic_depthmix_1387902"
TRACE_ROOT = ROOT / "paper_assets/02_experiments/traces/cifar100_lowresource_generic_depthmix"

METHOD_ORDER = ["FedAvg", "Uniform", "FedMATRIX"]
RAW_TO_METHOD = {
    "FedAvg_LoRA_homogeneous": "FedAvg",
    "Uniform_allocation": "Uniform",
    "Random_FIM": "FedMATRIX",
}
COLORS = {"FedAvg": "#777777", "Uniform": "#4c78a8", "FedMATRIX": "#e66101"}
DIST_ORDER = ["iid", "dirichlet:1.0", "dirichlet:0.1"]
DIST_LABEL = {"iid": "IID", "dirichlet:1.0": "Dirichlet-1.0", "dirichlet:0.1": "Dirichlet-0.1"}
TRACE_TOKEN = {"iid": "iid_b5400", "dirichlet:1.0": "dirichlet_1p0_b6300", "dirichlet:0.1": "dirichlet_0p1_b9000"}
CURRENT_FINAL = {
    "iid": (0.900, 0.894),
    "dirichlet:1.0": (0.896, 0.892),
    "dirichlet:0.1": (0.810, 0.780),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rolling_mean(values: list[float], width: int = 5) -> list[float]:
    return [sum(values[max(0, i - width + 1):i + 1]) / min(i + 1, width) for i in range(len(values))]


def discover() -> list[dict[str, object]]:
    records = []
    for spec in read_csv(MANIFEST):
        candidates = sorted(RESULT_ROOT.glob(f"{spec['experiment_id']}__Dist-*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one run for {spec['experiment_id']}; found {len(candidates)}")
        run_dir = candidates[0]
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        raw_method = spec["method_raw"]
        eval_path = run_dir / f"eval_{raw_method}.csv"
        rows = read_csv(eval_path)
        accuracy = [float(row["Accuracy"]) for row in rows]
        loss = [float(row["Loss"]) for row in rows]
        best = max(accuracy)
        records.append({
            "job_id": "1387902",
            "dataset": "cifar100",
            "distribution": metadata["distribution"],
            "total_budget": int(metadata["total_budget"]),
            "average_round_budget": int(metadata["total_budget"]) / int(metadata["max_rounds"]),
            "method": RAW_TO_METHOD[raw_method],
            "method_raw": raw_method,
            "seed": int(metadata["seed"]),
            "metric": "Accuracy",
            "rounds": len(rows),
            "best_metric": best,
            "best_round": accuracy.index(best) + 1,
            "final_metric": accuracy[-1],
            "last10_mean": sum(accuracy[-10:]) / 10,
            "peak_to_final_drop": best - accuracy[-1],
            "initial_loss": loss[0],
            "final_loss": loss[-1],
            "final_cumulative_time": float(rows[-1]["Cumulative_Time"]),
            "final_cumulative_budget": float(rows[-1]["Cumulative_Budget"]),
            "run_id": metadata["run_id"],
            "config_path": str(MANIFEST.relative_to(ROOT)),
            "raw_result_path": str(eval_path.relative_to(ROOT)),
            "status": "complete" if len(rows) == int(metadata["max_rounds"]) else "incomplete",
            "rows": rows,
        })
    return records


def audit_selection(records: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    audit = {}
    for record in records:
        distribution = str(record["distribution"])
        method = str(record["method"])
        raw_method = str(record["method_raw"])
        trace_path = TRACE_ROOT / TRACE_TOKEN[distribution] / f"selection_trace_{raw_method}.csv"
        rows = read_csv(trace_path)
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        module_types = Counter()
        depths = []
        for row in rows:
            grouped[(row["round"], row["client_id"])].append(int(row["matrix_index"]))
            name = row["module_name"].lower()
            if "query" in name or "q_proj" in name:
                module_types["Q"] += 1
            elif "key" in name or "k_proj" in name:
                module_types["K"] += 1
            elif "value" in name or "v_proj" in name:
                module_types["V"] += 1
            else:
                module_types["Other"] += 1
            depths.append(float(row["depth_norm"]))
        signatures = [tuple(sorted(indices)) for indices in grouped.values()]
        score_trace = TRACE_ROOT / TRACE_TOKEN[distribution] / f"score_trace_{raw_method}.csv"
        audit[(distribution, method)] = {
            "trace_path": str(trace_path.relative_to(ROOT)),
            "selected_rows": len(rows),
            "client_round_configs": len(signatures),
            "unique_matrix_sets": len(set(signatures)),
            "mean_selected_count": len(rows) / max(1, len(signatures)),
            "mean_depth": sum(depths) / max(1, len(depths)),
            "q_count": module_types["Q"],
            "k_count": module_types["K"],
            "v_count": module_types["V"],
            "other_count": module_types["Other"],
            "score_trace_present": score_trace.exists(),
        }
    return audit


def write_summary(records: list[dict[str, object]]) -> None:
    fields = [
        "job_id", "dataset", "distribution", "total_budget", "average_round_budget",
        "method", "method_raw", "seed", "metric", "rounds", "best_metric", "best_round",
        "final_metric", "last10_mean", "peak_to_final_drop", "initial_loss", "final_loss",
        "final_cumulative_time", "final_cumulative_budget", "run_id", "config_path",
        "raw_result_path", "status", "decision", "reason",
    ]
    grouped = defaultdict(list)
    for row in records:
        grouped[row["distribution"]].append(row)
    for group in grouped.values():
        by_method = {str(row["method"]): row for row in group}
        baseline_best = max(float(by_method[name]["best_metric"]) for name in ("FedAvg", "Uniform"))
        baseline_final = max(float(by_method[name]["final_metric"]) for name in ("FedAvg", "Uniform"))
        ours = by_method["FedMATRIX"]
        win = float(ours["best_metric"]) > baseline_best and float(ours["final_metric"]) >= baseline_final
        for row in group:
            row["decision"] = "diagnostic_only"
            row["reason"] = (
                "FedMATRIX wins this low-resource cell, but it is weaker than the currently adopted CIFAR100 result."
                if win else
                "FedMATRIX does not beat the strongest basic baseline in both best and final Accuracy."
            )
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(records, key=lambda item: (
            DIST_ORDER.index(str(item["distribution"])), METHOD_ORDER.index(str(item["method"]))
        )):
            writer.writerow({field: row[field] for field in fields})


def plot_curves(records: list[dict[str, object]]) -> None:
    lookup = {(str(row["distribution"]), str(row["method"])): row for row in records}
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 7.0))
    x_fields = [("Round", "Round"), ("Cumulative_Budget", "Cumulative budget"), ("Cumulative_Time", "Modeled elapsed time (s)")]
    for row_index, distribution in enumerate(DIST_ORDER):
        all_accuracy = []
        for method in METHOD_ORDER:
            all_accuracy.extend(float(item["Accuracy"]) for item in lookup[(distribution, method)]["rows"])
        lower = max(0.0, min(all_accuracy) - 0.035)
        upper = min(1.0, max(all_accuracy) + 0.025)
        for col_index, (field, xlabel) in enumerate(x_fields):
            ax = axes[row_index, col_index]
            for method in METHOD_ORDER:
                trace = lookup[(distribution, method)]["rows"]
                x = [float(item[field]) for item in trace]
                y = [float(item["Accuracy"]) for item in trace]
                ax.plot(x, y, color=COLORS[method], alpha=0.18, linewidth=0.55)
                ax.plot(x, rolling_mean(y), color=COLORS[method], linewidth=1.4, label=method)
            ax.set_ylim(lower, upper)
            ax.grid(alpha=0.22, linewidth=0.45)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(xlabel, fontsize=8)
            if col_index == 0:
                ax.set_ylabel(f"{DIST_LABEL[distribution]}\nAccuracy", fontsize=8)
            if row_index == 0:
                ax.set_title(xlabel, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"cifar100_lowresource_generic_depthmix_1387902_curves.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_decision(records: list[dict[str, object]], audit: dict[tuple[str, str], dict[str, object]]) -> None:
    grouped = defaultdict(list)
    for row in records:
        grouped[row["distribution"]].append(row)
    lines = [
        "# CIFAR100 低资源通用路径 1387902 审查结论",
        "",
        "## 最终判断",
        "",
        "- 数组任务 `1387902` 的 9 个子任务均以 `0:0` 正常结束，均包含 180 轮结果、metadata、overhead 和 selection trace；日志未检出 traceback、OOM 或运行时错误。",
        "- FedAvg 与 Uniform 已确认是基础版本：两者先均分每轮预算；FedAvg 在 CIFAR100 上使用可复现的随机矩阵选择，Uniform 使用固定等间隔矩阵选择。它们不计算 FIM，也不调用客户端配置 DP 或服务器 MCKP。",
        "- 整批实验未达到采纳条件：FedMATRIX 仅在 Dirichlet-1.0 胜出，在 IID 和 Dirichlet-0.1 均低于 Uniform，且三个 FedMATRIX 结果都弱于当前已采纳 CIFAR100 数据。",
        "- 本批只作为低资源通用评分诊断；不修改 `final_result_overrides.csv`、curated final data 或最终论文图表。",
        "",
        "## 准确率结果",
        "",
        "| 分布 | 总预算（平均/轮） | FedAvg best/final | Uniform best/final | FedMATRIX best/final | Ours 相对最强基线 best/final | 决定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for distribution in DIST_ORDER:
        by_method = {str(row["method"]): row for row in grouped[distribution]}
        ours = by_method["FedMATRIX"]
        base_best = max(float(by_method[name]["best_metric"]) for name in ("FedAvg", "Uniform"))
        base_final = max(float(by_method[name]["final_metric"]) for name in ("FedAvg", "Uniform"))
        win = float(ours["best_metric"]) > base_best and float(ours["final_metric"]) >= base_final
        lines.append(
            f"| {DIST_LABEL[distribution]} | {ours['total_budget']} ({ours['average_round_budget']:.1f}) | "
            f"{by_method['FedAvg']['best_metric']:.3f}/{by_method['FedAvg']['final_metric']:.3f} | "
            f"{by_method['Uniform']['best_metric']:.3f}/{by_method['Uniform']['final_metric']:.3f} | "
            f"{ours['best_metric']:.3f}/{ours['final_metric']:.3f} | "
            f"{float(ours['best_metric']) - base_best:+.3f}/{float(ours['final_metric']) - base_final:+.3f} | "
            f"{'局部胜出，但不替换现有最终数据' if win else '未达预期'} |"
        )
    lines.extend([
        "",
        "## 与当前最终 Ours 比较",
        "",
        "| 分布 | 当前最终 best/final | 本次 best/final | 变化 |",
        "|---|---:|---:|---:|",
    ])
    for distribution in DIST_ORDER:
        ours = next(row for row in grouped[distribution] if row["method"] == "FedMATRIX")
        current_best, current_final = CURRENT_FINAL[distribution]
        lines.append(
            f"| {DIST_LABEL[distribution]} | {current_best:.3f}/{current_final:.3f} | "
            f"{ours['best_metric']:.3f}/{ours['final_metric']:.3f} | "
            f"{float(ours['best_metric']) - current_best:+.3f}/{float(ours['final_metric']) - current_final:+.3f} |"
        )
    lines.extend([
        "",
        "## 基线基础性审计",
        "",
        "| 分布 | 方法 | FIM score trace | 客户端-轮次配置数 | 不同矩阵集合数 | 平均选中矩阵数 | 平均深度 | Q/K/V 次数 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for distribution in DIST_ORDER:
        for method in METHOD_ORDER:
            item = audit[(distribution, method)]
            lines.append(
                f"| {DIST_LABEL[distribution]} | {method} | {'有' if item['score_trace_present'] else '无'} | "
                f"{item['client_round_configs']} | {item['unique_matrix_sets']} | "
                f"{item['mean_selected_count']:.2f} | {item['mean_depth']:.3f} | "
                f"{item['q_count']}/{item['k_count']}/{item['v_count']} |"
            )
    lines.extend([
        "",
        "FedAvg 和 Uniform 都没有 score trace，证明运行时未进入重要性画像。FedAvg 的矩阵集合随客户端和轮次大量变化，符合固定种子的随机选择；Uniform 只出现少量固定集合，差异来自客户端显存允许的矩阵数量，符合等间隔选择。",
        "",
        "## 为什么仍未达到预期",
        "",
        "弱深度混合使 FedMATRIX 在 Dirichlet-1.0 从此前扫描中的落后转为领先，但选择轨迹表明它仍几乎完全选择 V，平均深度 0.591 也与此前通用路径接近，因此不能把提升解释为结构偏置已经消失。该配置在 IID 和强异构 D0.1 上反而低于 Uniform，尤其 D0.1 最佳 Accuracy 低 0.080。由此可见，降低预算可以放大位置调度的重要性，但固定的 85/15 深度混合没有形成跨异构程度稳定的选择机制。",
        "",
        "本次结果也不支持直接删除当前 CIFAR100 视觉结构项：D1 表明通用弱先验有潜力，但 IID 和 D0.1 尚不能覆盖现有最终结论。若继续追求完全通用路径，应先研究根据分数置信度或异构程度自适应选择 FIM/深度混合权重，而不是继续单向降低预算。",
        "",
        "## 可追溯文件",
        "",
        f"- 配置：`{MANIFEST.relative_to(ROOT)}`",
        f"- 完整 9 行结果账表：`{SUMMARY.relative_to(ROOT)}`",
        f"- 审查曲线：`{(FIG_DIR / 'cifar100_lowresource_generic_depthmix_1387902_curves.pdf').relative_to(ROOT)}`",
        "- 原始结果：详见汇总 CSV 的 `run_id`、`config_path` 和 `raw_result_path` 字段。",
        "- 选择轨迹：`paper_assets/02_experiments/traces/cifar100_lowresource_generic_depthmix/`。",
    ])
    DECISION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = discover()
    if len(records) != 9:
        raise RuntimeError(f"Expected 9 records; found {len(records)}")
    audit = audit_selection(records)
    write_summary(records)
    plot_curves(records)
    write_decision(records, audit)
    print(SUMMARY.relative_to(ROOT))
    print(DECISION.relative_to(ROOT))
    print(FIG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
