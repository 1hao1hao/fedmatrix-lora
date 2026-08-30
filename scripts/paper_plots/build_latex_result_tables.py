#!/usr/bin/env python3
"""Build compact LaTeX tables from final paper CSV ledgers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "paper_assets" / "06_tables" / "final"


METHOD_ORDER = [
    "Ours-FIM",
    "Fed-HeLLo",
    "MS",
    "LEGEND",
    "FedAvg-LoRA homogeneous",
    "Uniform allocation",
]

METHOD_SHORT = {
    "Ours-FIM": "FedMATRIX-LoRA",
    "Fed-HeLLo": "Fed-HeLLo",
    "MS": "MS",
    "LEGEND": "LEGEND",
    "FedAvg-LoRA homogeneous": "FedAvg",
    "Uniform allocation": "Uniform",
}


def esc(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def fmt_metric(v: str) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "--"


def setting_name(dataset: str, distribution: str) -> str:
    names = {
        "cifar100": "CIFAR100",
        "sst2": "SST-2",
        "qnli": "QNLI",
        "qqp": "QQP",
        "mnli": "MNLI",
        "samsum": "SAMSum",
    }
    dist = (
        distribution.replace("dirichlet:", "D")
        .replace("length_skew:", "Len-")
        .replace("iid", "IID")
    )
    return f"{names.get(dataset, dataset)} / {dist}"


def build_main_results() -> None:
    rows = list(csv.DictReader((TABLE_DIR / "table1_main_results.csv").open()))
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    metrics: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["dataset"], row["distribution"])
        grouped[key][row["method"]] = row
        metrics[key] = row["metric"]

    order = sorted(grouped, key=lambda x: (x[0], x[1]))
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\caption{Main results under resource-constrained federated LoRA fine-tuning. Each cell reports final/best primary metric. Classification tasks use Accuracy; SAMSum uses Rouge-L. Full traceability fields are provided in Appendix Table~\\ref{tab:app-full-main-grid}.}",
        "\\label{tab:main-results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}ll" + "c" * len(METHOD_ORDER) + "@{}}",
        "\\toprule",
        "Setting & Metric & " + " & ".join(METHOD_SHORT[m] for m in METHOD_ORDER) + " \\\\",
        "\\midrule",
    ]
    for key in order:
        vals = grouped[key]
        best = {}
        for method, row in vals.items():
            try:
                best[method] = float(row["best_metric"])
            except Exception:
                pass
        row_best = max(best.values()) if best else None
        cells = []
        for method in METHOD_ORDER:
            row = vals.get(method)
            if not row:
                cells.append("--")
                continue
            text = f"{fmt_metric(row['final_metric'])}/{fmt_metric(row['best_metric'])}"
            if row_best is not None and method in best and abs(best[method] - row_best) < 1e-12:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(f"{esc(setting_name(*key))} & {esc(metrics[key])} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
        "",
    ]
    (TABLE_DIR / "table1_main_results_compact.tex").write_text("\n".join(lines), encoding="utf-8")


def build_ablation() -> None:
    rows = list(csv.DictReader((TABLE_DIR / "table2_ablation_pairs.csv").open()))
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Core ablation study. Metric deltas are computed as full method minus ablated variant using the best primary metric; time deltas are ablated cumulative time minus full cumulative time.}",
        "\\label{tab:core-ablations}",
        "\\begin{tabular}{@{}llllrrr@{}}",
        "\\toprule",
        "ID & Dataset/split & Full & Ablated & Metric $\\Delta$ & Final $\\Delta$ & Time $\\Delta$ (s) \\\\",
        "\\midrule",
    ]
    method_display = {
        "Ours-FIM": "FedMATRIX-LoRA",
        "Ours-FIM MCKP": "FedMATRIX-LoRA MCKP",
        "Ours-FIM w/o depth prior": "FedMATRIX-LoRA w/o depth prior",
        "Ours-FIM additive-memory DP": "FedMATRIX-LoRA additive-memory DP",
        "Ours-FIM lambda_prime=0": "FedMATRIX-LoRA lambda_prime=0",
    }
    for row in rows:
        setting = setting_name(row["dataset"], row["distribution"])
        full_method = method_display.get(row["full_method"], row["full_method"])
        ablated_method = method_display.get(row["ablated_method"], row["ablated_method"])
        lines.append(
            f"{esc(row['component'])} & {esc(setting)} & {esc(full_method)} & "
            f"{esc(ablated_method)} & {float(row['metric_delta']):.4f} & "
            f"{float(row['final_metric_delta']):.4f} & {float(row['time_delta']):.1f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    (TABLE_DIR / "table2_ablation_pairs_compact.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    build_main_results()
    build_ablation()
    print(TABLE_DIR / "table1_main_results_compact.tex")
    print(TABLE_DIR / "table2_ablation_pairs_compact.tex")


if __name__ == "__main__":
    main()
