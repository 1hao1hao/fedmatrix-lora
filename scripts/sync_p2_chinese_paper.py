#!/usr/bin/env python3
"""Apply the translated P2 publication blocks to the Chinese reading copy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_assets/04_manuscript"
PAYLOAD = PAPER / "zh_payload"
STAMP = "% FM_P2_CHINESE_SYNC"


def replace_body(text: str, start: str, end: str, body: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Missing or ambiguous Chinese anchor: {start} -> {end}")
    first = text.index(start) + len(start)
    last = text.index(end, first)
    return text[:first] + "\n" + body.strip() + "\n\n" + text[last:]


def span(text: str, start: str, end: str) -> str:
    first = text.index(start)
    return text[first:text.index(end, first)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    paper = args.repo.resolve() / "paper_assets/04_manuscript"
    source = paper / "main_zh.tex"
    original = source.read_text(encoding="utf-8")
    if STAMP in original:
        raise RuntimeError("Chinese P2 synchronization has already been applied")

    def payload(name: str) -> str:
        return (paper / "zh_payload" / name).read_text(encoding="utf-8")

    text = replace_body(original, r"\begin{abstract}", r"\end{abstract}", payload("abstract.tex"))
    text = replace_body(text, r"\section{引言}", r"\section{相关工作}\label{Sec.relatedworks}",
                        payload("introduction.tex"))
    text = replace_body(text, r"\section{相关工作}\label{Sec.relatedworks}",
                        r"\section{系统模型与问题定义}\label{sec.model}", payload("related_work.tex"))
    text = replace_body(text, r"\subsection{框架概述}",
                        r"\subsection{客户端相关的矩阵画像}\label{sec:fm-profiling}", payload("overview.tex"))
    text = replace_body(text, r"\subsection{客户端相关的矩阵画像}\label{sec:fm-profiling}",
                        r"\subsection{资源可行的矩阵分配}\label{sec:fm-allocation}", payload("profiling.tex"))
    start = r"\subsection{评分可靠性}\label{sec:fm-score-study}"
    end = "\\FloatBarrier\n\\section{结论}\\label{Sec.conclusion}"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("Chinese controlled-experiment anchors changed")
    first = text.index(start)
    last = text.index(end, first)
    text = text[:first] + payload("controlled_experiments.tex").strip() + "\n\n" + text[last:]
    text = replace_body(text, r"\section{结论}\label{Sec.conclusion}", r"\appendices",
                        payload("conclusion.tex"))
    old_opening = "本节通过覆盖不同数据集与模型设置的广泛仿真实验，评估所提出的 \\method{} 的有效性。"
    new_opening = ("我们在主基准设置上评估完整框架，并利用受控研究区分选择粒度、评分可靠性与组成部分的作用。"
                   "资源效率比较采用文中所述的解析式客户端资源画像。")
    if text.count(old_opening) != 1:
        raise ValueError("Chinese evaluation opening changed")
    text = text.replace(old_opening, new_opening)
    bibliography = "\\clearpage\n\\begin{thebibliography}{99}"
    if text.count(bibliography) != 1:
        raise ValueError("Chinese appendix/bibliography boundary changed")
    text = text.replace(bibliography, payload("appendix_controls.tex").strip() +
                        "\n\n\\FloatBarrier\n\\begin{thebibliography}{99}")

    untouched = [
        (r"\section{系统模型与问题定义}\label{sec.model}",
         r"\section{FedMATRIX-LoRA 设计}\label{Sec.alg}"),
        (r"\subsection{资源可行的矩阵分配}\label{sec:fm-allocation}",
         r"\subsection{按矩阵聚合的收敛性}"),
        (r"\subsection{按矩阵聚合的收敛性}",
         r"\section{实验}\label{Sec.evaluation}"),
        (r"\section{按矩阵聚合的证明}", r"\section{完整的受控研究结果}"),
    ]
    for start, end in untouched:
        original_end = r"\clearpage" if end == r"\section{完整的受控研究结果}" else end
        if span(original, start, original_end).strip() != span(text, start, end).strip():
            raise ValueError(f"An unchanged Chinese block was modified: {start}")
    for label in ("fig:fm-motivation", "fig:overview", "eq:fim", "eq:depth", "eq:normalization",
                  "sec:fm-granularity-study", "sec:fm-score-study", "sec:fm-ablations",
                  "app:fm-p2-details", "app:fm-source-control"):
        if text.count("\\label{" + label + "}") != 1:
            raise ValueError(f"Missing or duplicate Chinese label: {label}")
    if r"\begin{picture}" in text or r"\FMtodo{" in text:
        raise ValueError("Legacy figure overlay or result placeholder remains")

    backup = paper / ".rewrite_backups" / ("main_zh_before_p2_" +
             hashlib.sha256(original.encode()).hexdigest()[:12] + ".tex")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    source.write_text(STAMP + "\n" + text, encoding="utf-8")
    print("Synchronized P2 Chinese manuscript:", source)
    print("Preserved Chinese system model, allocation, theory, Appendix A, main studies, and references")


if __name__ == "__main__":
    main()
