#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/paper_plots/build_infocom_figures.py

LATEX_ENGINE=pdflatex bash scripts/compile_full_paper.sh \
  paper_assets/04_manuscript/infocom_2027/infocom_main.tex \
  paper_assets/04_manuscript/infocom_2027/build

LATEX_ENGINE=xelatex bash scripts/compile_full_paper.sh \
  paper_assets/04_manuscript/infocom_2027/infocom_main_zh.tex \
  paper_assets/04_manuscript/infocom_2027/build_zh

cp paper_assets/04_manuscript/infocom_2027/build/infocom_main.pdf \
  paper_assets/04_manuscript/infocom_2027/infocom_main.pdf
cp paper_assets/04_manuscript/infocom_2027/build_zh/infocom_main_zh.pdf \
  paper_assets/04_manuscript/infocom_2027/infocom_main_zh.pdf

echo "Built INFOCOM English and Chinese PDFs."
