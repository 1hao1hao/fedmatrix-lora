#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PAPER=paper_assets/04_manuscript

build_one() {
  local language="$1" source engine output
  if [[ "$language" == zh ]]; then
    source=main_zh
    engine=xelatex
  else
    source=main
    engine=pdflatex
  fi
  output="$PAPER/build_${source}"
  LATEX_ENGINE="$engine" bash scripts/compile_full_paper.sh "$PAPER/$source.tex" "$output"
  cp "$output/$source.pdf" "$PAPER/$source.pdf"
}

case "${1:-en}" in
  en) build_one en ;;
  zh) build_one zh ;;
  all) build_one en; build_one zh ;;
  *) echo "Usage: $0 [en|zh|all]" >&2; exit 2 ;;
esac
