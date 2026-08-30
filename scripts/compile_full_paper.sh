#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX="${1:-paper_assets/04_manuscript/full_paper/full_paper_6_17.tex}"
OUT="${2:-paper_assets/04_manuscript/full_paper/build}"
TEXLIVE_BIN="/share/home/202520143645/texlive/2026/bin/x86_64-linux"
TEX_DIR="$(dirname "$TEX")"
LATEX_ENGINE="${LATEX_ENGINE:-xelatex}"

cd "$ROOT"

if [ -d "$TEXLIVE_BIN" ]; then
  export PATH="$TEXLIVE_BIN:$PATH"
fi

if ! command -v "$LATEX_ENGINE" >/dev/null 2>&1; then
  echo "ERROR: $LATEX_ENGINE is not installed or not on PATH." >&2
  echo "Install a TeX distribution with $LATEX_ENGINE support, then rerun this recipe." >&2
  exit 127
fi

mkdir -p "$OUT"
mkdir -p "$OUT/texlive-cache/texmf-var" "$OUT/texlive-cache/texmf-config"

export TEXMFVAR="$ROOT/$OUT/texlive-cache/texmf-var"
export TEXMFCONFIG="$ROOT/$OUT/texlive-cache/texmf-config"
export TEXINPUTS="$ROOT/$TEX_DIR:${TEXINPUTS:-}"

"$LATEX_ENGINE" -synctex=1 -interaction=nonstopmode -file-line-error -output-directory="$OUT" "$TEX"
"$LATEX_ENGINE" -synctex=1 -interaction=nonstopmode -file-line-error -output-directory="$OUT" "$TEX"
