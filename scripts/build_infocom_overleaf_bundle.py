#!/usr/bin/env python3
"""Build a minimal, self-contained Overleaf bundle for the INFOCOM paper."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_assets/04_manuscript/infocom_2027"
OUT = PAPER / "overleaf_package"

FIGURES = {
    "paper_assets/05_figures/cifar100_matrix_position_motivation_1380280/matrix_position_fixed_budget.pdf":
        "figures/fig1_matrix_motivation.pdf",
    "paper_assets/05_figures/final/fig1_method_overview.pdf":
        "figures/fig2_method_overview.pdf",
    "paper_assets/04_manuscript/infocom_2027/figures/fig2_compact_main_matrix.pdf":
        "figures/fig3_main_results.pdf",
    "paper_assets/04_manuscript/infocom_2027/figures/fig3_compact_budget_curves.pdf":
        "figures/fig4_budget_curves.pdf",
    "paper_assets/04_manuscript/infocom_2027/figures/fig3_compact_time_curves.pdf":
        "figures/fig5_time_curves.pdf",
    "paper_assets/04_manuscript/infocom_2027/figures/fig4_compact_ablation_overhead.pdf":
        "figures/fig5_ablation_overhead.pdf",
}

LOCAL_FONT_BLOCK = r"""\setmainfont[
  Path=/share/home/202520143645/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre/,
  UprightFont=texgyretermes-regular.otf,
  BoldFont=texgyretermes-bold.otf,
  ItalicFont=texgyretermes-italic.otf,
  BoldItalicFont=texgyretermes-bolditalic.otf
]{TeX Gyre Termes}
\setCJKmainfont[
  Path=/share/home/202520143645/texlive/2026/texmf-dist/fonts/opentype/public/fandol/,
  UprightFont=FandolSong-Regular.otf,
  BoldFont=FandolSong-Bold.otf,
  ItalicFont=FandolKai-Regular.otf
]{FandolSong}"""

PORTABLE_FONT_BLOCK = r"""\setmainfont{texgyretermes-regular.otf}[
  BoldFont=texgyretermes-bold.otf,
  ItalicFont=texgyretermes-italic.otf,
  BoldItalicFont=texgyretermes-bolditalic.otf
]
\setCJKmainfont{FandolSong-Regular.otf}[
  BoldFont=FandolSong-Bold.otf,
  ItalicFont=FandolKai-Regular.otf
]"""


def prepare_source(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    for project_path, bundle_path in FIGURES.items():
        text = text.replace(project_path, bundle_path)
    if source.name.endswith("_zh.tex"):
        if LOCAL_FONT_BLOCK not in text:
            raise RuntimeError("Chinese source font block changed; update portable Overleaf rewrite")
        text = text.replace(LOCAL_FONT_BLOCK, PORTABLE_FONT_BLOCK)
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)

    prepare_source(PAPER / "infocom_main.tex", OUT / "main.tex")
    prepare_source(PAPER / "infocom_main_zh.tex", OUT / "main_zh.tex")
    shutil.copy2(PAPER / "IEEEtran.cls", OUT / "IEEEtran.cls")
    shutil.copy2(PAPER / "reference_doi_registry.csv", OUT / "reference_doi_registry.csv")
    shutil.copy2(PAPER / "reference_doi_registry.md", OUT / "reference_doi_registry.md")

    for project_path, bundle_path in FIGURES.items():
        shutil.copy2(ROOT / project_path, OUT / bundle_path)

    readme = """INFOCOM 2027 Overleaf bundle

Main submission source: main.tex (compile with pdfLaTeX)
Chinese reading source: main_zh.tex (compile with XeLaTeX)

The bibliography is embedded in each TeX source, so no .bib file is required.
reference_doi_registry.csv lists publication and arXiv DOIs for verification.
Set main.tex as the Overleaf Main document for the English submission draft.
Generated PDFs, auxiliary files, experiment logs, and raw results are intentionally excluded.
"""
    (OUT / "README.txt").write_text(readme, encoding="utf-8")

    archive = shutil.make_archive(str(PAPER / "infocom_2027_overleaf"), "zip", OUT)
    print(OUT.relative_to(ROOT))
    print(Path(archive).relative_to(ROOT))


if __name__ == "__main__":
    main()
