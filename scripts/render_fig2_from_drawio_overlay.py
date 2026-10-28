#!/usr/bin/env python3
"""Apply revised draw.io stage headings to the original vector architecture art.

The original diagram remains intact. The overlay is generated from a copy of its
draw.io XML and placed on the old vector PDF, preserving all icons and arrows.
"""

import html
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "paper_assets/05_figures/fig_structure"
XML = ASSETS / "xml/fig2_paper_architecture_revised.drawio"
BASE = ASSETS / "preview/fig2_text_aligned_final.pdf"
BUILD = ASSETS / "preview/build_fig2_paper_architecture"
OUTPUT = ROOT / "paper_assets/04_manuscript/figures/fig2_method_overview_repaired.pdf"
PDFLATEX = Path("/share/home/202520143645/texlive/2026/bin/x86_64-linux/pdflatex")


def title_text(value):
    value = html.unescape(value)
    value = re.sub(r"<br\s*/?>", lambda _: r"\\", value)
    value = re.sub(r"</?b>", "", value)
    return value.replace("&", r"\&")


def main():
    root = ET.parse(XML).getroot()
    cells = {cell.get("id"): cell for cell in root.iter("mxCell")}
    stages = [
        ("clients-title", "clients-panel"),
        ("local-title", "local-panel"),
        ("server-title", "server-panel"),
        ("exec-title", "exec-panel"),
    ]
    layers = []
    for index, (title_id, panel_id) in enumerate(stages, 1):
        cell = cells[title_id]
        panel = cells[panel_id]
        geometry = cell.find("mxGeometry")
        # The preserved draw.io PDF is tightly cropped relative to its XML canvas.
        x, y = float(geometry.get("x")) - 24, float(geometry.get("y")) - 55
        w = float(geometry.get("width"))
        font_color = re.search(r"fontColor=#([0-9A-Fa-f]{6})", cell.get("style")).group(1)
        bg_color = re.search(r"fillColor=#([0-9A-Fa-f]{6})", panel.get("style")).group(1)
        font_size = int(re.search(r"fontSize=(\d+)", cell.get("style")).group(1))
        text = title_text(cell.get("value"))
        patch_right = x + w + (-15 if index == 3 else 5)
        layers.extend([
            rf"\definecolor{{stagebg{index}}}{{HTML}}{{{bg_color}}}",
            rf"\definecolor{{stagetext{index}}}{{HTML}}{{{font_color}}}",
            rf"\fill[stagebg{index}] ({x-3:g},{y-4:g}) rectangle ({patch_right:g},{y+49:g});",
        ])
        if index == 1:
            layers.append(rf"\fill[stagebg1] ({patch_right:g},{y+5:g}) rectangle ({x+w+14:g},{y+49:g});")
        layers.extend([
            rf"\node[anchor=north west,inner sep=0,align=left,text=stagetext{index},"
            rf"font=\sffamily\bfseries\fontsize{{{font_size}}}{{{font_size+1}}}\selectfont] "
            rf"at ({x:g},{y+1:g}) {{{text}}};",
        ])

    BUILD.mkdir(parents=True, exist_ok=True)
    tex = BUILD / "fig2_paper_architecture_revised.tex"
    definitions = "\n".join(line for line in layers if line.startswith(r"\definecolor"))
    drawing = "\n".join(line for line in layers if not line.startswith(r"\definecolor"))
    tex.write_text(
        "\n".join([
            r"\documentclass[border=0pt]{standalone}",
            r"\usepackage{tikz}",
            r"\usepackage{graphicx}",
            definitions,
            r"\begin{document}",
            r"\begin{tikzpicture}[x=1bp,y=-1bp]",
            rf"\node[anchor=north west,inner sep=0] at (0,0) {{\includegraphics[width=1611bp]{{{BASE}}}}};",
            drawing,
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]), encoding="utf-8")
    env = os.environ.copy()
    env["TEXMFVAR"] = str(BUILD / "texmf-var")
    env["TEXMFCONFIG"] = str(BUILD / "texmf-config")
    for key in ("TEXMFVAR", "TEXMFCONFIG"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error",
                    f"-output-directory={BUILD}", str(tex)], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    shutil.copyfile(BUILD / "fig2_paper_architecture_revised.pdf", OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
