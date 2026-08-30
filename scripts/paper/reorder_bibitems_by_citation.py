#!/usr/bin/env python3
"""Reorder manual LaTeX bibitems by first citation order."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CITE_RE = re.compile(r"\\cite(?:\s*\[[^\]]*\])?\{([^}]*)\}")
BIBITEM_RE = re.compile(
    r"(?ms)^\\bibitem\{([^}]+)\}\s*\n(.*?)(?=^\\bibitem\{|^\\end\{thebibliography\})"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=Path)
    args = parser.parse_args()

    text = args.tex.read_text(encoding="utf-8")
    marker = r"\begin{thebibliography}"
    marker_pos = text.find(marker)
    if marker_pos < 0:
        raise SystemExit("No thebibliography environment found")

    body_start = text.find("\n", marker_pos) + 1
    body_end = text.find(r"\end{thebibliography}", body_start)
    if body_start == 0 or body_end < 0:
        raise SystemExit("Malformed thebibliography environment")

    cited_order: list[str] = []
    for match in CITE_RE.finditer(text[:marker_pos]):
        for key in match.group(1).split(","):
            key = key.strip()
            if key and key not in cited_order:
                cited_order.append(key)

    bibliography_body = text[body_start:body_end]
    matches = list(BIBITEM_RE.finditer(bibliography_body + r"\end{thebibliography}"))
    blocks = {
        match.group(1): f"\\bibitem{{{match.group(1)}}}\n{match.group(2).strip()}"
        for match in matches
    }
    original_order = [match.group(1) for match in matches]

    missing = [key for key in cited_order if key not in blocks]
    if missing:
        raise SystemExit(f"Cited keys missing from bibliography: {', '.join(missing)}")

    reordered_keys = cited_order + [
        key for key in original_order if key not in cited_order
    ]
    reordered_body = "\n\n".join(blocks[key] for key in reordered_keys) + "\n"

    updated = text[:body_start] + reordered_body + text[body_end:]
    args.tex.write_text(updated, encoding="utf-8")

    uncited = [key for key in original_order if key not in cited_order]
    print(f"Reordered {len(reordered_keys)} bibitems; uncited retained: {len(uncited)}")


if __name__ == "__main__":
    main()
