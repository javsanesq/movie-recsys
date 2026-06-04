#!/usr/bin/env python3
"""Generate the movie-recsys technical deep-dive study guide (.docx).

This is a thin wrapper: the document is authored in JavaScript with the
`docx` library (matching the rag-assistant reference doc's toolchain). The JS
source lives next to this file. Running this script writes the .js and invokes
node, so the .docx is reproducible from the repo.

    python scripts/build_study_guide.py

Requires: node + the global `docx` npm package (`npm install -g docx`).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JS = Path(__file__).resolve().parent / "_build_study_guide.js"
OUT = ROOT / "output" / "doc" / "movie-recsys-technical-deep-dive.docx"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not JS.exists():
        print(f"missing {JS} (the docx-js authoring source)", file=sys.stderr)
        return 1
    # Resolve `docx` from the global npm prefix.
    gconf = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True)
    env_node_path = gconf.stdout.strip()
    proc = subprocess.run(
        ["node", str(JS), str(OUT)],
        env={"NODE_PATH": env_node_path, "PATH": __import__("os").environ["PATH"]},
    )
    if proc.returncode != 0:
        return proc.returncode
    _inject_heading_outline_levels(OUT)
    print(f"wrote {OUT}")
    return 0


def _inject_heading_outline_levels(path: Path) -> None:
    """Normalize the Heading1/Heading2 *style* definitions in word/styles.xml.

    This version of docx-js (a) emits a duplicate built-in heading style with
    the same styleId alongside our custom one, (b) names the style "Heading N"
    with a capital H, and (c) never writes <w:outlineLvl> into the style
    definition (only onto individual paragraphs). Word and downstream tools
    (pandoc, the TOC field, the navigation pane) recognize a heading by the
    lowercase built-in style name "heading N" and by the style's outline level,
    so we collapse the duplicate, lowercase the name, and ensure outlineLvl is
    present — matching the rag-assistant reference deep-dive. Patched in place
    with the stdlib zipfile (no extra deps).
    """
    import os
    import re
    import shutil
    import tempfile
    import zipfile

    def patch(styles: str) -> str:
        for level in (1, 2):
            sid = f"Heading{level}"
            blocks = re.findall(
                rf'<w:style [^>]*w:styleId="{sid}".*?</w:style>', styles, re.S
            )
            # Keep the richest definition (the one carrying our run/paragraph
            # formatting, i.e. the longest), drop any duplicate stub.
            keep = max(blocks, key=len)
            fixed = keep.replace(
                f'<w:name w:val="Heading {level}"/>',
                f'<w:name w:val="heading {level}"/>',
            )
            if "<w:outlineLvl" not in fixed:
                if "<w:pPr>" in fixed:
                    fixed = fixed.replace(
                        "<w:pPr>", f'<w:pPr><w:outlineLvl w:val="{level - 1}"/>', 1
                    )
                else:
                    fixed = fixed.replace(
                        "<w:qFormat/>",
                        f'<w:qFormat/><w:pPr><w:outlineLvl w:val="{level - 1}"/></w:pPr>',
                        1,
                    )
            # Remove all original blocks, then re-insert the single fixed one.
            for b in blocks:
                styles = styles.replace(b, "", 1)
            styles = styles.replace("</w:styles>", fixed + "</w:styles>", 1)
        return styles

    src = zipfile.ZipFile(path)
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".docx")
    os.close(tmp_fd)
    with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/styles.xml":
                data = patch(data.decode("utf-8")).encode("utf-8")
            dst.writestr(item, data)
    src.close()
    shutil.move(tmp_name, path)


if __name__ == "__main__":
    raise SystemExit(main())
