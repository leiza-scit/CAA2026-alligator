#!/usr/bin/env python3
"""Shared path constants for the CAA2026-alligator pipeline.

Every script resolves its paths through this module rather than recomputing
``Path(__file__)`` chains of its own, so moving a folder is a one-line change
here instead of an edit in several places.

This is the same module the wdt-* repository family uses, trimmed to the folders
this repository actually has:

    data/       input   Alligator TTL, findspot CSV, MoreEvents.csv
    src/        input   the Arretine workbook and the Alligator project files
    output/     output  figures, CSVs and the RDF graph
    docs/       output  the GitHub Pages site
    qmd/        output  the quarto-live notebook (optional to render)
    py/         code    the pipeline scripts
    py/templates/       Jinja2 templates + the canonical style.css

Scripts reach this module by putting ``py/`` on the path, which also makes them
independent of the working directory VS Code launches them from:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import wd_paths
"""

from __future__ import annotations

from pathlib import Path

# --- roots -----------------------------------------------------------------
PY = Path(__file__).resolve().parent           # <repo>/py
ROOT = PY.parent                               # <repo>

# --- input -----------------------------------------------------------------
DATA = ROOT / "data"                           # Alligator TTL + CSV inputs
SRC = ROOT / "src"                             # workbook and Alligator project
TEMPLATES = PY / "templates"                   # Jinja2 templates + style.css

# --- generated outputs -----------------------------------------------------
OUTPUT = ROOT / "output"                       # figures, CSVs, the RDF graph
GRAPH = OUTPUT / "arretine_sites_minigraph.ttl"
NOTEBOOK = ROOT / "notebook"                   # Jupyter notebook + its graph copy
DOCS = ROOT / "docs"                           # GitHub Pages site
DOWNLOADS = DOCS / "downloads"                 # .rq files and other downloads
QMD = ROOT / "qmd"                             # quarto-live notebook (optional)


def ensure_dirs() -> None:
    """Create every generated directory, so no step has to guard for itself."""
    for path in (OUTPUT, DOCS, DOWNLOADS, QMD):
        path.mkdir(parents=True, exist_ok=True)
