"""Orchestrator: run the whole CAA2026-alligator figure pipeline in one go.

Why this file exists
--------------------
``alligator_to_clean_rdf.py`` starts its run with ``setup_output_dir()``, which
deletes and recreates ``root/output/`` to guarantee a clean state. Anything the
companion script wrote earlier — the ``*_by_service*`` figures, the service
percentage CSVs — is wiped with it.

The fix is simply to fix the ORDER: the resetting script runs first, the
additive script second. This orchestrator enforces that order, so a single
command produces *both* sets of outputs and nothing is lost.

Usage
-----
    python py/main.py                 # run the full pipeline (default)
    python py/main.py --only rdf      # only alligator_to_clean_rdf.py
    python py/main.py --only service  # only events_timeline_by_service.py
    python py/main.py --list          # show the configured steps and exit

Each step runs as its own process, so a crash in one script cannot corrupt the
state of the other; its output streams straight to the terminal. The run stops
at the first failing step and returns a non-zero exit code, which makes this
safe to call from CI or a scheduled task.
"""

from __future__ import annotations

# ==============================================================================
# SECTION 1 · Imports
# ==============================================================================

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ==============================================================================
# SECTION 2 · Configuration
# ==============================================================================
# Directory layout: this file lives in root/py/ alongside the pipeline scripts.

PY_DIR = Path(__file__).resolve().parent
REPO_ROOT = PY_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"

# The pipeline steps, in the order they MUST run.
#
#   key      short name for the --only switch
#   script   file in root/py/
#   resets   True if the script clears root/output/ before writing
#   expects  representative output files (repo-root relative), checked after
#            the step runs — a step may write outside output/, e.g. into docs/
#
# ORDER MATTERS: the step with resets=True has to come first, otherwise it
# deletes the figures produced by the steps before it.
STEPS = [
    {
        "key": "rdf",
        "script": "alligator_to_clean_rdf.py",
        "label": "RDF pipeline · clusters, Allen matrix, events timeline",
        "resets": True,
        "expects": ["output/events_timeline.jpg", "output/events_timeline.svg"],
    },
    {
        "key": "service",
        "script": "events_timeline_by_service.py",
        "label": "Service composition · timeline, within-group variance/quality",
        "resets": False,
        "expects": [
            "output/events_timeline_by_service_en.jpg",
            "output/events_timeline_by_service_fr.jpg",
            "output/service_group_variability_en.jpg",
            "output/service_group_variability_fr.jpg",
            "output/service_percentages.csv",
            "output/service_group_variability.csv",
        ],
    },
    {
        # Not "services": "service" is already taken by the figure step above,
        # and two --only keys a letter apart is a trap.
        "key": "enrich",
        "script": "services_to_rdf.py",
        "label": "Service counts \u00b7 RDF layer over the minigraph",
        "resets": False,
        # Reads the minigraph the first step wrote and the workbook, and writes
        # a separate file beside them. It never touches the minigraph, so the
        # two layers stay separately citable - and it has to run before the
        # query page, which verifies the service queries against this file.
        "expects": ["output/arretine_services.ttl"],
    },
    {
        "key": "sparql",
        "script": "build_sparql.py",
        "label": "Interactive query page · docs/sparql.html, .rq files, qmd",
        "resets": False,
        "expects": [
            "docs/sparql.html",
            "docs/arretine_sites_minigraph.ttl",
            "docs/arretine_services.ttl",
            "docs/downloads/queries",
        ],
        # The page links style.css next to itself; the canonical copy lives in
        # py/templates/, so it is synced here rather than maintained twice.
        "copy_after": [("py/templates/style.css", "docs/style.css")],
    },
]


# ==============================================================================
# SECTION 3 · Step Runner
# ==============================================================================


def run_step(step: dict) -> bool:
    """Run one pipeline step as a subprocess. Returns True on success.

    Output is not captured, so the script's own progress messages appear in the
    terminal as they happen. The same interpreter that runs this orchestrator is
    used for the child process, which keeps virtual environments consistent.
    """
    script_path = PY_DIR / step["script"]

    print()
    print("=" * 78)
    print(f"▶ {step['script']}")
    print(f"  {step['label']}")
    if step["resets"]:
        print("  note: this step resets root/output/ before writing")
    print("=" * 78)

    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False

    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        print(f"\n✗ {step['script']} failed (exit code {result.returncode}) "
              f"after {elapsed:.1f} s")
        return False

    print(f"\n✓ {step['script']} finished in {elapsed:.1f} s")

    for src_rel, dst_rel in step.get("copy_after", []):
        src, dst = REPO_ROOT / src_rel, REPO_ROOT / dst_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"  copied {src_rel} -> {dst_rel}")
        else:
            print(f"  ⚠ {src_rel} not found; {dst_rel} not updated")

    # Verify the representative outputs actually landed in root/output/.
    missing = [f for f in step["expects"] if not (REPO_ROOT / f).exists()]
    if missing:
        print("  ⚠ Expected output(s) missing:")
        for f in missing:
            print(f"      · {f}")
    return True


def summarise_outputs() -> None:
    """Print the contents of root/output/ so the full result set is visible."""
    print()
    print("=" * 78)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 78)

    if not OUTPUT_DIR.exists():
        print("  (directory does not exist)")
        return

    files = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_file())
    if not files:
        print("  (empty)")
        return

    for p in files:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<48} {size_kb:>9,.1f} KB")
    print(f"\n  {len(files)} file(s) total")


# ==============================================================================
# SECTION 4 · Main Entry Point
# ==============================================================================


def main() -> int:
    """Run the configured steps in order. Returns a shell exit code."""
    parser = argparse.ArgumentParser(
        description="Run the CAA2026-alligator figure pipeline."
    )
    parser.add_argument(
        "--only",
        choices=[s["key"] for s in STEPS],
        help="run a single step instead of the whole pipeline",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the configured steps and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("Configured steps (in execution order):")
        for i, s in enumerate(STEPS, start=1):
            flag = "  [resets output/]" if s["resets"] else ""
            print(f"  {i}. {s['key']:<8} {s['script']}{flag}")
        return 0

    steps = STEPS if args.only is None else [s for s in STEPS if s["key"] == args.only]

    # Running the additive step on its own is fine; running the resetting step
    # on its own is what deletes the other figures, so warn about it.
    if args.only and steps and steps[0]["resets"]:
        print("⚠ Running this step alone clears root/output/ and therefore removes "
              "the figures produced by the other step.\n"
              "  Run 'python py/main.py' without --only to regenerate everything.")

    started = time.perf_counter()
    for step in steps:
        if not run_step(step):
            print("\nPipeline aborted.")
            return 1

    summarise_outputs()
    print(f"\n✓ Pipeline complete in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())