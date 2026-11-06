#!/usr/bin/env python3
"""The dating machine on GitHub Pages: reference data and a check of the JavaScript.

``docs/dating.html`` is the dating machine without ColdFusion. It runs entirely
in the browser: ``docs/dmgraph/dm-model.js`` computes the attribution and builds
the Turtle and the Mermaid source, ``docs/dmgraph/dm-app.js`` draws them. This
script does the two things the page cannot do for itself.

1. It writes ``docs/dmgraph/reference.js``, the pooled sherd counts of the five
   horizons, from ``output/horizon_reference_profiles.csv``. That is the only
   data the page needs.
2. It runs ``dm-model.js`` under node for a handful of assemblages and compares
   what it returns, character for character, with the Python mirror
   (``dm_graph_example.py``): the Turtle, the Mermaid source, and the
   probabilities to 1e-9. Skipped with a notice if node is not installed.

Usage
-----
    python py/build_dating.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wd_paths                                              # noqa: E402
import dm_graph_example as dm                                 # noqa: E402

DOCS = wd_paths.ROOT / "docs"
OUT_DIR = DOCS / "dmgraph"
MODEL_JS = OUT_DIR / "dm-model.js"
REFERENCE_JS = OUT_DIR / "reference.js"

# Whole assemblages of different kinds: the README's, an early and a late one,
# a single sherd, a large one, and both pseudocounts.
CASES = [
    ([0, 0, 13, 27, 60], 0.5),
    ([0, 0, 13, 27, 60], 1.0),
    ([6, 12, 30, 40, 12], 0.5),
    ([3, 5, 10, 8, 1], 1.0),
    ([0, 0, 0, 0, 1], 0.5),
    ([0, 0, 0, 1, 0], 0.5),
    ([20, 60, 300, 450, 200], 0.5),
]

NODE_SCRIPT = """
const m = require(process.argv[1]);
const ref = JSON.parse(process.argv[2]);
const out = JSON.parse(process.argv[3]).map(([obs, pc]) => {
  const p = m.posterior(obs, ref, pc);
  return {p: p, ttl: m.buildTurtle(obs, p, pc), mmd: m.buildMermaid(obs, p, pc)};
});
process.stdout.write(JSON.stringify(out));
"""


def write_reference() -> list[list[int]]:
    reference = dm.read_reference()
    import csv
    with dm.REFERENCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r["horizon"]))
    payload = {
        "source": "output/horizon_reference_profiles.csv",
        "labels": dm.LABELS,
        "horizons": [{"horizon": int(r["horizon"]), "n_total": int(r["n_total"]),
                      "counts": counts} for r, counts in zip(rows, reference)],
    }
    # A script rather than a .json: the page then also opens from disk, where a
    # browser refuses to fetch a neighbouring file.
    REFERENCE_JS.write_text(
        "/* Written by py/build_dating.py from output/horizon_reference_profiles.csv. */\n"
        "window.DMReference = " + json.dumps(payload, indent=1, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    print(f"  OK  {REFERENCE_JS.relative_to(wd_paths.ROOT)}  "
          f"({len(reference)} horizons, {sum(map(sum, reference))} sherds)")
    return reference


def check_javascript(reference: list[list[int]]) -> None:
    node = shutil.which("node")
    if node is None:
        print("  --  node not found: dm-model.js was not compared with the Python mirror")
        return
    run = subprocess.run(
        [node, "-e", NODE_SCRIPT, str(MODEL_JS), json.dumps(reference), json.dumps(CASES)],
        capture_output=True, text=True, encoding="utf-8")
    if run.returncode != 0:
        sys.exit(f"  !!  node failed:\n{run.stderr}")
    for (observed, pseudocount), got in zip(CASES, json.loads(run.stdout)):
        want = dm.posterior(observed, reference, pseudocount)
        worst = max(abs(a - b) for a, b in zip(want, got["p"]))
        assert worst < 1e-9, f"{observed}: probabilities differ by {worst}"
        # The Python builders get the probabilities the JavaScript computed, so the
        # texts are compared as text; the numbers were compared just above.
        assert got["ttl"] == dm.build_turtle(observed, got["p"], pseudocount), \
            f"{observed}: Turtle differs"
        assert got["mmd"] == dm.build_mermaid(observed, got["p"], pseudocount), \
            f"{observed}: Mermaid differs"
    print(f"  OK  dm-model.js equals the Python mirror for {len(CASES)} assemblages "
          f"(probabilities within 1e-9, Turtle and Mermaid identical)")


def main() -> None:
    check_javascript(write_reference())
    print("Done.")


if __name__ == "__main__":
    main()
