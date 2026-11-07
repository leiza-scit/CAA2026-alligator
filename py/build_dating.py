#!/usr/bin/env python3
"""The dating machine on GitHub Pages: data, vocabulary page and a check of the JavaScript.

``docs/dating.html`` is the dating machine without ColdFusion. It runs entirely
in the browser: ``docs/dmgraph/dm-model.js`` computes the attribution and builds
the Turtle and the Mermaid source, ``dm-charts.js`` and ``dm-app.js`` draw them.
This script does what the page cannot do for itself.

1. ``docs/dmgraph/reference.js``: the pooled sherd counts of the five horizons
   (from ``output/horizon_reference_profiles.csv``), a hash of those 25 counts,
   and the time interval and label of each horizon (from
   ``output/arretine_sites_minigraph.ttl``, the graph of the site). That is all
   the data the page needs.
2. ``docs/dm/``: the vocabulary of the result graph, ``dm.ttl`` (a copy of
   ``cfm/dmgraph/dm.ttl``) and ``index.html``, a readable page written from it.
   ``http://leiza-scit.github.io/CAA2026-alligator/dm#`` shows that page.
3. A check of ``dm-model.js``: it runs under node for a handful of assemblages and
   its Turtle and Mermaid source are compared, character for character, with the
   Python mirror (``dm_graph_example.py``); probabilities and log likelihoods to
   1e-9. Skipped with a notice if node is not installed.

Usage
-----
    python py/build_dating.py
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wd_paths                                              # noqa: E402
import dm_graph_example as dm                                 # noqa: E402

from rdflib import Graph, Literal, Namespace, RDF, RDFS       # noqa: E402

DOCS = wd_paths.ROOT / "docs"
OUT_DIR = DOCS / "dmgraph"
MODEL_JS = OUT_DIR / "dm-model.js"
REFERENCE_JS = OUT_DIR / "reference.js"
VOCAB_SRC = wd_paths.ROOT / "cfm" / "dmgraph" / "dm.ttl"
VOCAB_DIR = DOCS / "dm"
SITE_GRAPH = wd_paths.OUTPUT / "arretine_sites_minigraph.ttl"

TIME = Namespace("http://www.w3.org/2006/time#")

# Whole assemblages of different kinds: the README's, an early and a late one,
# a single sherd, a large one, and both pseudocounts. Every case runs with the
# software named, the second half also with the hash left out and no software.
WHEN = "2026-01-01T00:00:00Z"
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
const input = JSON.parse(process.argv[2]);
const out = input.cases.map(([obs, pc, plain]) => {
  const meta = plain ? m.makeMeta(input.when, '', '') : m.makeMeta(input.when, input.hash);
  const p = m.posterior(obs, input.reference, pc);
  const ll = m.logLikelihood(obs, input.reference, pc);
  return {p: p, ll: ll, ttl: m.buildTurtle(obs, p, ll, pc, meta), mmd: m.buildMermaid(obs, p, pc, meta)};
});
process.stdout.write(JSON.stringify(out));
"""


# ------------------------------------------------------------------------------
# 1 · reference.js
# ------------------------------------------------------------------------------

def horizon_intervals() -> dict[int, dict]:
    """Label and years of the five horizons, from the graph of the site."""
    graph = Graph().parse(SITE_GRAPH)
    AECOL = dm.AECOL
    found = {}
    for number in range(1, 6):
        node = AECOL[f"horizon_{number}"]
        labels = [l for l in graph.objects(node, RDFS.label)
                  if isinstance(l, Literal) and l.language == "en"]
        begin = graph.value(graph.value(node, TIME.hasBeginning), TIME.inXSDgYear)
        end = graph.value(graph.value(node, TIME.hasEnd), TIME.inXSDgYear)
        if not labels or begin is None or end is None:
            sys.exit(f"  !!  {node} has no English label or no begin/end year in {SITE_GRAPH.name}")
        found[number] = {"iri": str(node), "label": str(labels[0]),
                         "begin": int(str(begin)), "end": int(str(end))}
    return found


# The Allen relations of the site graph between the five horizons that the poster
# graph draws: the non-inverse ones except "before", which would join almost
# every pair and drown the rest. (Before, After and the inverses are in the graph.)
ALLEN_DRAWN = ["intervalMeets", "intervalStarts", "intervalDuring",
               "intervalOverlaps", "intervalFinishes"]


def horizon_relations() -> list[dict]:
    graph = Graph().parse(SITE_GRAPH)
    found = []
    for a in range(1, 6):
        for b in range(1, 6):
            for rel in ALLEN_DRAWN:
                if (dm.AECOL[f"horizon_{a}"], TIME[rel], dm.AECOL[f"horizon_{b}"]) in graph:
                    found.append({"from": a, "rel": "time:" + rel, "to": b})
    if not found:
        sys.exit(f"  !!  no Allen relations between the horizons in {SITE_GRAPH.name}")
    return found


def write_reference() -> list[list[int]]:
    import csv
    reference = dm.read_reference()
    with dm.REFERENCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r["horizon"]))
    intervals = horizon_intervals()
    payload = {
        "source": "output/horizon_reference_profiles.csv",
        "intervalSource": "output/arretine_sites_minigraph.ttl",
        "labels": dm.LABELS,
        "countsSha256": dm.reference_hash(reference),
        "allen": horizon_relations(),
        "horizons": [
            {"horizon": int(r["horizon"]), "n_total": int(r["n_total"]), "counts": counts,
             **intervals[int(r["horizon"])]}
            for r, counts in zip(rows, reference)],
    }
    # A script rather than a .json: the page then also opens from disk, where a
    # browser refuses to fetch a neighbouring file.
    REFERENCE_JS.write_text(
        "/* Written by py/build_dating.py from output/horizon_reference_profiles.csv\n"
        "   and output/arretine_sites_minigraph.ttl. */\n"
        "window.DMReference = " + json.dumps(payload, indent=1, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    print(f"  OK  {REFERENCE_JS.relative_to(wd_paths.ROOT)}  "
          f"({len(reference)} horizons, {sum(map(sum, reference))} sherds, "
          f"{intervals[1]['begin']}..{intervals[5]['end']}, "
          f"{len(payload['allen'])} Allen relations)")
    return reference


# ------------------------------------------------------------------------------
# 2 · docs/dm/
# ------------------------------------------------------------------------------

def write_vocabulary() -> None:
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VOCAB_SRC, VOCAB_DIR / "dm.ttl")
    graph = Graph().parse(VOCAB_SRC)
    ns = str(dm.DM)

    def entries(kind):
        items = sorted((s for s in graph.subjects(RDF.type, kind) if str(s).startswith(ns)), key=str)
        rows = []
        for s in items:
            name = str(s)[len(ns):]
            label = graph.value(s, RDFS.label)
            comment = graph.value(s, RDFS.comment)
            extra = []
            for prop, word in [(RDFS.subClassOf, "subclass of"), (RDFS.domain, "domain"),
                               (RDFS.range, "range")]:
                for o in graph.objects(s, prop):
                    short = graph.namespace_manager.normalizeUri(o).strip("<>")
                    extra.append(f"{word} <code>{html.escape(short)}</code>")
            rows.append(f'<dt id="{name}"><code>dm:{name}</code> &mdash; {html.escape(str(label))}</dt>'
                        f"<dd>{html.escape(str(comment))}"
                        + (f'<br><span class="meta">{"; ".join(extra)}</span>' if extra else "")
                        + "</dd>")
        return "\n".join(rows)

    page = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAA 2026 &mdash; dm: vocabulary</title>
<link rel="stylesheet" href="../style.css">
<style>dt {{ margin-top: 1rem; font-weight: 650; }} dd {{ margin-left: 0; color: var(--muted); }}
.meta {{ font-size: .85rem; }} pre {{ white-space: pre-wrap; background: var(--head-bg); padding: .75rem; border: 1px solid var(--rule); font-size: .82rem; }}</style>
</head>
<body>
<div class="container">
  <nav class="nav">
    <a href="../index.html">Overview</a>
    <a href="../sparql.html">Query</a>
    <a href="../dating.html">Dating machine</a>
  </nav>
  <header class="page-head">
    <h1>The <code>dm:</code> vocabulary</h1>
    <p class="intro">Namespace <code>{html.escape(ns)}</code>, version 0.2. The terms of the
    result graph of the <a href="../dating.html">dating machine</a>, on top of CIDOC-CRM.
    <a href="dm.ttl">Download as Turtle</a>.</p>
  </header>
  <h2>Classes</h2>
  <dl>
{entries(RDFS.Class)}
  </dl>
  <h2>Properties</h2>
  <dl>
{entries(RDF.Property)}
  </dl>
  <h2>Notes on the modelling</h2>
  <pre>{html.escape(chr(10).join(l[2:] for l in VOCAB_SRC.read_text(encoding="utf-8").splitlines() if l.startswith("# ")))}</pre>
  <footer class="page-foot">Written by <code>py/build_dating.py</code> from <code>cfm/dmgraph/dm.ttl</code>.</footer>
</div>
</body>
</html>
"""
    (VOCAB_DIR / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"  OK  {VOCAB_DIR.relative_to(wd_paths.ROOT)}/dm.ttl, index.html "
          f"({len(graph)} triples)")


# ------------------------------------------------------------------------------
# 3 · the check
# ------------------------------------------------------------------------------

def check_javascript(reference: list[list[int]]) -> None:
    node = shutil.which("node")
    if node is None:
        print("  --  node not found: dm-model.js was not compared with the Python mirror")
        return
    ref_hash = dm.reference_hash(reference)
    plain_from = 4                       # the last cases run without hash and software
    cases = [[obs, pc, i >= plain_from] for i, (obs, pc) in enumerate(CASES)]
    run = subprocess.run(
        [node, "-e", NODE_SCRIPT, str(MODEL_JS),
         json.dumps({"cases": cases, "reference": reference, "when": WHEN, "hash": ref_hash})],
        capture_output=True, text=True, encoding="utf-8")
    if run.returncode != 0:
        sys.exit(f"  !!  node failed:\n{run.stderr}")
    for (observed, pc, plain), got in zip(cases, json.loads(run.stdout)):
        want_p = dm.posterior(observed, reference, pc)
        want_l = dm.log_likelihood(observed, reference, pc)
        assert max(abs(a - b) for a, b in zip(want_p, got["p"])) < 1e-9, f"{observed}: probabilities"
        assert max(abs(a - b) for a, b in zip(want_l, got["ll"])) < 1e-9, f"{observed}: log likelihoods"
        # The Python builders get the numbers the JavaScript computed, so the texts
        # are compared as text; the numbers were compared just above.
        meta = (dm.make_meta(WHEN, software="") if plain
                else dm.make_meta(WHEN, ref_hash=ref_hash))
        assert got["ttl"] == dm.build_turtle(observed, got["p"], got["ll"], pc, meta), \
            f"{observed}: Turtle differs"
        assert got["mmd"] == dm.build_mermaid(observed, got["p"], pc, meta), \
            f"{observed}: Mermaid differs"
    print(f"  OK  dm-model.js equals the Python mirror for {len(CASES)} assemblages "
          f"(numbers within 1e-9, Turtle and Mermaid identical)")


def main() -> None:
    reference = write_reference()
    write_vocabulary()
    check_javascript(reference)
    print("Done.")


if __name__ == "__main__":
    main()
