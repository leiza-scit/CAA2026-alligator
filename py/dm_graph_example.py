#!/usr/bin/env python3
"""The dating-machine graph, worked through once: the example behind cfm/dmgraph/.

``cfm/dmgraph/dm_graph.cfm`` runs inside ColdFusion and builds, for every
assemblage a visitor enters, the triples of the result and a Mermaid diagram of
them. This script does the same for one assemblage in Python, so that

  * the output can be looked at without a ColdFusion server
    (``cfm/dmgraph/example/index.html`` opens straight from disk);
  * the Turtle is parsed and checked here, which the ColdFusion side cannot do
    for itself;
  * the numbers can be compared with ``horizon_assignment.py`` (the model is the
    same; the pseudocount differs, see below).

It is a mirror, not the source: the ColdFusion include is what runs in
production, and the two build the same text on purpose. Change one, change the
other, and diff the example against a real page.

Model
-----
The same as the form: Dirichlet-multinomial predictive likelihood, the reference
counts of the five horizons as Dirichlet parameters plus a pseudocount, equal
priors. The cfm adds 0.5 (Jeffreys); ``horizon_assignment.py`` adds 1.0. The
difference is small and is printed, so it is never a surprise. The value used is
written into the graph as ``dm:pseudocount``.

Input
-----
* output/horizon_reference_profiles.csv — the pooled counts per horizon, written
  by ``python py/main.py --only assign``. They are the rows of
  ``public.tblservicesquantities``.

Output
------
* cfm/dmgraph/example/example.ttl     the triples
* cfm/dmgraph/example/example.mmd     the Mermaid source, without colours
* cfm/dmgraph/example/index.html      a static page that draws them

Usage
-----
    python py/dm_graph_example.py
    python py/dm_graph_example.py --counts 0,0,13,27,60
    python py/dm_graph_example.py --counts 6,12,30,40,12 --pseudocount 1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wd_paths                                              # noqa: E402

from rdflib import Graph, Namespace, RDF                      # noqa: E402


# ==============================================================================
# SECTION 1 · Configuration
# ==============================================================================

REFERENCE_CSV = wd_paths.OUTPUT / "horizon_reference_profiles.csv"
EXAMPLE_DIR = wd_paths.ROOT / "cfm" / "dmgraph" / "example"

BASE = "http://leiza-scit.github.io/CAA2026-alligator/"
DM = Namespace(BASE + "dm/")
AEONT = Namespace(BASE + "ontology#")
AEVOC = Namespace(BASE + "vocabulary/")
AECOL = Namespace(BASE + "collections/")
CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
LADO = Namespace("http://archaeology.link/ontology#")

# The form's five categories, in form order, as stage-level concepts of the
# service typology, and the labels the form uses.
SLUGS = ["Schraegrandteller", "ServiceIa", "ServiceIb", "ServiceIc", "ServiceII"]
LABELS = ["BordOblique", "Ia", "Ib", "Ic", "II"]

DEFAULT_COUNTS = [0, 0, 13, 27, 60]     # the README's Ib=13, Ic=27, II=60 at n = 100
DEFAULT_PSEUDOCOUNT = 0.5               # Jeffreys, as in services_output_quantities.cfm


# ==============================================================================
# SECTION 2 · The attribution
# ==============================================================================

def read_reference(path: Path = REFERENCE_CSV) -> list[list[int]]:
    """Counts per horizon h1..h5, columns in form order."""
    if not path.exists():
        sys.exit(f"  !!  {path} not found. Run: python py/main.py --only assign")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = [name for name in rows[0] if name.startswith("count_")]
    if len(columns) != 5 or len(rows) != 5:
        sys.exit(f"  !!  expected 5 horizons x 5 count columns, "
                 f"found {len(rows)} x {len(columns)}")
    rows.sort(key=lambda row: int(row["horizon"]))
    return [[int(row[column]) for column in columns] for row in rows]


def posterior(observed: list[int], reference: list[list[int]],
              pseudocount: float) -> list[float]:
    """Normalised posterior over the five horizons, equal priors."""
    n = sum(observed)
    logs = []
    for counts in reference:
        alpha = [c + pseudocount for c in counts]
        total = sum(alpha)
        value = math.lgamma(total) - math.lgamma(total + n)
        for a, x in zip(alpha, observed):
            value += math.lgamma(a + x) - math.lgamma(a)
        logs.append(value - math.log(5))
    top = max(logs)
    weights = [math.exp(v - top) for v in logs]
    return [w / sum(weights) for w in weights]


# ==============================================================================
# SECTION 3 · The two texts
# ==============================================================================
# These mirror dm_graph.cfm line for line. Numbers go through decimal(), which
# does what dmDecimal() does there: integer arithmetic, no locale, no exponent.

def decimal(value: float, digits: int) -> str:
    scale = 10 ** digits
    scaled = math.floor(value * scale + 0.5)
    whole, frac = divmod(scaled, scale)
    return f"{whole}.{frac:0{digits}d}"


def build_turtle(observed, probabilities, pseudocount) -> str:
    prior = decimal(1 / 5, 6)
    t = [
        f"@prefix dm: <{BASE}dm/> .",
        f"@prefix aeont: <{BASE}ontology#> .",
        f"@prefix aevoc: <{BASE}vocabulary/> .",
        f"@prefix aecol: <{BASE}collections/> .",
        "@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .",
        "@prefix lado: <http://archaeology.link/ontology#> .",
        "",
        "_:assemblage a dm:Assemblage , crm:E18_Physical_Thing ;",
        "    dm:hasCount _:count_1 , _:count_2 , _:count_3 , _:count_4 , _:count_5 .",
    ]
    for i in range(5):
        t += ["",
              f"_:count_{i + 1} a aeont:ServiceCount ;",
              f"    aeont:serviceType aevoc:{SLUGS[i]} ;",
              f"    aeont:sherdCount {observed[i]} ."]
    t += ["",
          "[] a dm:HorizonAssessment , crm:E7_Activity ;",
          '    dm:method "Dirichlet-multinomial predictive likelihood" ;',
          f"    dm:pseudocount {decimal(pseudocount, 6)} ;",
          "    crm:P9_consists_of"]
    for i in range(5):
        end = " ," if i < 4 else " ."
        t.append("        [ a dm:HorizonResult , crm:E13_Attribute_Assignment"
                 " ; crm:P140_assigned_attribute_to _:assemblage"
                 f" ; crm:P141_assigned aecol:horizon_{i + 1}"
                 f" ; dm:prior {prior}"
                 f" ; dm:probability {decimal(probabilities[i], 6)} ]{end}")
    for i in range(5):
        t += ["", f"aecol:horizon_{i + 1} a lado:ChronologicalHorizon , crm:E4_Period ."]
    return "\n".join(t)


def build_mermaid(observed, probabilities, pseudocount) -> str:
    best = max(range(5), key=lambda i: probabilities[i]) + 1
    q = '"'
    m = ["flowchart TB;",
         f"asm({q}_:assemblage<br/>dm:Assemblage<br/>crm:E18_Physical_Thing{q});"]
    for i in range(5):
        m.append(f"c{i + 1}({q}_:count_{i + 1}<br/>aeont:ServiceCount<br/>"
                 f"{LABELS[i]}: {observed[i]} sherds{q});")
    for i in range(5):
        m.append(f"s{i + 1}({q}aevoc:{SLUGS[i]}{q});")
    m.append(f"asx({q}_:assessment<br/>dm:HorizonAssessment<br/>crm:E7_Activity<br/>"
             f"pseudocount {decimal(pseudocount, 6)}{q});")
    for i in range(5):
        m.append(f"r{i + 1}({q}dm:HorizonResult<br/>crm:E13_Attribute_Assignment<br/>P = "
                 f"{decimal(probabilities[i] * 100, 2)} %{q});")
    for i in range(5):
        m.append(f"h{i + 1}({q}aecol:horizon_{i + 1}"
                 f"<br/>lado:ChronologicalHorizon<br/>crm:E4_Period{q});")

    m.append(f"asx -- {q}crm:P9_consists_of{q} --> r1;")
    m += [f"asx --> r{i};" for i in range(2, 6)]
    m.append(f"r1 -- {q}crm:P141_assigned{q} --> h1;")
    m += [f"r{i} --> h{i};" for i in range(2, 6)]
    m += [f"r{i} --> asm;" for i in range(1, 5)]
    m.append(f"r5 -- {q}crm:P140_assigned_attribute_to{q} --> asm;")
    m.append(f"asm -- {q}dm:hasCount{q} --> c1;")
    m += [f"asm --> c{i};" for i in range(2, 6)]
    m.append(f"c1 -- {q}aeont:serviceType{q} --> s1;")
    m += [f"c{i} --> s{i};" for i in range(2, 6)]

    m.append("class asm,asx,c1,c2,c3,c4,c5,r1,r2,r3,r4,r5,h1,h2,h3,h4,h5 ent;")
    m.append("class s1,s2,s3,s4,s5 term;")
    m.append(f"class r{best},h{best} best;")
    return "\n".join(m)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dating machine graph: example</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 1100px; margin: 2.5rem auto; padding: 0 1rem; color: #222; }
  h1 { font-size: 1.55rem; }
  .note { color: #555; font-size: .92rem; line-height: 1.45; }
</style>
</head>
<body>
<h1>Dating machine graph: example</h1>
<p class="note">Static example for the assemblage {summary}. Generated by
py/dm_graph_example.py; the page it mirrors is services_output_quantities.cfm with
dmgraph/dm_graph.cfm included.</p>

<link rel="stylesheet" href="../dm-graph.css">

<section id="dm-graph" class="dm-graph">
    <h2>The result as a graph</h2>
    <p class="note">
        The same result as RDF triples. The assemblage is a blank node: it has no
        identifier of its own and exists only in this graph. Each result is a
        CIDOC-CRM attribute assignment (E13) that assigns one horizon to the
        assemblage. Arrows follow the triples; a label is printed once per group
        of parallel arrows. Concepts (purple) and horizons are those of the
        CAA2026-alligator graphs, so this graph can be merged with them. The
        heavy outline marks the most probable horizon.
    </p>

    <div id="dm-graph-canvas" class="dm-graph__canvas">Drawing the graph &hellip;</div>

    <p class="dm-graph__actions">
        <label>Layout
            <select id="dm-layout">
                <option value="TB">Top to bottom</option>
                <option value="LR">Left to right</option>
                <option value="BT">Bottom to top</option>
                <option value="RL">Right to left</option>
            </select>
        </label>
        <button type="button" id="dm-download-ttl">Download Turtle</button>
        <button type="button" id="dm-download-svg">Download SVG</button>
    </p>

    <details class="dm-graph__source">
        <summary>Show the triples (Turtle)</summary>
        <pre id="dm-turtle-view"></pre>
    </details>

    <script type="text/turtle" id="dm-turtle">{turtle}</script>
    <script type="text/x-mermaid" id="dm-mermaid">{mermaid}</script>
</section>

<script src="../vendor/mermaid.min.js"></script>
<script src="../dm-graph.js"></script>
</body>
</html>
"""


# ==============================================================================
# SECTION 4 · Checks
# ==============================================================================

def check(turtle: str, observed, probabilities) -> None:
    """Parse what was written and confirm it says what was meant."""
    graph = Graph().parse(data=turtle, format="turtle")

    counts = {}
    for node in graph.subjects(RDF.type, AEONT.ServiceCount):
        concept = graph.value(node, AEONT.serviceType)
        counts[str(concept).rsplit("/", 1)[-1]] = int(graph.value(node, AEONT.sherdCount))
    expected = dict(zip(SLUGS, observed))
    assert counts == expected, f"counts in graph {counts} != {expected}"

    assessments = list(graph.subjects(RDF.type, DM.HorizonAssessment))
    assert len(assessments) == 1, "expected exactly one assessment"
    assert (assessments[0], RDF.type, CRM.E7_Activity) in graph
    results = list(graph.objects(assessments[0], CRM.P9_consists_of))
    assert len(results) == 5, "expected five results"

    assemblages = list(graph.subjects(RDF.type, DM.Assemblage))
    assert len(assemblages) == 1
    assert (assemblages[0], RDF.type, CRM.E18_Physical_Thing) in graph
    assert all(not str(a).startswith("http") for a in assemblages), \
        "assemblage must be a blank node"
    assert set(graph.objects(assemblages[0], DM.hasCount)) == \
        set(graph.subjects(RDF.type, AEONT.ServiceCount)), "hasCount must reach every count"

    horizons = set()
    for r in results:
        assert (r, RDF.type, DM.HorizonResult) in graph
        assert (r, RDF.type, CRM.E13_Attribute_Assignment) in graph
        assert graph.value(r, CRM.P140_assigned_attribute_to) == assemblages[0]
        horizons.add(graph.value(r, CRM.P141_assigned))
    assert horizons == {AECOL[f"horizon_{i}"] for i in range(1, 6)}, "five distinct horizons"
    for h in horizons:
        assert (h, RDF.type, LADO.ChronologicalHorizon) in graph
        assert (h, RDF.type, CRM.E4_Period) in graph

    total = sum(float(graph.value(r, DM.probability)) for r in results)
    assert abs(total - 1.0) < 5e-6, f"probabilities sum to {total}"
    print(f"  OK  {len(graph)} triples parse; five counts, five results, "
          f"sum of probabilities {total:.6f}")


# ==============================================================================
# SECTION 5 · Entry point
# ==============================================================================

def parse_counts(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError:
        sys.exit("  !!  --counts takes five whole numbers, e.g. 0,0,13,27,60")
    if len(values) != 5 or min(values) < 0 or sum(values) == 0:
        sys.exit("  !!  --counts takes five non-negative whole numbers, not all zero")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--counts", default=",".join(map(str, DEFAULT_COUNTS)),
                        help="BordOblique,Ia,Ib,Ic,II (default: %(default)s)")
    parser.add_argument("--pseudocount", type=float, default=DEFAULT_PSEUDOCOUNT,
                        help="added to every reference count (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=EXAMPLE_DIR,
                        help="output folder (default: cfm/dmgraph/example)")
    args = parser.parse_args()

    observed = parse_counts(args.counts)
    reference = read_reference()
    probabilities = posterior(observed, reference, args.pseudocount)

    print("=" * 60)
    print("Dating-machine graph · one assemblage")
    print("=" * 60)
    print("  ..  assemblage " + ", ".join(f"{l} {c}" for l, c in zip(LABELS, observed))
          + f"  (n = {sum(observed)})")
    for i, p in enumerate(probabilities, start=1):
        print(f"  ..  horizon {i}: {p * 100:6.2f} %")

    other = 1.0 if args.pseudocount != 1.0 else 0.5
    alt = posterior(observed, reference, other)
    gap = max(abs(a - b) for a, b in zip(probabilities, alt)) * 100
    print(f"  ..  with pseudocount {other:g} instead of {args.pseudocount:g} the "
          f"largest difference is {gap:.2f} percentage points")

    turtle = build_turtle(observed, probabilities, args.pseudocount)
    mermaid = build_mermaid(observed, probabilities, args.pseudocount)
    check(turtle, observed, probabilities)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "example.ttl").write_text(turtle + "\n", encoding="utf-8", newline="\n")
    (args.out / "example.mmd").write_text(mermaid + "\n", encoding="utf-8", newline="\n")
    summary = ", ".join(f"{l} {c}" for l, c in zip(LABELS, observed))
    page = PAGE.replace("{summary}", summary) \
               .replace("{turtle}", turtle) \
               .replace("{mermaid}", mermaid)
    (args.out / "index.html").write_text(page, encoding="utf-8", newline="\n")

    rel = args.out.relative_to(wd_paths.ROOT) if args.out.is_relative_to(wd_paths.ROOT) else args.out
    print(f"  OK  {rel}/example.ttl, example.mmd, index.html")
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
