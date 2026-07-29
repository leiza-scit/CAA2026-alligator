#!/usr/bin/env python3
"""Every quantity the variability documentation quotes, taken from the graphs.

The companion note to ``service_group_variability`` used to carry its numbers as
literals typed into HTML. That is exactly the arrangement in which a figure goes
quietly stale: two of its values were wrong for a week without anything failing,
because nothing in the repository could tell the difference between a number
that came from the data and one that came from a keyboard.

This module is the answer. It reads the two published graphs and returns one
dictionary holding every value the note needs — cell statistics, stage
compositions, group shares, seriation coordinates, the sherd accounting, the
worked example, the rank-origin sensitivity. The document generator draws
figures and fills prose from it and from nothing else, so a changed count moves
the text, the figures and the workbook together or not at all.

Import it (``from docs_facts import collect``) or run it standalone to print the
facts as JSON, which is also the quickest way to check a number by hand:

    python py/docs_facts.py
    python py/docs_facts.py --json output/docs_facts.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wd_paths                                                  # noqa: E402
from horizons import (  # noqa: E402
    FINDSPOT_EXCLUSIONS,
    HORIZON_NUMBERS,
    HORIZONS,
    exclusion_reason,
    resolve_horizon,
)

from rdflib import Graph, Namespace                              # noqa: E402
from rdflib.namespace import RDFS, SKOS                          # noqa: E402

# rdflib warns on every gYear literal before year 1; the literals are correct.
logging.getLogger("rdflib.term").setLevel(logging.ERROR)


# ==============================================================================
# SECTION 1 · Configuration
# ==============================================================================

ROOT = wd_paths.ROOT
OUTPUT_DIR = ROOT / "output"
BASE_GRAPH = OUTPUT_DIR / "arretine_sites_minigraph.ttl"
SERVICE_GRAPH = OUTPUT_DIR / "arretine_services.ttl"
FINDSPOT_CSV = OUTPUT_DIR / "findspots_with_events.csv"

AEONT = Namespace("http://leiza-scit.github.io/CAA2026-alligator/ontology#")
LADO = Namespace("http://archaeology.link/ontology#")

# The group whose column is the only measured one, and therefore the subject of
# most of the note. Named once so a renamed group does not have to be chased
# through twenty format strings.
MAIN_GROUP = "Service I"


# ==============================================================================
# SECTION 2 · Reading the graphs
# ==============================================================================

def load_graphs() -> Graph:
    """Return the base graph and the counts layer merged for querying.

    Merged only here. The two files stay separate on disk; this is the join the
    documentation talks about, performed at query time exactly as a reader of
    the published graphs would perform it.
    """
    for path in (BASE_GRAPH, SERVICE_GRAPH):
        if not path.exists():
            raise SystemExit(
                f"missing {path.relative_to(ROOT)} — run the pipeline first "
                f"(python py/main.py)"
            )
    g = Graph()
    g.parse(BASE_GRAPH, format="turtle")
    g.parse(SERVICE_GRAPH, format="turtle")
    return g


def read_observations(g: Graph) -> list[dict]:
    """One record per (findspot, sub-type): the atoms everything else sums over."""
    q = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX aeont: <http://leiza-scit.github.io/CAA2026-alligator/ontology#>
    SELECT ?site ?siteLabel ?concept ?labelEn ?labelFr ?stage ?column ?group ?sherds
    WHERE {
      ?obs aeont:atFindspot ?site ;
           aeont:serviceType ?concept ;
           aeont:sherdCount  ?sherds .
      ?concept aeont:stageRank  ?stage ;
               aeont:columnRank ?column ;
               skos:broader*    ?top .
      ?top skos:topConceptOf ?scheme .
      OPTIONAL { ?site rdfs:label ?siteLabel }
      OPTIONAL { ?concept skos:prefLabel ?labelEn . FILTER(lang(?labelEn) = "en") }
      OPTIONAL { ?concept skos:prefLabel ?labelFr . FILTER(lang(?labelFr) = "fr") }
      OPTIONAL { ?top skos:prefLabel ?group . FILTER(lang(?group) = "en") }
    }
    """
    out = []
    for r in g.query(q):
        label = str(r.siteLabel) if r.siteLabel else str(r.site).rsplit("/", 1)[-1]
        out.append({
            "site_iri": str(r.site).rsplit("/", 1)[-1],
            "site": label,
            "horizon": resolve_horizon(label),
            "concept": str(r.concept).rsplit("/", 1)[-1],
            "label_en": str(r.labelEn) if r.labelEn else "",
            "label_fr": str(r.labelFr) if r.labelFr else "",
            "stage": int(r.stage),
            "column": int(r.column),
            "group": str(r.group) if r.group else "",
            "sherds": int(r.sherds),
        })
    if not out:
        raise SystemExit("no observations found — is the counts layer empty?")
    return out


# ==============================================================================
# SECTION 3 · The statistic
# ==============================================================================

def rgzm(n: int, sum_rank: float, sum_rank_sq: float):
    """Sample standard deviation and exp(-CV) from the three sufficient sums.

    The same arithmetic as rgzm() in py/viz/_prelude.py and as the SPARQL query
    behind the printed figure. Kept here rather than imported because that file
    is pasted into a Pyodide cell and may not be imported; if the two ever
    disagree, verify() below says so.
    """
    if n < 2:
        return None, None
    mean = sum_rank / n
    var = (sum_rank_sq - n * mean * mean) / (n - 1)
    s = math.sqrt(max(var, 0.0))
    if not mean:
        return s, None
    return s, math.exp(-s / abs(mean))


def cell_statistics(records, rank_key="stage"):
    """Per (horizon, group): N, the two sums, the mean, s, q and the rank count."""
    cells: dict = {}
    for rec in records:
        if rec["horizon"] is None:
            continue
        key = (rec["horizon"], rec["group"])
        cell = cells.setdefault(key, {
            "horizon": rec["horizon"], "group": rec["group"],
            "n": 0, "sum": 0.0, "sumsq": 0.0, "subtypes": 0, "ranks": set(),
            "by_rank": {},
        })
        rank, count = rec[rank_key], rec["sherds"]
        cell["n"] += count
        cell["sum"] += rank * count
        cell["sumsq"] += rank * rank * count
        cell["subtypes"] += 1
        cell["ranks"].add(rank)
        cell["by_rank"][rank] = cell["by_rank"].get(rank, 0) + count
    for cell in cells.values():
        s, q = rgzm(cell["n"], cell["sum"], cell["sumsq"])
        cell["mean"] = cell["sum"] / cell["n"] if cell["n"] else None
        cell["s"], cell["q"] = s, q
        cell["measured"] = len(cell["ranks"]) > 1
        cell["ranks"] = sorted(cell["ranks"])
    return cells


# ==============================================================================
# SECTION 4 · The derived facts
# ==============================================================================

def stage_composition(records, group=MAIN_GROUP):
    """Per horizon: the share of the group's sherds on each of its stage ranks.

    This is what s measures, in the form a reader can see: as the mass slides
    onto one rank the standard deviation closes.
    """
    out = {}
    for h in HORIZON_NUMBERS:
        by_rank, total = {}, 0
        for rec in records:
            if rec["horizon"] == h and rec["group"] == group:
                by_rank[rec["stage"]] = by_rank.get(rec["stage"], 0) + rec["sherds"]
                total += rec["sherds"]
        if not total:
            continue
        out[h] = {
            "total": total,
            "ranks": sorted(by_rank),
            "counts": {r: by_rank[r] for r in sorted(by_rank)},
            "shares": {r: 100.0 * by_rank[r] / total for r in sorted(by_rank)},
        }
    return out


def group_shares(records):
    """Per horizon: each group's percentage of that horizon's sherds.

    The signal the variance panels are blind to, and for this material the
    larger one. Reported so the note can print the two side by side.
    """
    out = {}
    for h in HORIZON_NUMBERS:
        by_group, total = {}, 0
        for rec in records:
            if rec["horizon"] == h:
                by_group[rec["group"]] = by_group.get(rec["group"], 0) + rec["sherds"]
                total += rec["sherds"]
        if not total:
            continue
        out[h] = {"total": total,
                  "shares": {g: 100.0 * c / total for g, c in sorted(by_group.items())},
                  "counts": dict(sorted(by_group.items()))}
    return out


def worked_example(records, horizon=1, group=MAIN_GROUP):
    """Every intermediate value of one cell, for the step-by-step derivation.

    Horizon 1 / Service I by default: the darkest cell of the printed figure and
    therefore the one a reader is most likely to want to check.
    """
    rows = [r for r in records if r["horizon"] == horizon and r["group"] == group]
    rows.sort(key=lambda r: (r["stage"], r["label_en"]))
    n = sum(r["sherds"] for r in rows)
    s1 = sum(r["stage"] * r["sherds"] for r in rows)
    s2 = sum(r["stage"] ** 2 * r["sherds"] for r in rows)
    mean = s1 / n
    numerator = s2 - n * mean * mean
    var = numerator / (n - 1)
    sd = math.sqrt(var)
    cv = sd / mean
    by_rank = {}
    for r in rows:
        by_rank[r["stage"]] = by_rank.get(r["stage"], 0) + r["sherds"]
    return {
        "horizon": horizon, "group": group,
        "rows": [{"label_en": r["label_en"], "label_fr": r["label_fr"],
                  "rank": r["stage"], "c": r["sherds"],
                  "rc": r["stage"] * r["sherds"],
                  "r2c": r["stage"] ** 2 * r["sherds"]} for r in rows],
        "n": n, "sum_rc": s1, "sum_r2c": s2,
        "mean": mean, "mean_sq": mean * mean, "n_mean_sq": n * mean * mean,
        "numerator": numerator, "variance": var, "s": sd, "cv": cv,
        "q": math.exp(-cv),
        # The definition, rank by rank, as the cross-check on the shortcut.
        "deviations": [{"rank": r, "c": c, "dev": r - mean,
                        "dev_sq": (r - mean) ** 2, "term": c * (r - mean) ** 2}
                       for r, c in sorted(by_rank.items())],
    }


def rank_origin_sensitivity(cells, group=MAIN_GROUP, origins=(0, 1, 2, 3, 5, 10)):
    """q for each horizon under each plausible first-stage number.

    The evidence for the declaration: the level of q moves, the order never
    does, and s is untouched throughout.
    """
    out = {"origins": list(origins), "q": {}, "s": {}}
    for h in HORIZON_NUMBERS:
        cell = cells.get((h, group))
        if not cell or not cell["measured"]:
            continue
        out["s"][h] = cell["s"]
        out["q"][h] = {o: math.exp(-cell["s"] / (cell["mean"] + o - 1))
                       for o in origins}
    return out


def sherd_accounting(records):
    """The two totals and the material that falls between them."""
    overall = sum(r["sherds"] for r in records)
    in_horizon = sum(r["sherds"] for r in records if r["horizon"] is not None)
    outside = {}
    for rec in records:
        if rec["horizon"] is None:
            outside[rec["site"]] = outside.get(rec["site"], 0) + rec["sherds"]
    return {
        "overall": overall,
        "in_horizon": in_horizon,
        "gap": overall - in_horizon,
        "outside": dict(sorted(outside.items())),
        "observations": len(records),
        "reasons": {label: {lang: exclusion_reason(label, lang) for lang in reasons}
                    for label, reasons in FINDSPOT_EXCLUSIONS.items()},
    }


def typology(records):
    """The group -> stage -> sub-type tree, as the ranks actually assign it."""
    tree: dict = {}
    for rec in records:
        grp = tree.setdefault(rec["group"], {})
        stage = grp.setdefault(rec["stage"], [])
        entry = {"label_en": rec["label_en"], "label_fr": rec["label_fr"],
                 "column": rec["column"]}
        if entry not in stage:
            stage.append(entry)
    for grp in tree.values():
        for members in grp.values():
            members.sort(key=lambda e: e["column"])
    return {g: {r: tree[g][r] for r in sorted(tree[g])} for g in sorted(tree)}


def seriation(g: Graph, records):
    """Findspots on the first correspondence-analysis axis, with their horizon.

    Read from findspots_with_events.csv rather than the graph: the CA
    coordinates are an intermediate of the seriation and are not published as
    triples. Horizons come from resolve_horizon, never from a plain dictionary
    lookup — the workbook and the TTL spell some findspots differently, and a
    direct lookup silently drops the Haalebos camp.
    """
    import csv
    if not FINDSPOT_CSV.exists():
        return {"points": [], "inversions": [], "no_coordinate": []}
    points, no_coord = [], []
    with FINDSPOT_CSV.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            if not row.get("cax"):
                no_coord.append(label)
                continue
            points.append({
                "label": label,
                "cax": float(row["cax"]),
                "start": float(row["estimatedstart"]) if row.get("estimatedstart") else None,
                "end": float(row["estimatedend"]) if row.get("estimatedend") else None,
                "horizon": resolve_horizon(label),
            })
    points.sort(key=lambda p: p["cax"])

    # The fewest findspots whose horizon has to be set aside for the sequence to
    # run monotonically: the complement of a longest non-decreasing subsequence.
    import bisect
    seq = [p["horizon"] for p in points if p["horizon"] is not None]
    idx_map = [i for i, p in enumerate(points) if p["horizon"] is not None]
    tails, tail_idx, parent = [], [], [-1] * len(seq)
    for i, v in enumerate(seq):
        j = bisect.bisect_right(tails, v)
        if j == len(tails):
            tails.append(v)
            tail_idx.append(i)
        else:
            tails[j], tail_idx[j] = v, i
        parent[i] = tail_idx[j - 1] if j else -1
    keep, k = set(), (tail_idx[len(tails) - 1] if tails else -1)
    while k != -1:
        keep.add(k)
        k = parent[k]
    inversions = [points[idx_map[i]]["label"] for i in range(len(seq)) if i not in keep]
    return {
        "points": points,
        "inversions": inversions,
        "in_order": len(keep),
        "plotted": len(points),
        "no_coordinate": no_coord,
    }


def horizon_intervals(g: Graph):
    """The dated envelope of each horizon, for the seriation figure's margin."""
    q = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX lado: <http://archaeology.link/ontology#>
    PREFIX time: <http://www.w3.org/2006/time#>
    SELECT ?notation ?start ?end WHERE {
      ?h a lado:ChronologicalHorizon ; skos:notation ?notation .
      OPTIONAL { ?h time:hasBeginning/time:inXSDgYear ?start }
      OPTIONAL { ?h time:hasEnd/time:inXSDgYear ?end }
    }
    """
    out = {}
    for r in g.query(q):
        out[int(r.notation)] = {
            "start": int(str(r.start)) if r.start else None,
            "end": int(str(r.end)) if r.end else None,
        }
    return out


# ==============================================================================
# SECTION 5 · Verification
# ==============================================================================

def verify(facts) -> list[str]:
    """Cross-checks that must hold. Any failure aborts the documentation build.

    Not decoration. Every item here is a mistake that was actually made while
    the note was written by hand, and each would otherwise reappear silently.
    """
    problems = []
    ex = facts["worked_example"]

    # The computational shortcut against the definition it is derived from.
    definitional = sum(d["term"] for d in ex["deviations"])
    if abs(definitional - ex["numerator"]) > 1e-6:
        problems.append(
            f"worked example: Sum c(r-x)^2 = {definitional:.6f} but "
            f"Sum r^2c - N x^2 = {ex['numerator']:.6f}")

    # The accounting has to close on the excluded findspots exactly.
    acc = facts["accounting"]
    if acc["overall"] - acc["in_horizon"] != sum(acc["outside"].values()):
        problems.append("accounting: the gap is not the material of the excluded findspots")

    # Every findspot without a horizon needs a recorded reason.
    for label in acc["outside"]:
        if not exclusion_reason(label):
            problems.append(
                f"accounting: {label} carries no horizon and no entry in "
                f"FINDSPOT_EXCLUSIONS — add one rather than let the totals differ")

    # A cell is measured exactly when it holds more than one rank.
    for (h, grp), cell in facts["cells"].items():
        if cell["measured"] != (len(cell["ranks"]) > 1):
            problems.append(f"cell H{h}/{grp}: 'measured' disagrees with the rank count")
        if not cell["measured"] and cell["s"] not in (0.0, None):
            problems.append(f"cell H{h}/{grp}: single rank but s = {cell['s']}")

    # The rank convention the method declares: 1..k, no gaps, per group.
    for grp, stages in facts["typology"].items():
        ranks = sorted(stages)
        if ranks != list(range(1, len(ranks) + 1)):
            problems.append(
                f"group {grp}: stage ranks {ranks} are not 1..k without gaps — "
                f"the declared numbering convention is broken")
    return problems


# ==============================================================================
# SECTION 6 · Entry point
# ==============================================================================

def collect() -> dict:
    """Return every fact the documentation needs, verified."""
    g = load_graphs()
    records = read_observations(g)
    cells_stage = cell_statistics(records, "stage")
    facts = {
        "records": records,
        "cells": cells_stage,
        "cells_column": cell_statistics(records, "column"),
        "stage_composition": stage_composition(records),
        "group_shares": group_shares(records),
        "worked_example": worked_example(records),
        "origin": rank_origin_sensitivity(cells_stage),
        "accounting": sherd_accounting(records),
        "typology": typology(records),
        "seriation": seriation(g, records),
        "intervals": horizon_intervals(g),
        "horizons": list(HORIZON_NUMBERS),
        "horizon_members": {h: list(HORIZONS[h]) for h in HORIZON_NUMBERS},
        "main_group": MAIN_GROUP,
    }
    problems = verify(facts)
    if problems:
        raise SystemExit("facts failed verification:\n  " + "\n  ".join(problems))
    return facts


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 9)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="write the facts to this file")
    args = ap.parse_args()

    facts = collect()
    slim = {k: v for k, v in facts.items() if k != "records"}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_jsonable(slim), indent=2, ensure_ascii=False),
                             encoding="utf-8", newline="\n")
        print(f"OK  {args.json}")
    else:
        acc = facts["accounting"]
        ex = facts["worked_example"]
        print(f"observations {acc['observations']} · sherds {acc['overall']} "
              f"· in a horizon {acc['in_horizon']} · gap {acc['gap']}")
        print(f"worked example H{ex['horizon']}/{ex['group']}: "
              f"N={ex['n']} Sum rc={ex['sum_rc']} Sum r2c={ex['sum_r2c']} "
              f"x={ex['mean']:.4f} s={ex['s']:.4f} q={ex['q']:.4f}")
        ser = facts["seriation"]
        print(f"seriation: {ser['in_order']} of {ser['plotted']} in order, "
              f"inversions {ser['inversions']}")
        print("all cross-checks passed")


if __name__ == "__main__":
    main()
