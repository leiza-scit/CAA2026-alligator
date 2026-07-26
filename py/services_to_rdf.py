"""Arretine service-type counts as an RDF layer alongside the Alligator graph.

The minigraph written by ``alligator_to_clean_rdf.py`` publishes the *result* of
the seriation — horizons, intervals, Allen relations — but not the evidence it
rests on. The sherd counts live in the workbook and nowhere else, so anyone
wanting to check the chronology has to leave the graph. This script puts them in.

It is a separate step, and writes a separate file, on purpose. The two layers
have different sources and different warrants: one is derived from the Alligator
output, the other from a workbook maintained by hand. Keeping them apart means
each can carry its own provenance, each can be cited on its own, and the
minigraph stays exactly what it claims to be. Neither file is ever modified in
place, so the step is idempotent — running it twice produces the same bytes.

Inputs
------
* root/src/ArretineDatedSitesServicesI_II.xlsx  (counts, columns B-J)
* root/output/arretine_sites_minigraph.ttl      (findspot IRIs to attach to)

Output
------
* root/output/arretine_services.ttl

Run it after ``alligator_to_clean_rdf.py`` and before ``build_sparql.py``:
the query page verifies the service queries against this file, so it has to
exist first.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

# This graph dates events BC, so it carries xsd:gYear literals with negative
# years. rdflib tries to map every literal onto a Python datetime.date, whose
# minimum year is 1 - the conversion raises and rdflib logs a full traceback per
# literal at WARNING. Dozens of them scroll past and look like a failure, while
# nothing is wrong: the literals keep their lexical form, and this script only
# reads labels and IRIs from the graph anyway. Silencing this one logger keeps
# real warnings visible. Same line, same reason, as in build_sparql.py.
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import RDF, RDFS, SKOS, XSD

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wd_paths                                              # noqa: E402
from horizons import PERCENT_LABEL_CORRECTIONS               # noqa: E402


# ==============================================================================
# SECTION 1 · Configuration
# ==============================================================================

XLSX_FILE = wd_paths.ROOT / "src" / "ArretineDatedSitesServicesI_II.xlsx"
BASE_GRAPH = wd_paths.OUTPUT / "arretine_sites_minigraph.ttl"
OUT_FILE = wd_paths.OUTPUT / "arretine_services.ttl"

SHEET_NAME = 0
SERVICE_COL_START = 1        # column B, 0-based
SERVICE_COL_END = 10         # column J inclusive, so the slice stop is 10

BASE = "http://leiza-scit.github.io/CAA2026-alligator/"

AE = Namespace(BASE)
AEVOC = Namespace(BASE + "vocabulary/")      # the service typology
AEOBS = Namespace(BASE + "observations/")    # one node per count
AEONT = Namespace(BASE + "ontology#")        # the four local properties
FSL = Namespace("http://fuzzy-sl.squirrel.link/ontology/")
PROV = Namespace("http://www.w3.org/ns/prov#")
DCT = Namespace("http://purl.org/dc/terms/")

PREFIXES = [
    ("aeont", AEONT), ("aeobs", AEOBS), ("aevoc", AEVOC),
    ("dct", DCT), ("fsl", FSL), ("prov", PROV),
    ("rdfs", RDFS), ("skos", SKOS), ("xsd", XSD),
]


# ==============================================================================
# SECTION 2 · The service typology
# ==============================================================================
# Two axes, deliberately, rather than nine flat types.
#
#   stage   Schrägrandteller -> Service Ia -> Ib -> Ic -> Service II
#   form    Tasse (cup) | Teller (plate)
#
# The stage axis is what the seriation orders and what the RGZM variance method
# groups over. The form axis is orthogonal to it. Whether a Tasse and a Teller
# of the same stage should also share a chronological rank is an open question
# — modelling the two as one flat list of nine would answer it by accident, and
# in the affirmative, which is precisely what should not happen silently.
#
# Column order below is the workbook's, which is the conventional order of the
# typology. It is recorded as aeont:typologicalRank so queries can sort by it,
# and it says nothing about the question above.

TASSE = "Tasse"
TELLER = "Teller"

# Three top concepts: the groups the RGZM within-group variance is computed over.
GROUPS = {
    "Schraegrandteller": {
        "de": "Schrägrandteller", "en": "Oblique-rim plate",
        "fr": "Assiette à bord oblique"},
    "ServiceI": {"de": "Service I", "en": "Service I", "fr": "Service I"},
    "ServiceII": {"de": "Service II", "en": "Service II", "fr": "Service II"},
}

# Stages, each under a group. A stage that is its own group has no sub-division.
STAGES = {
    "Schraegrandteller": ("Schraegrandteller", None),
    "ServiceIa": ("ServiceI", "Service Ia"),
    "ServiceIb": ("ServiceI", "Service Ib"),
    "ServiceIc": ("ServiceI", "Service Ic"),
    "ServiceII": ("ServiceII", None),
}

# The nine leaf types, in workbook column order: (slug, stage, form, labels).
# The workbook's own headers carry stray spaces, so the counts are read by
# position and the names are taken from here instead.
TYPES = [
    ("Schraegrandteller", "Schraegrandteller", TELLER,
     {"de": "Schrägrandteller", "en": "Oblique-rim plate",
      "fr": "Assiette à bord oblique"}),
    ("ServiceIa_Tasse", "ServiceIa", TASSE,
     {"de": "Service Ia Tasse", "en": "Service Ia cup",
      "fr": "Service Ia tasse"}),
    ("ServiceIa_Teller", "ServiceIa", TELLER,
     {"de": "Service Ia Teller", "en": "Service Ia plate",
      "fr": "Service Ia assiette"}),
    ("ServiceIb_Tasse", "ServiceIb", TASSE,
     {"de": "Service Ib Tasse", "en": "Service Ib cup",
      "fr": "Service Ib tasse"}),
    ("ServiceIb_Teller", "ServiceIb", TELLER,
     {"de": "Service Ib Teller", "en": "Service Ib plate",
      "fr": "Service Ib assiette"}),
    ("ServiceIc_Tasse", "ServiceIc", TASSE,
     {"de": "Service Ic Tasse", "en": "Service Ic cup",
      "fr": "Service Ic tasse"}),
    ("ServiceIc_Teller", "ServiceIc", TELLER,
     {"de": "Service Ic Teller", "en": "Service Ic plate",
      "fr": "Service Ic assiette"}),
    ("ServiceII_Tasse", "ServiceII", TASSE,
     {"de": "Service II Tasse", "en": "Service II cup",
      "fr": "Service II tasse"}),
    ("ServiceII_Teller", "ServiceII", TELLER,
     {"de": "Service II Teller", "en": "Service II plate",
      "fr": "Service II assiette"}),
]

FORMS = {
    TASSE: {"de": "Tasse", "en": "Cup", "fr": "Tasse"},
    TELLER: {"de": "Teller", "en": "Plate", "fr": "Assiette"},
}


# ==============================================================================
# SECTION 3 · Reading the inputs
# ==============================================================================

def read_counts(path: Path = XLSX_FILE) -> dict[str, dict[str, int]]:
    """Workbook label -> {type slug: count}, empty cells omitted.

    Columns are taken by position, not by header: the workbook's headers carry
    leading and trailing spaces that differ from column to column, and matching
    on them has already caused one silent mismatch in this project.
    """
    if not path.exists():
        sys.exit(f"  !!  workbook not found: {path}")

    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=0)
    block = raw.iloc[:, SERVICE_COL_START:SERVICE_COL_END]
    if block.shape[1] != len(TYPES):
        sys.exit(f"  !!  expected {len(TYPES)} service columns, "
                 f"found {block.shape[1]}")

    counts: dict[str, dict[str, int]] = {}
    for row, label in enumerate(raw.iloc[:, 0]):
        label = str(label).strip()
        cells = {}
        for col, (slug, *_rest) in enumerate(TYPES):
            value = block.iat[row, col]
            if pd.isna(value):
                continue
            if float(value) != int(value):
                sys.exit(f"  !!  {label} / {slug}: {value} is not a whole "
                         f"number of sherds")
            cells[slug] = int(value)
        counts[label] = cells
    return counts


def read_findspots(path: Path = BASE_GRAPH) -> dict[str, URIRef]:
    """Workbook label -> findspot IRI, read from the graph being enriched.

    The IRIs are the join. Reading them here rather than reconstructing them
    means this layer cannot invent a findspot the base graph does not have: a
    label that fails to resolve is reported, not quietly minted.
    """
    if not path.exists():
        sys.exit(f"  !!  base graph not found: {path}\n"
                 f"      Run alligator_to_clean_rdf.py first.")

    graph = Graph().parse(path, format="turtle")
    found: dict[str, URIRef] = {}
    for site in graph.subjects(RDF.type, FSL.Site):
        label = str(graph.value(site, RDFS.label))
        found[PERCENT_LABEL_CORRECTIONS.get(label, label)] = site
    return found


# ==============================================================================
# SECTION 4 · Building the layer
# ==============================================================================

def build_graph(counts, findspots) -> tuple[Graph, dict[str, int]]:
    """Assemble the whole layer as an rdflib graph.

    Built here even though Section 5 writes the Turtle by hand: the two are
    compared before anything is saved, so the hand-written file is checked
    against a serialisation that cannot have an escaping or a layout bug.
    """
    g = Graph()

    # --- the local properties, so the file explains its own vocabulary ------
    g.add((AEONT.ServiceCount, RDF.type, RDFS.Class))
    g.add((AEONT.ServiceCount, RDFS.label, Literal(
        "Service type count", lang="en")))
    g.add((AEONT.ServiceCount, RDFS.comment, Literal(
        "The number of Arretine sherds of one service type recorded at one "
        "findspot. One node per non-empty cell of the workbook.", lang="en")))

    for prop, label, comment, domain, rng in [
        (AEONT.atFindspot, "at findspot",
         "The findspot the count was recorded at.",
         AEONT.ServiceCount, FSL.Site),
        (AEONT.serviceType, "service type",
         "The service type counted, as a concept of the Arretine typology.",
         AEONT.ServiceCount, SKOS.Concept),
        (AEONT.sherdCount, "sherd count",
         "The count itself. Percentages are deliberately not stored: they "
         "follow from these counts, and a second copy of a derived number is "
         "a second thing to keep true.",
         AEONT.ServiceCount, XSD.integer),
        (AEONT.vesselForm, "vessel form",
         "Cup or plate. Orthogonal to the stage axis carried by skos:broader.",
         SKOS.Concept, SKOS.Concept),
        (AEONT.typologicalRank, "typological rank",
         "Position in the conventional order of the whole typology, as the "
         "workbook columns give it, 1 to 9. A display sort key.",
         SKOS.Concept, XSD.integer),
        (AEONT.columnRank, "column rank",
         "Rank within the concept's own group, numbering the group's sub-types "
         "1..k in column order, so a cup and a plate of the same stage are one "
         "step apart. The 'column' reading of the RGZM rank schemes; kept so "
         "the alternative can be re-checked, not because it is preferred.",
         SKOS.Concept, XSD.integer),
        (AEONT.stageRank, "stage rank",
         "Rank within the concept's own group, numbering stages rather than "
         "sub-types, so a cup and a plate of the same stage share a rank. The "
         "reading this project reports: a change of vessel form is not a "
         "chronological step, and treating it as one would be an artefact. "
         "The two readings agree closely in any case (r = 0.998).",
         SKOS.Concept, XSD.integer),
    ]:
        g.add((prop, RDF.type, RDF.Property))
        g.add((prop, RDFS.label, Literal(label, lang="en")))
        g.add((prop, RDFS.comment, Literal(comment, lang="en")))
        g.add((prop, RDFS.domain, domain))
        g.add((prop, RDFS.range, rng))

    # --- the concept scheme -------------------------------------------------
    scheme = AEVOC.services
    g.add((scheme, RDF.type, SKOS.ConceptScheme))
    for lang, text in [("en", "Arretine service typology"),
                       ("de", "Typologie der arretinischen Services"),
                       ("fr", "Typologie des services arétins")]:
        g.add((scheme, SKOS.prefLabel, Literal(text, lang=lang)))
    g.add((scheme, RDFS.comment, Literal(
        "Two axes: a stage sequence carried by skos:broader, and a vessel "
        "form carried by aeont:vesselForm. The three top concepts are the "
        "groups the within-group variance is computed over.", lang="en")))

    for form, labels in FORMS.items():
        concept = AEVOC[form]
        g.add((concept, RDF.type, SKOS.Concept))
        g.add((concept, SKOS.inScheme, scheme))
        for lang, text in labels.items():
            g.add((concept, SKOS.prefLabel, Literal(text, lang=lang)))

    for slug, labels in GROUPS.items():
        concept = AEVOC[slug]
        g.add((concept, RDF.type, SKOS.Concept))
        g.add((concept, SKOS.inScheme, scheme))
        g.add((concept, SKOS.topConceptOf, scheme))
        g.add((scheme, SKOS.hasTopConcept, concept))
        for lang, text in labels.items():
            g.add((concept, SKOS.prefLabel, Literal(text, lang=lang)))

    for slug, (group, label) in STAGES.items():
        if label is None:                 # the stage is its own group
            continue
        concept = AEVOC[slug]
        g.add((concept, RDF.type, SKOS.Concept))
        g.add((concept, SKOS.inScheme, scheme))
        g.add((concept, SKOS.prefLabel, Literal(label, lang="de")))
        g.add((concept, SKOS.broader, AEVOC[group]))
        g.add((AEVOC[group], SKOS.narrower, concept))

    # The two within-group rank readings, derived from the typology rather than
    # written out by hand, so they cannot fall out of step with TYPES. A group's
    # sub-types are ranked 1..k in column order; its stages are ranked in the
    # order they first appear, and every sub-type of a stage shares that rank.
    column_rank, stage_rank = {}, {}
    for group in GROUPS:
        members = [slug for slug, st, *_ in TYPES if STAGES[st][0] == group]
        stage_order, seen = [], set()
        for slug in members:
            st = next(s for sl, s, *_ in TYPES if sl == slug)
            if st not in seen:
                seen.add(st)
                stage_order.append(st)
        for i, slug in enumerate(members, start=1):
            st = next(s for sl, s, *_ in TYPES if sl == slug)
            column_rank[slug] = i
            stage_rank[slug] = stage_order.index(st) + 1

    for rank, (slug, stage, form, labels) in enumerate(TYPES, start=1):
        concept = AEVOC[slug]
        g.add((concept, RDF.type, SKOS.Concept))
        g.add((concept, SKOS.inScheme, scheme))
        for lang, text in labels.items():
            g.add((concept, SKOS.prefLabel, Literal(text, lang=lang)))
        # Schrägrandteller is a group with exactly one member: itself. Linking
        # it to itself with skos:broader would make the concept its own
        # ancestor, which SKOS treats as an error and which makes every
        # skos:broader* path match it twice - the counts come out doubled.
        if AEVOC[stage] != concept:
            g.add((concept, SKOS.broader, AEVOC[stage]))
            g.add((AEVOC[stage], SKOS.narrower, concept))
        g.add((concept, AEONT.vesselForm, AEVOC[form]))
        g.add((concept, AEONT.typologicalRank,
               Literal(rank, datatype=XSD.integer)))
        g.add((concept, AEONT.columnRank,
               Literal(column_rank[slug], datatype=XSD.integer)))
        g.add((concept, AEONT.stageRank,
               Literal(stage_rank[slug], datatype=XSD.integer)))

    # --- the observations ---------------------------------------------------
    tally = {"observations": 0, "sherds": 0, "findspots": 0, "unmatched": 0}
    for label in sorted(counts):
        site = findspots.get(label)
        if site is None:
            print(f"  !!  workbook findspot not in the base graph: {label}")
            tally["unmatched"] += 1
            continue
        if counts[label]:
            tally["findspots"] += 1
        for slug, count in counts[label].items():
            node = AEOBS[f"{str(site).rsplit('/', 1)[-1]}_{slug}"]
            g.add((node, RDF.type, AEONT.ServiceCount))
            g.add((node, AEONT.atFindspot, site))
            g.add((node, AEONT.serviceType, AEVOC[slug]))
            g.add((node, AEONT.sherdCount, Literal(count, datatype=XSD.integer)))
            tally["observations"] += 1
            tally["sherds"] += count

    # --- provenance ---------------------------------------------------------
    # The point of building this as a separate file: it can say where it came
    # from without that claim also covering the Alligator-derived triples.
    layer = AE["graph/arretine_services"]
    workbook = AE["source/ArretineDatedSitesServicesI_II.xlsx"]
    g.add((layer, RDF.type, PROV.Entity))
    g.add((layer, RDFS.label, Literal(
        "Arretine service-type counts", lang="en")))
    g.add((layer, DCT.description, Literal(
        "Sherd counts per findspot and service type, the evidence the "
        "seriation in arretine_sites_minigraph.ttl was computed from. Merge "
        "the two graphs to query across both.", lang="en")))
    g.add((layer, PROV.wasDerivedFrom, workbook))
    g.add((layer, PROV.wasDerivedFrom, AE["graph/arretine_sites_minigraph"]))
    g.add((workbook, RDF.type, PROV.Entity))
    g.add((workbook, RDFS.label, Literal(
        "ArretineDatedSitesServicesI_II.xlsx", lang="en")))

    return g, tally


# ==============================================================================
# SECTION 5 · Writing Turtle
# ==============================================================================
# Written by hand rather than with Graph.serialize so the file has a stable
# order and a readable shape: a graph rebuilt from unchanged inputs must be
# byte-identical, or every run shows up as a change in git.

# A local name a prefixed form may use. Turtle allows more than this, but only
# with escaping; anything outside it is written as a full IRI instead, which is
# always legal and never needs an escape.
_SIMPLE_LOCAL = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)*\Z")


def _term(node) -> str:
    """A term in Turtle, using the prefixes declared at the top of the file."""
    if isinstance(node, Literal):
        if node.datatype == XSD.integer:
            return str(int(node))
        text = (str(node).replace("\\", "\\\\").replace('"', '\\"')
                .replace("\n", "\\n"))
        return f'"{text}"@{node.language}' if node.language else f'"{text}"'
    text = str(node)
    for prefix, namespace in [*PREFIXES, ("rdf", RDF), ("ae", AE)]:
        if text.startswith(str(namespace)):
            local = text[len(str(namespace)):]
            if _SIMPLE_LOCAL.match(local):
                return f"{prefix}:{local}"
    return f"<{text}>"


def write_turtle(g: Graph, path: Path, tally: dict[str, int]) -> None:
    lines = [
        "# Arretine service-type counts — a layer over arretine_sites_minigraph.ttl.",
        "#",
        "# Generated by py/services_to_rdf.py from src/ArretineDatedSitesServicesI_II.xlsx.",
        "# Do not edit by hand: rerunning the script overwrites this file.",
        "#",
        f"# {tally['observations']} counts over {tally['findspots']} findspots, "
        f"{tally['sherds']} sherds in total.",
        "",
        "@prefix ae: <http://leiza-scit.github.io/CAA2026-alligator/> .",
    ]
    lines += [f"@prefix {p}: <{n}> ." for p, n in PREFIXES]
    lines += ["@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .", ""]

    for subject in sorted(set(g.subjects()), key=str):
        pairs = sorted(((_term(p), _term(o)) for p, o in g.predicate_objects(subject)),
                       key=lambda pair: (pair[0] != "rdf:type", pair))
        lines.append(f"{_term(subject)}")
        for i, (p, o) in enumerate(pairs):
            end = " ." if i == len(pairs) - 1 else " ;"
            lines.append(f"    {p} {o}{end}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ==============================================================================
# SECTION 6 · Verification
# ==============================================================================

def verify(g: Graph, path: Path) -> None:
    """Read the file back and check it says exactly what was built.

    Hand-written Turtle buys a stable byte order at the cost of doing the
    escaping itself, so the file is parsed again and compared with the graph it
    was written from. An escaping bug shows up here rather than in a query that
    quietly returns one row too few.
    """
    reparsed = Graph().parse(path, format="turtle")
    if not isomorphic(reparsed, g):
        only_written = set(g) - set(reparsed)
        only_read = set(reparsed) - set(g)
        for triple in list(only_written)[:5]:
            print(f"  !!  lost in writing: {triple}")
        for triple in list(only_read)[:5]:
            print(f"  !!  appeared on reading: {triple}")
        sys.exit("  !!  the written file does not match the graph built.")
    print(f"  OK  re-parsed and identical ({len(reparsed)} triples)")


# ==============================================================================
# SECTION 7 · Entry point
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print("Service-type counts -> RDF layer (xlsx -> output/)")
    print("=" * 60)

    counts = read_counts()
    print(f"  ..  {XLSX_FILE.name}: {len(counts)} findspots")

    findspots = read_findspots()
    print(f"  ..  {BASE_GRAPH.name}: {len(findspots)} findspot IRIs")

    graph, tally = build_graph(counts, findspots)
    if tally["unmatched"]:
        sys.exit(f"  !!  {tally['unmatched']} findspot(s) could not be matched; "
                 f"add them to PERCENT_LABEL_CORRECTIONS in horizons.py.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_turtle(graph, OUT_FILE, tally)
    print(f"  OK  output/{OUT_FILE.name}  ({len(graph)} triples, "
          f"{tally['observations']} counts, {tally['sherds']} sherds)")

    verify(graph, OUT_FILE)
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()