"""
Alligator Event Mapping Script
===============================
Builds a mapping table between rdfs:label values and Event URIs from Alligator output files.
Converts the enriched CSV data into a full RDF graph using CIDOC-CRM and GeoSPARQL vocabularies.

Pipeline
--------
1. Load Alligator events from a Turtle (.ttl) file
2. Load findspot data from a CSV file
3. Create a label-based mapping table (exact + fuzzy matching)
4. Merge mapped event data into the findspot table
5. Convert the enriched table into an RDF graph
6. Load additional events from MoreEvents.csv and add them as OWL-Time + CIDOC-CRM nodes
7. Serialise the combined graph as Turtle and save all outputs

Note on notebook conversion
----------------------------
This script is structured to be straightforward to convert into a Jupyter notebook.
Each top-level section (marked with # ── SECTION ──) maps to one or more notebook cells.
The `main()` function calls steps sequentially and can be replaced by individual cells
that run each step interactively. The Logger/sys.stdout redirect is only activated inside
`main()` and does not affect interactive use.
"""

# ==============================================================================
# SECTION 1 · Imports
# ==============================================================================

import sys
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import DC, XSD


# ==============================================================================
# SECTION 2 · Configuration
# ==============================================================================
# All paths and namespace constants are defined here so that a notebook can
# override individual values in a dedicated "Configuration" cell before running
# the rest of the pipeline.

# ---------------------------------------------------------------------------
# Directory layout
# Script lives in root/py/; input data in root/data/; outputs in root/output/
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------
TTL_FILE = DATA_DIR / "ArretineDatedSitesServicesI_II.ttl"
CSV_FILE = DATA_DIR / "ArretineDatedSitesServicesI_II_findspots.csv"
MORE_EVENTS_FILE = DATA_DIR / "MoreEvents.csv"

# ---------------------------------------------------------------------------
# Namespaces – Alligator input
# ---------------------------------------------------------------------------
ALLIGATOR = Namespace("http://archaeology.link/ontology#")
AE = Namespace("http://leiza-scit.github.io/CAA2026-alligator/")

# ---------------------------------------------------------------------------
# Namespaces – RDF output
# ---------------------------------------------------------------------------
GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")
SF = Namespace("http://www.opengis.net/ont/sf#")
FSL = Namespace("http://fuzzy-sl.squirrel.link/ontology/")
PLEIADES_VOCAB = Namespace("https://pleiades.stoa.org/places/vocab#")
WIKIDATA = Namespace("http://www.wikidata.org/entity/")
PLEIADES_PLACE = Namespace("https://pleiades.stoa.org/places/")
LADO = Namespace("http://archaeology.link/ontology#")
CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
AE_SITES = Namespace("http://leiza-scit.github.io/CAA2026-alligator/")
AE_COLLECTIONS = Namespace("http://leiza-scit.github.io/CAA2026-alligator/collections/")
OWL_TIME = Namespace("http://www.w3.org/2006/time#")

# ---------------------------------------------------------------------------
# Feature Collection URI
# ---------------------------------------------------------------------------
FEATURE_COLLECTION_URI = AE_COLLECTIONS["arretine_sites"]

# Columns to carry over from the mapping table into the merged findspot table
MAPPING_COLS = [
    "csv_label",
    "event_uri",
    "event_identifier",
    "estimatedstart",
    "estimatedend",
    "cax",
    "cay",
    "caz",
    "startfixed",
    "endfixed",
    "nfsn",
    "nfen",
]


# ==============================================================================
# SECTION 3 · Helper Utilities
# ==============================================================================


class Logger:
    """Tees stdout to both the terminal and a log file simultaneously.

    Used only inside `main()` for script runs. Not needed in a notebook
    context – Jupyter captures cell output natively.
    """

    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def sanitize_id(label: str) -> str:
    """Produces a clean, URI-safe identifier string from a human-readable label."""
    id_str = label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    id_str = id_str.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return id_str


# rdflib delegates xsd:gYear validation to isodate, which does not handle
# negative (BCE) years and raises ISO8601Error for values like "-0015".
#
# Fix: monkey-patch rdflib.term._castLexicalToPython to intercept gYear
# calls and return the lexical string unchanged. This is more robust than
# manipulating the internal _toPythonMapping dict, whose name varies across
# rdflib versions. rdflib only uses the return value for in-memory Python
# objects; the serialised Turtle output is always taken from the lexical form
# and is therefore unaffected.
try:
    import rdflib.term as _rdflib_term

    _original_cast = _rdflib_term._castLexicalToPython  # type: ignore[attr-defined]
    _GYEAR_URI = str(XSD.gYear)

    def _patched_cast(lexical: str, datatype):  # type: ignore[no-untyped-def]
        if str(datatype) == _GYEAR_URI:
            return lexical  # Return the string as-is; isodate cannot handle BCE years
        return _original_cast(lexical, datatype)

    _rdflib_term._castLexicalToPython = _patched_cast  # type: ignore[attr-defined]
except Exception:
    pass  # Harmless if the internal API changes in a future rdflib version


def _year_to_xsd_gyear(year: int) -> str:
    """Convert an integer year to an XSD gYear lexical value.

    XSD gYear uses astronomical year numbering:
      1 BCE  →  0000
      2 BCE  → -0001
      1 CE   →  0001

    Parameters
    ----------
    year : int
        Astronomical year (negative = BCE, 0 = 1 BCE).

    Returns
    -------
    str
        Zero-padded XSD gYear string, e.g. "-0015" or "0009".
    """
    # Historian's BCE years (e.g. -15 meaning "15 BCE") are already in
    # astronomical notation in this dataset, so no offset adjustment needed.
    if year < 0:
        return f"-{abs(year):04d}"
    return f"{year:04d}"


# ==============================================================================
# SECTION 4 · Output Directory Setup
# ==============================================================================


def setup_output_dir() -> Path:
    """Clears the output directory (if it exists) and recreates it fresh.

    Called once at the start of a pipeline run to ensure a clean state.
    In a notebook, run this cell only when a full reset is desired.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ==============================================================================
# SECTION 5 · Load Alligator Events (TTL)
# ==============================================================================


def load_alligator_events(ttl_path: Path) -> dict:
    """Load a Turtle file and extract all Alligator events with their labels.

    Parameters
    ----------
    ttl_path : Path
        Path to the Alligator output Turtle file.

    Returns
    -------
    dict
        Mapping of rdfs:label → dict of event URI and all associated properties.
    """
    print(f"Loading TTL file: {ttl_path}")

    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path.absolute()}")

    g = Graph()
    g.parse(str(ttl_path.absolute()), format="turtle")

    events = {}

    for event_uri in g.subjects(RDF.type, ALLIGATOR.event):
        label = g.value(event_uri, RDFS.label)

        if not label:
            continue

        label_str = str(label)
        events[label_str] = {
            "uri": str(event_uri),
            "identifier": str(
                g.value(
                    event_uri, Namespace("http://purl.org/dc/elements/1.1/").identifier
                )
                or ""
            ),
            "label": label_str,
            "estimatedstart": str(g.value(event_uri, ALLIGATOR.estimatedstart) or ""),
            "estimatedend": str(g.value(event_uri, ALLIGATOR.estimatedend) or ""),
            "cax": str(g.value(event_uri, ALLIGATOR.cax) or ""),
            "cay": str(g.value(event_uri, ALLIGATOR.cay) or ""),
            "caz": str(g.value(event_uri, ALLIGATOR.caz) or ""),
            "startfixed": str(g.value(event_uri, ALLIGATOR.startfixed) or ""),
            "endfixed": str(g.value(event_uri, ALLIGATOR.endfixed) or ""),
            "nfsn": str(g.value(event_uri, ALLIGATOR.nfsn) or ""),
            "nfen": str(g.value(event_uri, ALLIGATOR.nfen) or ""),
        }

    print(f"✓ {len(events)} events found")
    return events


# ==============================================================================
# SECTION 6 · Load Findspot Data (CSV)
# ==============================================================================


def load_findspots_csv(csv_path: Path) -> pd.DataFrame:
    """Load the findspot CSV file into a Pandas DataFrame.

    Rows with inconsistent column counts are reported as warnings rather
    than raising errors.

    Parameters
    ----------
    csv_path : Path
        Path to the findspot CSV file.

    Returns
    -------
    pd.DataFrame
        One row per findspot.
    """
    print(f"Loading CSV file: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.absolute()}")

    df = pd.read_csv(csv_path, on_bad_lines="warn")
    print(f"✓ {len(df)} findspots loaded")
    return df


# ==============================================================================
# SECTION 7 · Create Mapping Table
# ==============================================================================


def create_mapping_table(events: dict, findspots_df: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping table between TTL events and CSV findspots via label matching.

    Matching strategy
    -----------------
    1. Exact match  – label strings are identical.
    2. Fuzzy match  – one label is a prefix of the other
                      (e.g. "Nijmegen, Lager" ↔ "Nijmegen, Lager (Haalebos)").

    Parameters
    ----------
    events       : dict        Output of `load_alligator_events`.
    findspots_df : pd.DataFrame Output of `load_findspots_csv`.

    Returns
    -------
    pd.DataFrame
        One row per findspot label with match type and associated event properties.
    """
    print("\nBuilding mapping table...")

    # Template for rows where no match is found
    _empty_row = dict.fromkeys(
        [
            "ttl_label",
            "event_uri",
            "event_identifier",
            "estimatedstart",
            "estimatedend",
            "cax",
            "cay",
            "caz",
            "startfixed",
            "endfixed",
            "nfsn",
            "nfen",
        ],
        None,
    )

    mapping_data = []

    for csv_label in findspots_df["label"]:
        # --- Exact match ---
        match = events.get(csv_label)
        match_type = "exact" if match else None

        # --- Fuzzy prefix match (fallback) ---
        if not match:
            for ttl_label, event_data in events.items():
                if csv_label.startswith(ttl_label) or ttl_label.startswith(csv_label):
                    match = event_data
                    match_type = "fuzzy"
                    break

        if match:
            row = {
                "csv_label": csv_label,
                "ttl_label": match["label"],
                "event_uri": match["uri"],
                "event_identifier": match["identifier"],
                "match_type": match_type,
                "estimatedstart": match["estimatedstart"],
                "estimatedend": match["estimatedend"],
                "cax": match["cax"],
                "cay": match["cay"],
                "caz": match["caz"],
                "startfixed": match["startfixed"],
                "endfixed": match["endfixed"],
                "nfsn": match["nfsn"],
                "nfen": match["nfen"],
            }
        else:
            row = {"csv_label": csv_label, "match_type": "no_match", **_empty_row}

        mapping_data.append(row)

    mapping_df = pd.DataFrame(mapping_data)

    # Summary statistics
    counts = mapping_df["match_type"].value_counts()
    print(f"\n✓ Mapping complete:")
    print(f"  Exact matches : {counts.get('exact',    0)}")
    print(f"  Fuzzy matches : {counts.get('fuzzy',    0)}")
    print(f"  No match      : {counts.get('no_match', 0)}")

    unmatched = mapping_df[mapping_df["match_type"] == "no_match"]["csv_label"].tolist()
    if unmatched:
        print(f"\n  ⚠ Unmatched findspots ({len(unmatched)}):")
        for label in unmatched:
            print(f"    · {label}")

    return mapping_df


# ==============================================================================
# SECTION 8 · Merge Findspot & Event Data
# ==============================================================================


def merge_findspots_and_events(
    findspots_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join mapped event properties back onto the original findspot table.

    Uses the site label as the join key. Unmatched rows retain NaN for all
    event columns (left join).

    Parameters
    ----------
    findspots_df : pd.DataFrame  Output of `load_findspots_csv`.
    mapping_df   : pd.DataFrame  Output of `create_mapping_table`.

    Returns
    -------
    pd.DataFrame
        Enriched findspot table with event columns appended.
    """
    merged_df = findspots_df.merge(
        mapping_df[MAPPING_COLS],
        left_on="label",
        right_on="csv_label",
        how="left",
    ).drop(columns="csv_label")

    return merged_df


# ==============================================================================
# SECTION 9 · RDF Graph Utilities
# ==============================================================================


def create_rdf_graph() -> Graph:
    """Create a new RDF graph with all required namespace bindings."""
    g = Graph()
    g.bind("geosparql", GEOSPARQL)
    g.bind("sf", SF)
    g.bind("fsl", FSL)
    g.bind("pleiades", PLEIADES_VOCAB)
    g.bind("wikidata", WIKIDATA)
    g.bind("pleiadesplace", PLEIADES_PLACE)
    g.bind("lado", LADO)
    g.bind("crm", CRM)
    g.bind("ae", AE_SITES)
    g.bind("aecol", AE_COLLECTIONS)
    g.bind("rdfs", RDFS)
    g.bind("time", OWL_TIME)
    return g


def add_site_to_graph(
    g: Graph,
    row: pd.Series,
    site_uri: URIRef,
    has_event: bool = False,
):
    """Add a single findspot to the RDF graph.

    Parameters
    ----------
    g         : Graph    RDF graph to write into.
    row       : Series   DataFrame row containing site data.
    site_uri  : URIRef   URI for the site (Alligator event URI or fallback).
    has_event : bool     Whether this site has a matched Alligator event.
    """
    # --- Type assertions ---
    g.add((site_uri, RDF.type, CRM.E53_Place))
    g.add((site_uri, RDF.type, FSL.Site))
    g.add((site_uri, RDF.type, FSL.ArchaeologicalSite))
    g.add((site_uri, RDF.type, PLEIADES_VOCAB.Place))

    if has_event:
        g.add((site_uri, RDF.type, ALLIGATOR.event))
        g.add((site_uri, RDF.type, URIRef("http://www.w3.org/2006/time#Interval")))

    # --- Label ---
    g.add((site_uri, RDFS.label, Literal(row["label"], lang="en")))

    # --- DC identifier (events only) ---
    if (
        has_event
        and pd.notna(row.get("event_identifier"))
        and str(row["event_identifier"]).strip()
    ):
        g.add((site_uri, DC.identifier, Literal(row["event_identifier"])))

    # --- GeoSPARQL geometry ---
    if pd.notna(row["wkt"]) and row["wkt"].strip():
        geom_uri = URIRef(str(site_uri) + "_geom")
        wkt_literal = Literal(
            f"<http://www.opengis.net/def/crs/EPSG/0/4326> {row['wkt'].strip()}",
            datatype=GEOSPARQL.wktLiteral,
        )
        g.add((site_uri, GEOSPARQL.hasGeometry, geom_uri))
        g.add((geom_uri, RDF.type, SF.Point))
        g.add((geom_uri, GEOSPARQL.asWKT, wkt_literal))
        g.add(
            (
                geom_uri,
                FSL.certaintyDesc,
                Literal("Mapping done by Allard Mees.", lang="en"),
            )
        )

    # --- Wikidata link ---
    if pd.notna(row["wikidata"]):
        wikidata_id = str(row["wikidata"]).strip()
        if wikidata_id.startswith("Q") and wikidata_id[1:].isdigit():
            g.add((site_uri, LADO.wikidata, WIKIDATA[wikidata_id]))
        elif wikidata_id:
            print(
                f"  ⚠ Invalid Wikidata ID ignored: '{wikidata_id}' for {row['label']}"
            )

    # --- Pleiades link ---
    if pd.notna(row["pleiades"]):
        try:
            pleiades_id = (
                str(int(row["pleiades"]))
                if isinstance(row["pleiades"], (int, float))
                else str(row["pleiades"]).strip()
            )
            if pleiades_id and pleiades_id.isdigit():
                g.add((site_uri, LADO.pleiades, PLEIADES_PLACE[pleiades_id]))
            elif pleiades_id:
                print(
                    f"  ⚠ Invalid Pleiades ID ignored: '{pleiades_id}' for {row['label']}"
                )
        except (ValueError, TypeError):
            print(f"  ⚠ Pleiades ID conversion error for {row['label']}")

    # --- Alligator temporal data (lado vocabulary, kept for compatibility) ---
    if pd.notna(row.get("estimatedstart")):
        g.add(
            (
                site_uri,
                ALLIGATOR.estimatedstart,
                Literal(row["estimatedstart"], datatype=XSD.decimal),
            )
        )
    if pd.notna(row.get("estimatedend")):
        g.add(
            (
                site_uri,
                ALLIGATOR.estimatedend,
                Literal(row["estimatedend"], datatype=XSD.decimal),
            )
        )

    # --- OWL-Time Interval pattern (only for Alligator events with temporal data) ---
    # Mirrors the lado:estimatedstart/end values as a standards-compliant
    # time:Interval with reified time:Instant nodes carrying xsd:gYear literals.
    # The estimatedstart/end values are Alligator's centroid years (decimals);
    # we round to the nearest integer for the gYear representation.
    if has_event:
        has_start = pd.notna(row.get("estimatedstart"))
        has_end = pd.notna(row.get("estimatedend"))

        if has_start or has_end:
            g.add((site_uri, RDF.type, OWL_TIME.Interval))

        if has_start:
            begin_uri = URIRef(str(site_uri) + "_begin")
            g.add((site_uri, OWL_TIME.hasBeginning, begin_uri))
            g.add((begin_uri, RDF.type, OWL_TIME.Instant))
            g.add(
                (
                    begin_uri,
                    OWL_TIME.inXSDgYear,
                    Literal(
                        _year_to_xsd_gyear(round(float(row["estimatedstart"]))),
                        datatype=XSD.gYear,
                    ),
                )
            )

        if has_end:
            end_uri = URIRef(str(site_uri) + "_end")
            g.add((site_uri, OWL_TIME.hasEnd, end_uri))
            g.add((end_uri, RDF.type, OWL_TIME.Instant))
            g.add(
                (
                    end_uri,
                    OWL_TIME.inXSDgYear,
                    Literal(
                        _year_to_xsd_gyear(round(float(row["estimatedend"]))),
                        datatype=XSD.gYear,
                    ),
                )
            )

    # --- Alligator coordinates (cax / cay / caz) ---
    for coord in ("cax", "cay", "caz"):
        if pd.notna(row.get(coord)) and str(row[coord]).strip():
            g.add(
                (site_uri, ALLIGATOR[coord], Literal(row[coord], datatype=XSD.decimal))
            )

    # --- Alligator fixed flags ---
    for flag in ("startfixed", "endfixed"):
        if pd.notna(row.get(flag)) and str(row[flag]).strip():
            g.add((site_uri, ALLIGATOR[flag], Literal(row[flag], datatype=XSD.boolean)))

    # --- Alligator neighbourhood properties ---
    for prop in ("nfsn", "nfen"):
        if pd.notna(row.get(prop)) and str(row[prop]).strip():
            g.add((site_uri, ALLIGATOR[prop], Literal(row[prop])))


def create_feature_collection(g: Graph, site_uris: list):
    """Declare a GeoSPARQL FeatureCollection and link all site URIs to it.

    GeoSPARQL supports both directions of the membership relationship,
    so both geosparql:memberOf and geosparql:hasFeature are asserted.
    """
    g.add((FEATURE_COLLECTION_URI, RDF.type, GEOSPARQL.FeatureCollection))
    g.add(
        (
            FEATURE_COLLECTION_URI,
            RDFS.label,
            Literal("Arretine Dated Sites Collection", lang="en"),
        )
    )
    g.add(
        (
            FEATURE_COLLECTION_URI,
            RDFS.comment,
            Literal(
                "Collection of archaeological sites with Arretine pottery finds",
                lang="en",
            ),
        )
    )

    for site_uri in site_uris:
        g.add((site_uri, GEOSPARQL.memberOf, FEATURE_COLLECTION_URI))
        g.add((FEATURE_COLLECTION_URI, GEOSPARQL.hasFeature, site_uri))

    print(f"✓ FeatureCollection created with {len(site_uris)} features")


# ==============================================================================
# SECTION 10 · RDF Conversion
# ==============================================================================


def convert_to_rdf(merged_df: pd.DataFrame) -> Graph:
    """Convert the enriched findspot DataFrame into an RDF graph.

    Sites with a matched Alligator event reuse the event URI directly.
    Unmatched sites receive a fallback URI derived from their label.

    Parameters
    ----------
    merged_df : pd.DataFrame  Output of `merge_findspots_and_events`.

    Returns
    -------
    Graph
        The populated rdflib Graph object (also serialised to disk as Turtle).
    """
    print("\n" + "=" * 60)
    print("RDF Conversion")
    print("=" * 60)

    g = create_rdf_graph()
    site_uris = []
    sites_with_events = 0
    sites_without_events = 0

    for _, row in merged_df.iterrows():
        event_uri_str = str(row.get("event_uri", "")).strip()

        if (
            pd.notna(row.get("event_uri"))
            and event_uri_str
            and event_uri_str.lower() != "nan"
        ):
            # Use the Alligator event URI directly
            site_uri = URIRef(event_uri_str)
            has_event = True
            sites_with_events += 1
        else:
            # Generate a fallback URI from the site label
            site_uri = AE_SITES[f"site_{sanitize_id(row['label'])}"]
            has_event = False
            sites_without_events += 1

        site_uris.append(site_uri)
        add_site_to_graph(g, row, site_uri, has_event)

    print(f"\n✓ {len(site_uris)} sites added to graph")
    print(f"  With event URI : {sites_with_events}")
    print(f"  Fallback URI   : {sites_without_events}")

    create_feature_collection(g, site_uris)

    # Serialise as Turtle
    output_file = OUTPUT_DIR / "arretine_sites_minigraph.ttl"
    g.serialize(destination=str(output_file), format="turtle")
    print(f"\n✓ RDF graph saved: {output_file}")
    print(f"\nGraph statistics:")
    print(f"  Triples total      : {len(g)}")
    print(f"  Sites              : {len(site_uris)}")
    print(f"  FeatureCollections : 1")

    return g


# ==============================================================================
# SECTION 11 · Load Additional Events (MoreEvents.csv)
# ==============================================================================
# These are historically significant events (e.g. military campaigns) that are
# not part of the Alligator findspot pipeline but should appear in the same RDF
# graph as temporal context. Each row carries a label, a start year, and an end
# year (negative values = BCE).


def load_more_events(csv_path: Path) -> pd.DataFrame:
    """Load the supplementary events CSV into a Pandas DataFrame.

    Expected columns
    ----------------
    label : str   Human-readable event name.
    start : int   Start year (negative = BCE).
    end   : int   End year   (negative = BCE).

    Parameters
    ----------
    csv_path : Path
        Path to MoreEvents.csv.

    Returns
    -------
    pd.DataFrame
        One row per event, with whitespace stripped from the label column.
    """
    print(f"Loading additional events: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"MoreEvents file not found: {csv_path.absolute()}")

    df = pd.read_csv(csv_path)

    # Strip accidental whitespace from the start/end columns (e.g. "- 9" → "-9")
    df["start"] = pd.to_numeric(
        df["start"].astype(str).str.replace(r"\s+", "", regex=True), errors="coerce"
    )
    df["end"] = pd.to_numeric(
        df["end"].astype(str).str.replace(r"\s+", "", regex=True), errors="coerce"
    )
    df["label"] = df["label"].str.strip()

    print(f"✓ {len(df)} additional events loaded")
    return df


# ==============================================================================
# SECTION 12 · Add MoreEvents to the RDF Graph
# ==============================================================================
# Modelling pattern
# -----------------
# Each event is typed as:
#   crm:E7_Activity   – CIDOC-CRM class for intentional human activities
#                       (campaigns are goal-directed, so E7 is preferred over
#                        the more abstract E5_Event)
#   time:Interval     – OWL-Time interval to carry temporal boundaries
#
# Temporal boundaries use the OWL-Time reified pattern:
#   <event>  time:hasBeginning  <event_begin>
#   <event>  time:hasEnd        <event_end>
#   <event_begin/end>  rdf:type          time:Instant
#   <event_begin/end>  time:inXSDgYear   "YYYY"^^xsd:gYear
#
# xsd:gYear uses the proleptic Gregorian calendar. Negative years are
# serialised as "-0014" (note: XSD gYear uses astronomical year numbering,
# so 1 BCE = 0000, 2 BCE = -0001, etc.).
# The rdflib patch and _year_to_xsd_gyear helper are defined in Section 3
# so they are available to both this section and add_site_to_graph (Section 9).


def add_more_events_to_graph(g: Graph, more_events_df: pd.DataFrame) -> list:
    """Add supplementary events from MoreEvents.csv to an existing RDF graph.

    Each event receives:
    - rdf:type  crm:E7_Activity  (CIDOC-CRM intentional activity)
    - rdf:type  time:Interval    (OWL-Time temporal interval)
    - rdfs:label
    - time:hasBeginning / time:hasEnd  →  reified time:Instant nodes
      with time:inXSDgYear literals

    Parameters
    ----------
    g               : Graph        The graph to write into (modified in place).
    more_events_df  : pd.DataFrame Output of `load_more_events`.

    Returns
    -------
    list
        URIRefs of all event nodes added.
    """
    print("\nAdding supplementary events to graph...")

    event_uris = []

    for _, row in more_events_df.iterrows():
        event_id = sanitize_id(row["label"])
        event_uri = AE_SITES[f"event_{event_id}"]
        event_uris.append(event_uri)

        # --- Type assertions ---
        g.add((event_uri, RDF.type, CRM.E7_Activity))  # CIDOC-CRM: intentional activity
        g.add((event_uri, RDF.type, OWL_TIME.Interval))  # OWL-Time: temporal interval

        # --- Label ---
        g.add((event_uri, RDFS.label, Literal(row["label"], lang="en")))

        # --- Temporal boundaries (OWL-Time reified pattern) ---
        if pd.notna(row["start"]):
            begin_uri = URIRef(str(event_uri) + "_begin")
            g.add((event_uri, OWL_TIME.hasBeginning, begin_uri))
            g.add((begin_uri, RDF.type, OWL_TIME.Instant))
            g.add(
                (
                    begin_uri,
                    OWL_TIME.inXSDgYear,
                    Literal(_year_to_xsd_gyear(int(row["start"])), datatype=XSD.gYear),
                )
            )

        if pd.notna(row["end"]):
            end_uri = URIRef(str(event_uri) + "_end")
            g.add((event_uri, OWL_TIME.hasEnd, end_uri))
            g.add((end_uri, RDF.type, OWL_TIME.Instant))
            g.add(
                (
                    end_uri,
                    OWL_TIME.inXSDgYear,
                    Literal(_year_to_xsd_gyear(int(row["end"])), datatype=XSD.gYear),
                )
            )

    print(f"✓ {len(event_uris)} supplementary events added to graph")
    return event_uris


# ==============================================================================
# SECTION 13 · Main Entry Point
# ==============================================================================
# When converting to a notebook, replace this function with one cell per step.
# The Logger/sys.stdout redirect is intentionally confined here and should be
# omitted in a notebook (Jupyter handles cell output natively).


def main():
    """Run the full pipeline end to end."""
    setup_output_dir()

    log_file = OUTPUT_DIR / "report.txt"
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print("=" * 60)
        print("Alligator Event Mapping")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Step 1 – Load TTL events
        events = load_alligator_events(TTL_FILE)

        # Step 2 – Load findspot CSV
        findspots_df = load_findspots_csv(CSV_FILE)

        # Step 3 – Build mapping table
        mapping_df = create_mapping_table(events, findspots_df)

        # Step 4 – Save mapping table
        mapping_output = OUTPUT_DIR / "event_mapping.csv"
        mapping_df.to_csv(mapping_output, index=False)
        print(f"\n✓ Mapping table saved: {mapping_output}")

        # Step 5 – Merge findspot + event data
        merged_df = merge_findspots_and_events(findspots_df, mapping_df)
        merged_output = OUTPUT_DIR / "findspots_with_events.csv"
        merged_df.to_csv(merged_output, index=False)
        print(f"✓ Enriched findspot table saved: {merged_output}")

        # Step 6 – RDF conversion of findspot data
        rdf_graph = convert_to_rdf(merged_df)

        # Step 7 – Load and integrate supplementary events (MoreEvents.csv)
        more_events_df = load_more_events(MORE_EVENTS_FILE)
        add_more_events_to_graph(rdf_graph, more_events_df)

        # Step 8 – Re-serialise the combined graph (overwrites the Step 6 file)
        output_file = OUTPUT_DIR / "arretine_sites_minigraph.ttl"
        rdf_graph.serialize(destination=str(output_file), format="turtle")
        print(f"\n✓ Combined RDF graph saved: {output_file}")
        print(f"  Triples total: {len(rdf_graph)}")

        print("\n" + "=" * 60)
        print("Done!")
        print(f"Report saved: {log_file}")
        print("=" * 60)

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
