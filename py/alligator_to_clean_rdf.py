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
6. Serialise the graph as Turtle and save all outputs

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

    # --- Alligator temporal data ---
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
# SECTION 11 · Main Entry Point
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

        # Step 6 – RDF conversion
        convert_to_rdf(merged_df)

        print("\n" + "=" * 60)
        print("Done!")
        print(f"Report saved: {log_file}")
        print("=" * 60)

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
