"""
Alligator Event Mapping Script
Erstellt eine Mapping-Tabelle zwischen rdfs:label und Event-URIs aus Alligator-Ausgaben.
Konvertiert die CSV-Daten dann in einen vollständigen RDF-Graph mit CIDOC-CRM und GeoSPARQL.
"""

import pandas as pd
from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD, DC
from pathlib import Path
import sys
from datetime import datetime
import shutil

# Pfade definieren (relativ zum Script-Verzeichnis)
# Script liegt in root/py/, Daten in root/data/
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

TTL_FILE = DATA_DIR / "ArretineDatedSitesServicesI_II.ttl"
CSV_FILE = DATA_DIR / "ArretineDatedSitesServicesI_II_findspots.csv"

# Namespaces für Alligator
ALLIGATOR = Namespace("http://archaeology.link/ontology#")
AE = Namespace("http://leiza-scit.github.io/CAA2026-alligator/")

# Namespaces für RDF-Ausgabe
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

# Feature Collection URI
FEATURE_COLLECTION_URI = AE_COLLECTIONS["arretine_sites"]


# Logger-Klasse für Terminal + Datei
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_output_dir():
    """
    Bereitet Output-Verzeichnis vor: leert es und erstellt es neu.
    """
    if OUTPUT_DIR.exists():
        # Verzeichnis leeren
        shutil.rmtree(OUTPUT_DIR)

    # Neu erstellen
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return OUTPUT_DIR


def load_alligator_events(ttl_path):
    """
    Lädt die TTL-Datei und extrahiert alle Alligator Events mit ihren Labels.

    Returns:
        dict: Mapping von Label zu Event-URI und allen Properties
    """
    print(f"Lade TTL-Datei: {ttl_path}")

    # Prüfen ob Datei existiert
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL-Datei nicht gefunden: {ttl_path.absolute()}")

    g = Graph()
    # Absoluten Pfad verwenden und in URI konvertieren
    g.parse(str(ttl_path.absolute()), format="turtle")

    events = {}

    # Alle alligator:event Subjects finden
    for event_uri in g.subjects(RDF.type, ALLIGATOR.event):
        # Label extrahieren
        label = g.value(event_uri, RDFS.label)

        if label:
            label_str = str(label)

            # Alle Properties des Events sammeln
            event_data = {
                "uri": str(event_uri),
                "identifier": str(
                    g.value(
                        event_uri,
                        Namespace("http://purl.org/dc/elements/1.1/").identifier,
                    )
                    or ""
                ),
                "label": label_str,
                "estimatedstart": str(
                    g.value(event_uri, ALLIGATOR.estimatedstart) or ""
                ),
                "estimatedend": str(g.value(event_uri, ALLIGATOR.estimatedend) or ""),
                "cax": str(g.value(event_uri, ALLIGATOR.cax) or ""),
                "cay": str(g.value(event_uri, ALLIGATOR.cay) or ""),
                "caz": str(g.value(event_uri, ALLIGATOR.caz) or ""),
                "startfixed": str(g.value(event_uri, ALLIGATOR.startfixed) or ""),
                "endfixed": str(g.value(event_uri, ALLIGATOR.endfixed) or ""),
                "nfsn": str(g.value(event_uri, ALLIGATOR.nfsn) or ""),
                "nfen": str(g.value(event_uri, ALLIGATOR.nfen) or ""),
            }

            events[label_str] = event_data

    print(f"✓ {len(events)} Events gefunden")
    return events


def load_findspots_csv(csv_path):
    """
    Lädt die CSV-Datei mit den Fundorten.

    Returns:
        DataFrame: Pandas DataFrame mit den Fundorten
    """
    print(f"Lade CSV-Datei: {csv_path}")

    # Prüfen ob Datei existiert
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path.absolute()}")

    # CSV mit flexiblem Parser laden (erlaubt unterschiedliche Spaltenanzahl)
    df = pd.read_csv(csv_path, on_bad_lines="warn")
    print(f"✓ {len(df)} Fundorte gefunden")
    return df


def create_mapping_table(events, findspots_df):
    """
    Erstellt eine Mapping-Tabelle zwischen TTL-Events und CSV-Fundorten.
    Verwendet das Label als Schlüssel und versucht Fuzzy-Matching.

    Returns:
        DataFrame: Mapping-Tabelle
    """
    print("\nErstelle Mapping-Tabelle...")

    mapping_data = []

    for csv_label in findspots_df["label"]:
        # Exakte Matches
        exact_match = None
        if csv_label in events:
            exact_match = events[csv_label]

        # Fuzzy Match: Prüfe ob TTL-Label am Anfang des CSV-Labels steht
        # z.B. "Nijmegen, Lager" matched "Nijmegen, Lager (Haalebos)"
        fuzzy_match = None
        if not exact_match:
            for ttl_label, event_data in events.items():
                # CSV-Label startet mit TTL-Label
                if csv_label.startswith(ttl_label):
                    fuzzy_match = event_data
                    break
                # TTL-Label startet mit CSV-Label (umgekehrt)
                elif ttl_label.startswith(csv_label):
                    fuzzy_match = event_data
                    break

        match = exact_match or fuzzy_match

        if match:
            mapping_data.append(
                {
                    "csv_label": csv_label,
                    "ttl_label": match["label"],
                    "event_uri": match["uri"],
                    "event_identifier": match["identifier"],
                    "match_type": "exact" if exact_match else "fuzzy",
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
            )
        else:
            # Kein Match gefunden
            mapping_data.append(
                {
                    "csv_label": csv_label,
                    "ttl_label": None,
                    "event_uri": None,
                    "event_identifier": None,
                    "match_type": "no_match",
                    "estimatedstart": None,
                    "estimatedend": None,
                    "cax": None,
                    "cay": None,
                    "caz": None,
                    "startfixed": None,
                    "endfixed": None,
                    "nfsn": None,
                    "nfen": None,
                }
            )

    mapping_df = pd.DataFrame(mapping_data)

    # Statistik
    exact_matches = len(mapping_df[mapping_df["match_type"] == "exact"])
    fuzzy_matches = len(mapping_df[mapping_df["match_type"] == "fuzzy"])
    no_matches = len(mapping_df[mapping_df["match_type"] == "no_match"])

    print(f"\n✓ Mapping erstellt:")
    print(f"  - Exakte Matches: {exact_matches}")
    print(f"  - Fuzzy Matches: {fuzzy_matches}")
    print(f"  - Keine Matches: {no_matches}")

    if no_matches > 0:
        print(f"\nWarnung: {no_matches} Fundorte ohne Match:")
        print(mapping_df[mapping_df["match_type"] == "no_match"]["csv_label"].tolist())

    return mapping_df


# ============================================================
# RDF-Konvertierung
# ============================================================


def create_rdf_graph():
    """
    Erstellt einen neuen RDF-Graph mit allen Namespaces.
    """
    g = Graph()

    # Namespaces binden
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


def sanitize_id(label):
    """
    Erstellt eine saubere ID aus einem Label.
    """
    # Ersetze Leerzeichen und Sonderzeichen
    id_str = label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    id_str = id_str.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return id_str


def add_site_to_graph(g, row, site_uri, has_event=False):
    """
    Fügt einen einzelnen Fundort zum RDF-Graph hinzu.

    Args:
        g: RDF Graph
        row: DataFrame row mit Site-Daten
        site_uri: URI für den Site (kann Event-URI oder fallback sein)
        has_event: Boolean ob dieser Site ein Event hat
    """
    # rdf:type Aussagen
    g.add((site_uri, RDF.type, CRM.E53_Place))
    g.add((site_uri, RDF.type, FSL.Site))
    g.add((site_uri, RDF.type, FSL.ArchaeologicalSite))
    g.add((site_uri, RDF.type, PLEIADES_VOCAB.Place))

    # Wenn das ein Event ist, auch die Event-Types hinzufügen
    if has_event:
        g.add((site_uri, RDF.type, ALLIGATOR.event))
        g.add((site_uri, RDF.type, URIRef("http://www.w3.org/2006/time#Interval")))

    # Label
    g.add((site_uri, RDFS.label, Literal(row["label"], lang="en")))

    # DC Identifier (wenn Event vorhanden)
    if (
        has_event
        and pd.notna(row.get("event_identifier"))
        and str(row["event_identifier"]).strip()
    ):
        g.add((site_uri, DC.identifier, Literal(row["event_identifier"])))

    # GeoSPARQL Geometrie
    if pd.notna(row["wkt"]) and row["wkt"].strip():
        geom_uri = URIRef(str(site_uri) + "_geom")

        # WKT mit CRS formatieren
        wkt_value = row["wkt"].strip()
        wkt_literal = Literal(
            f"<http://www.opengis.net/def/crs/EPSG/0/4326> {wkt_value}",
            datatype=GEOSPARQL.wktLiteral,
        )

        # Geometrie-Triples
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

    # Wikidata Verlinkung
    if pd.notna(row["wikidata"]):
        wikidata_id = str(row["wikidata"]).strip()
        # Validierung: Muss mit Q beginnen und dann Zahl(en) haben
        if wikidata_id and wikidata_id.startswith("Q") and wikidata_id[1:].isdigit():
            wikidata_uri = WIKIDATA[wikidata_id]
            g.add((site_uri, LADO.wikidata, wikidata_uri))
        elif wikidata_id:
            print(
                f"  ⚠ Ungültige Wikidata-ID ignoriert: '{wikidata_id}' für {row['label']}"
            )

    # Pleiades Verlinkung
    if pd.notna(row["pleiades"]):
        # Pleiades IDs sind immer Zahlen
        try:
            if isinstance(row["pleiades"], (int, float)):
                pleiades_id = str(int(row["pleiades"]))
            else:
                pleiades_id = str(row["pleiades"]).strip()

            # Validierung: Muss eine Zahl sein
            if pleiades_id and pleiades_id.isdigit():
                pleiades_uri = PLEIADES_PLACE[pleiades_id]
                g.add((site_uri, LADO.pleiades, pleiades_uri))
            elif pleiades_id:
                print(
                    f"  ⚠ Ungültige Pleiades-ID ignoriert: '{pleiades_id}' für {row['label']}"
                )
        except (ValueError, TypeError):
            print(f"  ⚠ Fehler bei Pleiades-ID Konvertierung für {row['label']}")

    # Event-Zeitdaten (falls vorhanden, direkt am Site da Site = Event)
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

    # Alligator Koordinaten (cax, cay, caz)
    if pd.notna(row.get("cax")) and str(row["cax"]).strip():
        g.add((site_uri, ALLIGATOR.cax, Literal(row["cax"], datatype=XSD.decimal)))
    if pd.notna(row.get("cay")) and str(row["cay"]).strip():
        g.add((site_uri, ALLIGATOR.cay, Literal(row["cay"], datatype=XSD.decimal)))
    if pd.notna(row.get("caz")) and str(row["caz"]).strip():
        g.add((site_uri, ALLIGATOR.caz, Literal(row["caz"], datatype=XSD.decimal)))

    # Alligator Fixed-Flags
    if pd.notna(row.get("startfixed")) and str(row["startfixed"]).strip():
        g.add(
            (
                site_uri,
                ALLIGATOR.startfixed,
                Literal(row["startfixed"], datatype=XSD.boolean),
            )
        )
    if pd.notna(row.get("endfixed")) and str(row["endfixed"]).strip():
        g.add(
            (
                site_uri,
                ALLIGATOR.endfixed,
                Literal(row["endfixed"], datatype=XSD.boolean),
            )
        )

    # Alligator Nachbarschaften (nfsn, nfen)
    if pd.notna(row.get("nfsn")) and str(row["nfsn"]).strip():
        g.add((site_uri, ALLIGATOR.nfsn, Literal(row["nfsn"])))
    if pd.notna(row.get("nfen")) and str(row["nfen"]).strip():
        g.add((site_uri, ALLIGATOR.nfen, Literal(row["nfen"])))


def create_feature_collection(g, site_uris):
    """
    Erstellt eine GeoSPARQL FeatureCollection mit allen Sites.
    """
    # FeatureCollection definieren
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

    # Alle Sites zur Collection hinzufügen
    # GeoSPARQL erlaubt beide Richtungen:
    for site_uri in site_uris:
        # Feature → Collection
        g.add((site_uri, GEOSPARQL.memberOf, FEATURE_COLLECTION_URI))
        # Collection → Feature (inverse relationship)
        g.add((FEATURE_COLLECTION_URI, GEOSPARQL.hasFeature, site_uri))

    print(f"✓ FeatureCollection erstellt mit {len(site_uris)} Features")


def convert_to_rdf(merged_df):
    """
    Konvertiert die erweiterte CSV in einen RDF-Graph.
    """
    print("\n" + "=" * 60)
    print("RDF-Konvertierung")
    print("=" * 60)

    # RDF-Graph erstellen
    print("\nErstelle RDF-Graph...")
    g = create_rdf_graph()

    site_uris = []
    sites_with_events = 0
    sites_without_events = 0

    # Jeden Fundort zum Graph hinzufügen
    for idx, row in merged_df.iterrows():
        # Prüfen ob ein Event-Mapping existiert
        if pd.notna(row.get("event_uri")):
            event_uri_str = str(row["event_uri"]).strip()
            if event_uri_str and event_uri_str.lower() != "nan":
                # Event existiert - verwende Event-URI als Site-URI
                site_uri = URIRef(event_uri_str)
                has_event = True
                sites_with_events += 1
            else:
                # Kein Event - erstelle Fallback-URI
                site_id = sanitize_id(row["label"])
                site_uri = AE_SITES[f"site_{site_id}"]
                has_event = False
                sites_without_events += 1
        else:
            # Kein Event - erstelle Fallback-URI
            site_id = sanitize_id(row["label"])
            site_uri = AE_SITES[f"site_{site_id}"]
            has_event = False
            sites_without_events += 1

        site_uris.append(site_uri)

        # Site zum Graph hinzufügen
        add_site_to_graph(g, row, site_uri, has_event)

    print(f"✓ {len(site_uris)} Sites zum Graph hinzugefügt")
    print(f"  - Mit Event-URI: {sites_with_events}")
    print(f"  - Ohne Event (Fallback-URI): {sites_without_events}")

    # GeoSPARQL FeatureCollection erstellen
    print("\nErstelle GeoSPARQL FeatureCollection...")
    create_feature_collection(g, site_uris)

    # Graph speichern (Turtle-Format)
    output_file = OUTPUT_DIR / "arretine_sites_minigraph.ttl"
    g.serialize(destination=str(output_file), format="turtle")
    print(f"\n✓ RDF-Graph gespeichert: {output_file}")

    # Statistik
    print(f"\nGraph-Statistik:")
    print(f"  - Triples gesamt: {len(g)}")
    print(f"  - Sites: {len(site_uris)}")
    print(f"  - FeatureCollections: 1")

    return g


def main():
    """Hauptfunktion"""
    # Output-Verzeichnis vorbereiten (leeren und neu erstellen)
    setup_output_dir()

    # Logger initialisieren (schreibt in Terminal UND Datei)
    log_file = OUTPUT_DIR / "report.txt"
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print("=" * 60)
        print("Alligator Event Mapping")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 1. TTL-Events laden
        events = load_alligator_events(TTL_FILE)

        # 2. CSV-Fundorte laden
        findspots_df = load_findspots_csv(CSV_FILE)

        # 3. Mapping-Tabelle erstellen
        mapping_df = create_mapping_table(events, findspots_df)

        # 4. Mapping-Tabelle speichern
        output_file = OUTPUT_DIR / "event_mapping.csv"
        mapping_df.to_csv(output_file, index=False)
        print(f"\n✓ Mapping-Tabelle gespeichert: {output_file}")

        # 5. Erweiterte Tabelle: CSV + TTL-Daten zusammenführen
        merged_df = findspots_df.merge(
            mapping_df[
                [
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
            ],
            left_on="label",
            right_on="csv_label",
            how="left",
        )
        merged_df = merged_df.drop("csv_label", axis=1)

        output_merged = OUTPUT_DIR / "findspots_with_events.csv"
        merged_df.to_csv(output_merged, index=False)
        print(f"✓ Erweiterte Fundort-Tabelle gespeichert: {output_merged}")

        # 6. RDF-Konvertierung
        rdf_graph = convert_to_rdf(merged_df)

        print("\n" + "=" * 60)
        print("Fertig!")
        print(f"Report gespeichert: {log_file}")
        print("=" * 60)

    finally:
        # Logger schließen und stdout wiederherstellen
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
