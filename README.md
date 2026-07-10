# CAA2026-alligator

Converting [Alligator](http://archaeology.link) seriation output for **Arretine ware (terra sigillata) find-spots** into a clean, standards-compliant RDF knowledge graph, and deriving temporal cluster analyses with **Allen's interval algebra**.

The pipeline takes the dating events produced by the Alligator correspondence-analysis tool, enriches them with spatial and authority data, serialises everything as Linked Open Data (CIDOC-CRM, GeoSPARQL, OWL-Time), groups sites into shared period clusters, and renders a set of timeline and Allen-relation visualisations.

This repository accompanies a CAA 2026 contribution.

---

## Repository structure

```
CAA2026-alligator/
├── data/                                   # Pipeline inputs (read by the script)
│   ├── ArretineDatedSitesServicesI_II.ttl              # Alligator output — dating events
│   ├── ArretineDatedSitesServicesI_II_findspots.csv    # Spatial + authority data per site
│   └── MoreEvents.csv                                   # Supplementary historical events
├── src/                                    # Upstream Alligator inputs (NOT read by the script)
│   └── …_CoordinatesCA_9999.{agt,tsv,ods,pdf}          # CA coordinates + dating table
├── help/                                   # Reference Turtle snippets (Haltern, Velsen)
├── py/
│   └── alligator_to_clean_rdf.py           # The pipeline (single entry point)
├── notebook/                               # Notebook variant + interactive HTML views
├── output/                                 # Generated artefacts (see below)
├── requirements.txt
└── LICENSE                                 # MIT
```

### Inputs

| File | Role |
| --- | --- |
| `data/…_ServicesI_II.ttl` | Alligator dating events: `estimatedstart/end`, `startfixed/endfixed`, CA coordinates (`cax/cay/caz`), nearest-fixed-neighbour hints (`nfsn/nfen`). **This is the only temporal source the script reads.** |
| `data/…_findspots.csv` | Per-site geometry (`wkt`) and authority identifiers (Wikidata, Pleiades, OSM). No temporal data. |
| `data/MoreEvents.csv` | Historical events (e.g. campaigns, *Clades Variana*) added as OWL-Time / CIDOC-CRM nodes. |

The `src/` files are the *upstream* input to the external Alligator tool (correspondence-analysis coordinates and the dating table in `.agt`, `.tsv`, `.ods` and `.pdf` form). They are kept for provenance but are **not** consulted by `alligator_to_clean_rdf.py`.

### Outputs (`output/`)

| File | Description |
| --- | --- |
| `arretine_sites_minigraph.ttl` | Final combined RDF graph (sites, events, period clusters, Allen relations). |
| `event_mapping.csv` | Label-based mapping between find-spot labels and Alligator event URIs. |
| `findspots_with_events.csv` | Enriched find-spot table after the merge step. |
| `events_timeline.jpg` | Per-site interval timeline (fixed vs calculated boundaries). |
| `cluster_timeline.jpg` | Timeline of the detected period clusters. |
| `allen_matrix.jpg` | Pairwise Allen-relation matrix between clusters. |
| `allen_chain.jpg` | Allen-relation chain diagram. |
| `report.txt` | Full run log with mapping and clustering statistics. |

---

## Pipeline

`py/alligator_to_clean_rdf.py` runs the following steps sequentially (each marked with a `# ── SECTION ──` comment so it maps cleanly onto notebook cells):

1. Load Alligator events from the Turtle file.
2. Load find-spot data from CSV.
3. Build a label-based mapping table (exact + fuzzy matching, with a small `TTL_LABEL_CORRECTIONS` table for known typos).
4. Merge the mapped event data into the find-spot table.
5. Convert the enriched table into an RDF graph.
6. Add supplementary events from `MoreEvents.csv` as OWL-Time + CIDOC-CRM nodes.
7. Detect period clusters (sites sharing an exact `estimatedstart`/`estimatedend` pair).
8. Compute Allen interval relations between clusters.
9. Serialise the combined graph and render all visualisations.

### Vocabularies

CIDOC-CRM, GeoSPARQL (+ Simple Features), OWL-Time and SKOS for the standards-compliant output; the Alligator/LADO ontology (`archaeology.link`) and `fuzzy-sl.squirrel.link` for the source semantics; Pleiades and Wikidata for place authorities.

---

## Installation

Requires Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `matplotlib`, `networkx`, `rdflib`, `shapely` (all other imports are standard library).

## Usage

```bash
python3 py/alligator_to_clean_rdf.py
```

All paths are resolved relative to the script location, so no arguments are needed. Every artefact in `output/` is regenerated on each run, and a timestamped log is written to `output/report.txt`.

---

## Editing a site's dating

Because the script reads temporal data **only** from `data/…_ServicesI_II.ttl`, changing a site's date range is a matter of editing that site's block and re-running the pipeline. For example, to fix *Oberaden* to a closed interval of 11–7 BC:

```turtle
ae:rJXxA7 alligator:estimatedstart "-11.0" .   # left boundary
ae:rJXxA7 alligator:estimatedend   "-7.0"  .   # right boundary
ae:rJXxA7 alligator:startfixed     "true"  .   # boundary certain
ae:rJXxA7 alligator:endfixed       "true"  .   # boundary certain
#   (remove any alligator:nfen / alligator:nfsn line once the boundary is fixed)
```

Re-running the script then propagates the change everywhere: the period clustering, the Allen relations, the final RDF and all four visualisations. For full provenance you may also mirror the change in the `src/` table (`Oberaden … -11  -7  fixed`) and, if you have the Alligator tool to hand, regenerate the Turtle from it.

### Visualisation colour coding

* **Steel blue** — both boundaries fixed (`startfixed` **and** `endfixed` are `true`).
* **Gold** — at least one boundary is Alligator-calculated; the label then carries the nearest-fixed-neighbour hint (`label-->nfsn,nfen`).

---

## Citation

If you use this work, please cite the accompanying CAA 2026 contribution. See `LICENSE` for reuse terms.

## Licence

MIT © 2026 Leibniz-Zentrum für Archäologie (LEIZA)
