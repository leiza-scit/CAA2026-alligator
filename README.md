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
│   ├── main.py                             # Orchestrator — runs every step in order
│   ├── alligator_to_clean_rdf.py           # RDF pipeline, period clusters, Allen relations
│   ├── events_timeline_by_service.py       # Service composition and horizon figures
│   ├── services_to_rdf.py                  # Service-type counts as a sidecar RDF layer
│   ├── horizon_assignment.py               # Horizon assignment from a composition
│   ├── build_sparql.py                     # Interactive query page and notebooks
│   ├── build_docs.py                       # The variability note, generated
│   └── horizons.py                         # Shared find-spot → horizon table
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
| `horizon_reference_profiles.csv` | Pooled sherd counts and proportions per horizon — the reference of the horizon assignment. |
| `horizon_assignment_examples.csv` | Worked examples: three compositions × three sample sizes × both methods. |
| `horizon_assignment_loo.csv` | Leave-one-out validation: every find-spot re-assigned against a reference built without it. |

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

### Assigning a new find to a horizon

`py/horizon_assignment.py` reads the sequence in the opposite direction to the figures. Given an assemblage counted by service type, which of the five horizons does it belong to — and how sure is that?

Run without arguments it prints the reference profiles, three worked examples, the number of sherds needed to reach a given confidence, and a leave-one-out validation over all 44 seriated find-spots. It then writes the three CSVs listed above.

```bash
python3 py/horizon_assignment.py
python3 py/horizon_assignment.py --no-write        # print only, write nothing
```

To classify an assemblage of your own, give its percentage shares and the number of sherds behind them:

```bash
python3 py/horizon_assignment.py --assign "Ib=13,Ic=27,II=60" --n 100
python3 py/horizon_assignment.py --assign "Ib=6,Ic=24,II=70" --n 100 --steps
python3 py/horizon_assignment.py --assign "Service II=90,Ic=8,Ib=2" --n 30
```

Category names are matched loosely, so `II`, `Service II` and `srt` all reach the right one, and shares that do not sum to 100 are rescaled with a note. `--steps` prints the whole arithmetic — the log-likelihood of every horizon, its normalisation into probabilities, and the Dirichlet-multinomial correction that allows for the reference being a sample too — so each figure can be checked with a calculator.

The five categories are Schrägrandteller, Service Ia, Ib, Ic and Service II. Cup and plate of one stage are merged, because a change of form is not a step in time. The three service groups on their own are not enough: most of the discrimination between neighbouring horizons sits in the Ia/Ib/Ic subdivision.

This step also runs as part of the full pipeline (`python3 py/main.py`), or on its own with `python3 py/main.py --only assign`.

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
