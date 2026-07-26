# Where everything goes

Unpack over `c:\git\CAA2026-alligator\` — every path below is relative to the
repository root, so the archive's own structure already puts each file in the
right place. Then run `python py/main.py` once.

Delete `MANIFEST.md` afterwards; it is not part of the repository.

## Source — replace (6)

| File | What changed |
|---|---|
| `queries.yaml` | `aeont:` and `fsl:` prefixes; `graph.extra.services`; `qmd.viz_prelude`, `qmd.explore`, `qmd.footer`; four new queries (`allen-relations-to-clades-variana`, `findspot-coordinates`, `service-composition`, `service-within-group-variability`) |
| `py/build_sparql.py` | `load_viz()` reads the figure code; `load_extra()` verifies a locally built graph from disk; publishes extra graphs to `docs/`; `_inline_json()` escapes `</script>` |
| `py/main.py` | new pipeline step `enrich` between `service` and `sparql` |
| `py/templates/sparql.html.j2` | figure slots, `Frame` helper, `render_figure()`, figures cleared on Reset |
| `py/templates/sparql.qmd.j2` | figure cells, `results` registry, `Frame` helper; Explore section and footer now come from `queries.yaml` instead of being hard-coded |
| `py/templates/style.css` | `.figures`, `.figure`, `.figure-body` rules |

## Source — new (7)

| File | What it is |
|---|---|
| `py/services_to_rdf.py` | the enrichment step: workbook → `output/arretine_services.ttl` |
| `py/viz/_prelude.py` | shared constants and helpers, inlined into the notebook's setup cell |
| `py/viz/allen_distribution.py` | Allen relation distribution (Chart.js) |
| `py/viz/clades_variana_timeline.py` | interval timeline against AD 9 (SVG) |
| `py/viz/site_map.py` | findspots by horizon (Leaflet) |
| `py/viz/service_composition.py` | proportional service composition per findspot (SVG) |
| `py/viz/service_variability.py` | RGZM variance/quality per horizon, both rank readings (HTML) |

## Generated — included for convenience, not required (16)

`python py/main.py` rewrites all of these from the source above. They are in the
archive so the state can be inspected without running anything.

- `output/arretine_services.ttl` — the service layer, 1236 triples
- `docs/arretine_services.ttl` — its published copy
- `docs/sparql.html`, `docs/style.css`
- `docs/downloads/queries/*.rq` — 11 files
- `qmd/arretine-chronology-sparql-live.qmd`

Not included, because rerunning the pipeline regenerates them and shipping them
would only create noise in git: everything else under `output/`, and
`docs/arretine_sites_minigraph.ttl`. Their content is unchanged — the reference
CSVs `service_percentages.csv` and `service_group_variability.csv` were checked
against the copies in the archive you sent and are numerically identical.

## One thing to decide separately

`.gitignore` carries `downloads/` from the Python packaging block, which also
matches `docs/downloads/`. The `.rq` files therefore never reach the repository.
Nothing links to them yet, so nothing is broken; add `!docs/downloads/` if they
should be published.
