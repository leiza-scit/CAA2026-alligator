"""Chronological horizons — the single shared definition.

Both pipeline scripts need to know which findspot belongs to which horizon:
`alligator_to_clean_rdf.py` writes the horizons into the RDF graph, and
`events_timeline_by_service.py` draws them. Keeping the table in one module
means the graph and the figures can never drift apart — change a findspot's
horizon here and a single pipeline run updates the RDF, the CSVs and every
figure at once.

Import it the same way as wd_repro, i.e. after putting this file's own
directory on sys.path, so the working directory VS Code launches from does not
matter:

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from horizons import FINDSPOT_HORIZON, resolve_horizon, build_horizon_intervals

Horizon 1 is the latest, horizon 5 the earliest.
"""

from __future__ import annotations


# ==============================================================================
# SECTION 1 · Label bridge
# ==============================================================================
# Where the Alligator TTL and the workbook spell a findspot differently, this
# maps the TTL spelling to the workbook spelling. Used by resolve_horizon so a
# findspot is found whichever spelling reaches it.

PERCENT_LABEL_CORRECTIONS: dict[str, str] = {
    # The TTL calls the Haalebos camp simply "Nijmegen, Lager"; the workbook
    # disambiguates it from the Brunsting camp with a suffix.
    "Nijmegen, Lager": "Nijmegen, Lager (Haalebos)",
}


# ==============================================================================
# SECTION 2 · Horizon assignment
# ==============================================================================
# THE table to edit: to move a findspot to another horizon, change its number.
# Everything downstream — RDF graph, CSVs, figures — follows automatically.

FINDSPOT_HORIZON = {
    # Horizon 1 (latest)
    "Zurzach, Lager": 1,
    "Velsen": 1,
    "Vechten": 1,
    "Maastricht": 1,
    "Augst, Theater": 1,
    "Augst, Insula 20": 1,
    # Horizon 2
    "Wiesbaden": 2,
    "Vindonissa, Königsfelden": 2,
    "Oberwinterthur, Römerstr. 186": 2,
    "Nijmegen, Valkhof": 2,
    "Nijmegen, Trajanusplein": 2,
    "Friedberg": 2,
    "Bregenz": 2,
    "Avenches, Insula 15": 2,
    "Augst, Insula 31": 2,
    "Augsburg, Stadt": 2,
    # Horizon 3
    "Vindonissa, Scheuerhof": 3,
    "Vetera I": 3,
    "Tongeren": 3,
    "Nijmegen, Lager (Brunsting)": 3,
    "Mainz, Legionslager": 3,
    "Lorenzberg": 3,
    "Haltern": 3,
    "Conimbriga": 3,
    "Braives": 3,
    "Bonn, Boeselagerhof": 3,
    "Bad Nauheim": 3,
    "Asberg, Lagerdorf": 3,
    # Horizon 4
    "Rödgen": 4,
    "Namur": 4,
    "Worms": 4,
    "Vindonissa, Militärstation": 4,
    "Neuss": 4,
    "Liberchies": 4,
    "Lausanne-Vidy": 4,
    "Basel, Lagerdorf": 4,
    "Augsburg-Oberhausen": 4,
    "Asberg, Lager": 4,
    # Horizon 5 (earliest)
    "Oberaden": 5,   # moved from Horizon 4
    "Zürich, Lindenhof": 5,
    "Titelberg": 5,
    "Nijmegen, Lager": 5,
    "Dangstetten": 5,
    "Basel, Lager": 5,
}


# Derived lookups: findspot -> horizon, and horizon -> ordered member list.
HORIZON_OF = dict(FINDSPOT_HORIZON)
HORIZONS: dict[int, list[str]] = {}
for _label, _h in FINDSPOT_HORIZON.items():
    HORIZONS.setdefault(_h, []).append(_label)
HORIZONS = {h: HORIZONS[h] for h in sorted(HORIZONS)}  # order 1..5


# ==============================================================================
# SECTION 3 · Lookup helpers
# ==============================================================================


def resolve_horizon(label: str):
    """Return the horizon number for a findspot label, or None if unassigned.

    Tries the label directly, then via PERCENT_LABEL_CORRECTIONS (so an xlsx
    spelling like "Nijmegen, Lager (Haalebos)" resolves to its TTL horizon).
    """
    if label in HORIZON_OF:
        return HORIZON_OF[label]
    bridged = PERCENT_LABEL_CORRECTIONS.get(label)
    if bridged and bridged in HORIZON_OF:
        return HORIZON_OF[bridged]
    # xlsx→TTL inverse bridge (e.g. "Nijmegen, Lager (Haalebos)" → "Nijmegen, Lager")
    for ttl_label, xlsx_label in PERCENT_LABEL_CORRECTIONS.items():
        if xlsx_label == label and ttl_label in HORIZON_OF:
            return HORIZON_OF[ttl_label]
    return None


def build_horizon_intervals(events: dict) -> list:
    """Return one interval dict per horizon, shaped like a period cluster.

    The dicts carry the same keys the ported plot functions expect ("start",
    "end", "members") plus "horizon", so the code taken from
    alligator_to_clean_rdf.py works unchanged.

    The interval is the envelope over the horizon's findspots: start = earliest
    estimatedstart, end = latest estimatedend. Horizons without a single dated
    event are skipped.

    Each member also carries "event_uri", which the RDF writer needs to link the
    horizon to the findspot nodes already in the graph.
    """
    buckets: dict = {}
    for label, ev in events.items():
        h = resolve_horizon(label)
        if h is None:
            continue
        try:
            start = float(ev["estimatedstart"])
            end = float(ev["estimatedend"])
        except (ValueError, TypeError):
            continue
        buckets.setdefault(h, []).append({
            "label": label,
            "start": start,
            "end": end,
            "event_uri": ev.get("uri", ""),
        })

    horizons = []
    for h in sorted(buckets):
        members = buckets[h]
        horizons.append({
            "horizon": h,
            "start": min(m["start"] for m in members),
            "end": max(m["end"] for m in members),
            "members": members,
        })
    # Earliest first, matching the ported timeline's own ordering.
    horizons.sort(key=lambda c: (c["start"], c["end"]))
    return horizons
