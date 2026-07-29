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

Findspots the correspondence analysis could not place carry no horizon at
all; they are listed with their reason in FINDSPOT_EXCLUSIONS (section 3), so
that a gap between the workbook totals and the figure totals is always
accounted for rather than inferred.

Horizon 1 is the EARLIEST, horizon 5 the latest — the archaeological reading,
in which the numbering runs with time rather than against it.

Note for figures: several of them stack the horizons vertically and keep the
timeline convention of the latest material at the top. They therefore iterate
over HORIZON_DISPLAY_ORDER (5 → 1), not over the plain numerical order, so that
renumbering changed the labels without moving a single row.
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
    # Horizon 5 (latest)
    "Zurzach, Lager": 5,
    "Velsen": 5,
    "Vechten": 5,
    "Maastricht": 5,
    "Augst, Theater": 5,
    "Augst, Insula 20": 5,
    # Horizon 4
    "Wiesbaden": 4,
    "Vindonissa, Königsfelden": 4,
    "Oberwinterthur, Römerstr. 186": 4,
    "Nijmegen, Valkhof": 4,
    "Nijmegen, Trajanusplein": 4,
    "Friedberg": 4,
    "Bregenz": 4,
    "Avenches, Insula 15": 4,
    "Augst, Insula 31": 4,
    "Augsburg, Stadt": 4,
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
    # Horizon 2
    "Rödgen": 2,
    "Namur": 2,
    "Worms": 2,
    "Vindonissa, Militärstation": 2,
    "Neuss": 2,
    "Liberchies": 2,
    "Lausanne-Vidy": 2,
    "Basel, Lagerdorf": 2,
    "Augsburg-Oberhausen": 2,
    "Asberg, Lager": 2,
    # Horizon 1 (earliest)
    "Oberaden": 1,   # moved from Horizon 4
    "Zürich, Lindenhof": 1,
    "Titelberg": 1,
    "Nijmegen, Lager": 1,
    "Dangstetten": 1,
    "Basel, Lager": 1,
}


# Derived lookups: findspot -> horizon, and horizon -> ordered member list.
HORIZON_OF = dict(FINDSPOT_HORIZON)
HORIZONS: dict[int, list[str]] = {}
for _label, _h in FINDSPOT_HORIZON.items():
    HORIZONS.setdefault(_h, []).append(_label)
HORIZONS = {h: HORIZONS[h] for h in sorted(HORIZONS)}  # order 1..5

# Numerical order (earliest first) for tables and the RDF graph, and display
# order (latest first) for figures that stack horizons top to bottom.
HORIZON_NUMBERS = sorted(HORIZONS)
HORIZON_DISPLAY_ORDER = list(reversed(HORIZON_NUMBERS))


# ==============================================================================
# SECTION 3 · Findspots outside the seriation
# ==============================================================================
# The second table to edit. A findspot listed here carries no horizon and is
# absent from every horizon-keyed output — not because it was overlooked, but
# because the correspondence analysis could not place it. Recording the reason
# here keeps the exclusion machine-readable and keeps the sherd totals
# reconcilable: the difference between the workbook total and the figure total
# is exactly the material of the findspots named below.
#
# The correspondence analysis works on co-occurrence: a findspot earns its
# position from the company its types keep elsewhere. A findspot with a single
# type represented offers none, drops out of the analysis, and therefore has no
# CA coordinate, no Alligator date and no horizon.

# Reasons are language-keyed, like FORMS and GROUPS in services_to_rdf.py, so
# the bilingual outputs do not have to fall back to English. Add a "de" entry
# when a German output needs one; exclusion_reason falls back to "en".
FINDSPOT_EXCLUSIONS: dict[str, dict[str, str]] = {
    "Trier-Petrisberg": {
        "en": ("one service type represented (Schrägrandteller only), so the "
               "correspondence analysis drops the findspot: no CA coordinate, "
               "hence no Alligator date and no horizon"),
        "fr": ("un seul type de service représenté (Schrägrandteller uniquement) : "
               "l'analyse des correspondances écarte donc le site, qui n'a ni "
               "coordonnée AFC, ni date Alligator, ni horizon"),
    },
}


def exclusion_reason(label: str, lang: str = "en"):
    """Return why a findspot carries no horizon, or None if it is not excluded.

    Follows the same label bridge as resolve_horizon, so either spelling of a
    findspot finds its entry. Falls back to English where a language is missing.
    """
    entry = FINDSPOT_EXCLUSIONS.get(label)
    if entry is None:
        bridged = PERCENT_LABEL_CORRECTIONS.get(label)
        if bridged:
            entry = FINDSPOT_EXCLUSIONS.get(bridged)
    if entry is None:
        for ttl_label, xlsx_label in PERCENT_LABEL_CORRECTIONS.items():
            if xlsx_label == label:
                entry = FINDSPOT_EXCLUSIONS.get(ttl_label)
                break
    if entry is None:
        return None
    return entry.get(lang) or entry["en"]


# ==============================================================================
# SECTION 4 · Lookup helpers
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
