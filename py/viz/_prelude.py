# Shared helpers for the figures in py/viz/, inlined into the notebook's setup
# cell (queries.yaml -> qmd.viz_prelude). Nothing here is generic: it is the
# vocabulary this particular graph is discussed in.
#
# This file is *not* imported. It is pasted into a Pyodide cell, so it may only
# use the standard library and must not assume a working directory.

# The Allen relation palette from alligator_to_clean_rdf.py, so a relation is
# the same colour in the browser figures and in the figures of the paper.
ALLEN_COLOUR = {
    "before":       "#4a90d9",   # blue
    "after":        "#2c5f8a",   # dark blue
    "meets":        "#7ab3e0",   # light blue
    "metBy":        "#5a9fc5",   # mid blue
    "overlaps":     "#f0a500",   # orange
    "overlappedBy": "#c97d00",   # dark orange
    "contains":     "#d94a4a",   # red
    "during":       "#a03030",   # dark red
    "starts":       "#e07070",   # light red
    "startedBy":    "#c05050",   # mid red
    "finishes":     "#e09090",   # pink-red
    "finishedBy":   "#b04060",   # rose
    "equals":       "#4caf50",   # green
}

# Allen's own ordering, from wholly earlier to wholly later. Sorting the bar
# chart by frequency would hide that the occupied relations form one block.
ALLEN_ORDER = [
    "before", "meets", "overlaps", "finishedBy", "contains", "starts",
    "equals", "startedBy", "during", "finishes", "overlappedBy", "metBy",
    "after",
]

# Relations that mean "contemporary with, or later than" the reference event,
# copied from the companion notebook so both give the same answer. 'meets' is
# deliberately included: an interval ending exactly where the event begins was
# still in use when it began.
#
# 'overlaps' is the one omission, and it is worth knowing about. Against a
# single-year event it would be unreachable, but the graph dates the Clades
# Variana AD 8 to 9, so a findspot ending in AD 8 would fall into it — and
# would then be dropped, although it was in use when the event opened. No
# findspot in this corpus does, so the two readings agree here; whether they
# should agree in general is a question for Allard.
CONTEMPORARY_OR_LATER = {
    "after", "metBy", "equals", "during", "starts", "startedBy", "finishes",
    "finishedBy", "overlappedBy", "contains", "meets",
}

# One colour per chronological horizon, cold (early) to warm (late), so the
# map can be read as a sequence without consulting the legend.
HORIZON_COLOUR = {
    "1": "#0d4a70",
    "2": "#2e86ab",
    "3": "#7fb800",
    "4": "#f0a500",
    "5": "#c1440e",
}

# The reference event, for the two temporal figures. This is the year the
# timeline marks, i.e. the defeat itself; the Allen relations in the query are
# computed against the full interval the graph records, AD 8 to 9.
EVENT_LABEL = "Clades Variana"
EVENT_YEAR = 9
EVENT_COLOUR = "#993c1d"


def year(value):
    """Astronomical year number to a reading label: -9 -> '9 BC', 9 -> 'AD 9'.

    The graph stores xsd:gYear in the proleptic Gregorian calendar, which has a
    year zero; historical year numbering does not. Within this corpus nothing
    is dated to year 0, so the shift can be ignored and the conversion is the
    plain sign flip below.
    """
    v = int(value)
    return f"{-v} BC" if v < 0 else f"AD {v}"
