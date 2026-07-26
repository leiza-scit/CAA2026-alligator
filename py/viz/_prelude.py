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


# The service-type palette, taken from py/events_timeline_by_service.py by
# sampling the same colormaps at the same points (Reds 0.35-0.90 over the six
# Service I types, Greens 0.45-0.88 over the two Service II types). Written out
# here because Pyodide has no matplotlib: a type must be the same colour in the
# browser as in the printed figure, and the only way to guarantee that without
# matplotlib is to carry the values.
SERVICE_COLOUR = {
    "Schrägrandteller":  "#1f77b4",
    "Service Ia Tasse":  "#fc9b7c",
    "Service Ia Teller": "#fb7757",
    "Service Ib Tasse":  "#f4503a",
    "Service Ib Teller": "#de2b25",
    "Service Ic Tasse":  "#be151a",
    "Service Ic Teller": "#980c13",
    "Service II Tasse":  "#86cc85",
    "Service II Teller": "#006b2b",
}

# Fill for a heatmap cell whose value follows from the rank assignment alone
# rather than from the sherds — see the quality figure. Same grey as the
# printed version uses for those cells.
STRUCTURAL_GREY = "#eeeeee"

# Matplotlib's RdYlGn and YlOrRd, sampled at eleven stops. The printed figures
# use them directly; Pyodide has no matplotlib, so the stops travel with the
# code and ramp() interpolates between them. Same colormap, same values, same
# colours in the browser as on the page.
RDYLGN = ["#a50026", "#d62f27", "#f46d43", "#fdad60", "#fee08b", "#feffbe",
          "#d9ef8b", "#a5d86a", "#66bd63", "#199750", "#006837"]
YLORRD = ["#ffffcc", "#fff1a9", "#fee187", "#feca66", "#feab49", "#fd8c3c",
          "#fc5b2e", "#ed2e21", "#d41020", "#b00026", "#800026"]


def ramp(stops, t):
    """Colour at position t in [0, 1] along a list of hex stops."""
    t = min(max(t, 0.0), 1.0)
    span = t * (len(stops) - 1)
    i = min(int(span), len(stops) - 2)
    f = span - i
    a = [int(stops[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(stops[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * f)
                                   for k in range(3))


def ink_on(hex_colour):
    """Black or white text, whichever stays legible on the given fill."""
    r, g, b = (int(hex_colour[k:k + 2], 16) for k in (1, 3, 5))
    return "#1a1a1a" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def colourbar(stops, lo, hi, width=190, height=10, fmt="{:.1f}"):
    """A horizontal colourbar as pure SVG.

    Drawn as thin segments rather than a gradient element so the figure stays
    a plain vector that survives being saved or printed — the same reason the
    printed figures avoid a rasterised colorbar.
    """
    steps = 60
    bars = "".join(
        f'<rect x="{i * width / steps:.2f}" y="0" '
        f'width="{width / steps + 0.6:.2f}" height="{height}" '
        f'fill="{ramp(stops, i / (steps - 1))}"/>' for i in range(steps))
    ticks = "".join(
        f'<text x="{f * width:.1f}" y="{height + 11}" font-size="9" '
        f'fill="#888" text-anchor="{a}">{fmt.format(lo + f * (hi - lo))}</text>'
        for f, a in ((0.0, "start"), (0.5, "middle"), (1.0, "end")))
    return (f'<svg width="{width}" height="{height + 15}" '
            f'xmlns="http://www.w3.org/2000/svg">{bars}{ticks}</svg>')


def rgzm(n, sum_rank, sum_rank_sq):
    """RGZM within-group variance and quality from the sums a query can return.

    SPARQL has no EXP or SQRT — rdflib's does not even parse them — so the query
    returns the three sums and the last step happens here:

        x̄  = Σ(rank·count) / N
        s  = sqrt( (Σ(rank²·count) − N·x̄²) / (N − 1) )    (STDDEV_SAMP)
        q  = exp(−s/|x̄|)                                  (quality, in (0, 1])

    Every sherd is one observation valued by the rank of its sub-type, which is
    what makes the sums above sufficient. q → 1 means the group's material sits
    on one rank; q → 0 means it is spread across the group's sequence.
    """
    import math

    if n < 2:
        return float("nan"), float("nan")
    mean = sum_rank / n
    variance = (sum_rank_sq - n * mean * mean) / (n - 1)
    s = math.sqrt(max(variance, 0.0))
    if mean == 0:
        return s, float("nan")
    return s, math.exp(-(s / abs(mean)))


def year(value):
    """Astronomical year number to a reading label: -9 -> '9 BC', 9 -> 'AD 9'.

    The graph stores xsd:gYear in the proleptic Gregorian calendar, which has a
    year zero; historical year numbering does not. Within this corpus nothing
    is dated to year 0, so the shift can be ignored and the conversion is the
    plain sign flip below.
    """
    v = int(value)
    return f"{-v} BC" if v < 0 else f"AD {v}"


# Service-type colours, sampled from the same matplotlib colormaps and the same
# ranges as build_service_colours() in events_timeline_by_service.py, so a type
# is the same colour in the browser as in the figures of the paper.
# Schrägrandteller is a fixed blue; Service I runs light to dark red over its
# six sub-types, Service II light to dark green over its two.
SERVICE_COLOUR = {
    "Schraegrandteller": "#1f77b4",
    "ServiceIa_Tasse":   "#fc9b7c",
    "ServiceIa_Teller":  "#fb7757",
    "ServiceIb_Tasse":   "#f4503a",
    "ServiceIb_Teller":  "#de2b25",
    "ServiceIc_Tasse":   "#be151a",
    "ServiceIc_Teller":  "#980c13",
    "ServiceII_Tasse":   "#86cc85",
    "ServiceII_Teller":  "#006b2b",
}

# Rows of the horizon figures run latest at the top, the timeline convention.
HORIZON_DISPLAY_ORDER = ["5", "4", "3", "2", "1"]
