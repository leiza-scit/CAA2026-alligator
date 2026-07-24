"""Alligator events timeline, recoloured by service-type composition.

This is a standalone variant of the events timeline produced by
``alligator_to_clean_rdf.py``. The bar *positions* are identical (each bar
still spans an Alligator event's estimated start/end year), but instead of
colouring bars by boundary state (fixed / calculated / gradient), every bar is
split into nine proportional segments — one per Arretine service type — using
the per-findspot percentages computed from the workbook.

Design notes
------------
* ``alligator_to_clean_rdf.py`` is NOT imported or modified. The two passages
  this figure genuinely needs — ``load_alligator_events`` and
  ``_format_year_label`` — are copied verbatim below (clearly marked) so this
  script runs on its own.
* The workbook-reading / percentage maths (originally in
  ``read_arretine_services.py``) are inlined in Section 3a, so this script has
  no local-module dependency and runs from any working directory.
* Boundary state (startfixed / endfixed) is deliberately ignored here, as is
  the "-->nfsn,nfen" suffix: each bar is labelled with the plain findspot name.

Inputs
------
* root/data/ArretineDatedSitesServicesI_II.ttl   (Alligator events → bar spans)
* root/src/ArretineDatedSitesServicesI_II.xlsx    (service-type counts, B-J)

Outputs (root/output/)
----------------------
* events_timeline_by_service_en.jpg / .svg   (English)
* events_timeline_by_service_fr.jpg / .svg   (French)
* service_percentages.csv                    (per-findspot percentages)
"""

from __future__ import annotations

# ==============================================================================
# SECTION 1 · Imports
# ==============================================================================

from pathlib import Path
import sys
import textwrap

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — no display required

# Byte-reproducible figure output: pins the SVG hash salt and SOURCE_DATE_EPOCH,
# so an unchanged figure produces an identical file and a diff appears only when
# something really changed. Imported for its effect; there is nothing to call.
# The sys.path line makes the sibling module resolvable no matter which working
# directory the script is launched from (VS Code often uses the repository root).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wd_repro  # noqa: E402, F401  (imported for its effect)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from rdflib import Graph, Namespace, RDF, RDFS


# ==============================================================================
# SECTION 2 · Configuration
# ==============================================================================
# Directory layout mirrors alligator_to_clean_rdf.py: this script lives in
# root/py/; Alligator TTL in root/data/; the xlsx in root/src/; figures go to
# root/output/.

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
SRC_DIR = REPO_ROOT / "src"
OUTPUT_DIR = REPO_ROOT / "output"

TTL_FILE = DATA_DIR / "ArretineDatedSitesServicesI_II.ttl"

# Workbook holding the service-type counts (columns B-J). The reading logic is
# inlined below (Section 3a) so this script has no local module dependencies.
XLSX_FILE = SRC_DIR / "ArretineDatedSitesServicesI_II.xlsx"
SHEET_NAME = "ArretineDatedSitesServicesI_II"
LABEL_COLUMN = "findspot"      # column A carries no header; we name it explicitly
SERVICE_COL_START = 1          # column B, 0-based
SERVICE_COL_END = 10           # column J inclusive -> slice stop is exclusive

# Alligator input namespace (copied from alligator_to_clean_rdf.py, Section 2).
ALLIGATOR = Namespace("http://archaeology.link/ontology#")

# Known label corrections for confirmed typos in the Alligator TTL output.
# (Copied from alligator_to_clean_rdf.py, Section 2.)
TTL_LABEL_CORRECTIONS = {
    "Vindoniss, Militärstation": "Vindonissa, Militärstation",
    "Avences, Insula 15": "Avenches, Insula 15",
}

# Optional bridge for cases where a (corrected) TTL event label still differs
# from the authoritative xlsx findspot label. Map  TTL label -> xlsx label.
# Populate this if the mismatch warning below lists any findspots.
PERCENT_LABEL_CORRECTIONS: dict[str, str] = {
    # The TTL calls the Haalebos camp simply "Nijmegen, Lager"; the workbook
    # disambiguates it from the Brunsting camp with a suffix.
    "Nijmegen, Lager": "Nijmegen, Lager (Haalebos)",
}

# Colour scheme, assigned by service-type *group* (not a flat palette):
#   · "Schrägrandteller"  → a single blue
#   · "Service I..."      → a red gradient    (Ia → Ib → Ic, light → dark)
#   · "Service II..."     → a green gradient  (Tasse → Teller, light → dark)
BLUE_COLOUR = "#1f77b4"        # Schrägrandteller
RED_CMAP = "Reds"              # Service I family
GREEN_CMAP = "Greens"          # Service II family
RED_RANGE = (0.35, 0.90)       # sampled span of the Reds colormap (avoids white)
GREEN_RANGE = (0.45, 0.88)     # sampled span of the Greens colormap

# Neutral fill for any bar whose findspot has no matching percentage row.
NO_MATCH_COLOUR = "#d9d9d9"

# ---------------------------------------------------------------------------
# Localisation (English / French)
# The DataFrame keeps the original German column names; only the *displayed*
# legend labels are translated. Edit these mappings to refine the archaeological
# terminology if required.
# ---------------------------------------------------------------------------
SERVICE_LABELS = {
    "en": {
        "Schrägrandteller": "Oblique-rim plate",
        "Service Ia Tasse": "Service Ia cup",
        "Service Ia Teller": "Service Ia plate",
        "Service Ib Tasse": "Service Ib cup",
        "Service Ib Teller": "Service Ib plate",
        "Service Ic Tasse": "Service Ic cup",
        "Service Ic Teller": "Service Ic plate",
        "Service II Tasse": "Service II cup",
        "Service II Teller": "Service II plate",
    },
    "fr": {
        "Schrägrandteller": "Assiette à bord oblique",
        "Service Ia Tasse": "Service Ia tasse",
        "Service Ia Teller": "Service Ia assiette",
        "Service Ib Tasse": "Service Ib tasse",
        "Service Ib Teller": "Service Ib assiette",
        "Service Ic Tasse": "Service Ic tasse",
        "Service Ic Teller": "Service Ic assiette",
        "Service II Tasse": "Service II tasse",
        "Service II Teller": "Service II assiette",
    },
}

STRINGS = {
    "en": {
        "legend_title": "Service type (share of assemblage)",
        "xlabel": "Year",
        "no_data": "no data",
        "var_panel": "Within-group variance  (STDDEV_SAMP of sub-type ranks)",
        "qual_panel": "Within-group quality  (q = exp(−CV))",
        "n_note": "n = sherds in assemblage; low n printed in grey (less reliable).",
        "single_note": (
            "Schrägrandteller is a single form — no internal variability (—)."
        ),
        "horizon": "Horizon",
        "share_xlabel": "Share of assemblage (%)",
        "spread_note": (
            "RGZM method, applied WITHIN each group: every sherd is one observation "
            "valued by the rank of its sub-type inside its group (column order 1..k). "
            "variance = STDDEV_SAMP of those ranks · quality q = exp(−CV), CV = s/|x̄|. "
            "q → 1 = the group is concentrated in few sub-types; q → 0 = spread across "
            "its sub-type sequence. n = sherds in that group. Schrägrandteller has one "
            "sub-type (variance 0, q = 1 where present)."
        ),
    },
    "fr": {
        "legend_title": "Type de service (part de l'assemblage)",
        "xlabel": "Année",
        "no_data": "pas de données",
        "var_panel": "Variance intra-groupe  (STDDEV_SAMP des rangs de sous-types)",
        "qual_panel": "Qualité intra-groupe  (q = exp(−CV))",
        "n_note": "n = tessons de l'assemblage ; n faible en gris (moins fiable).",
        "single_note": (
            "Schrägrandteller est une forme unique — pas de variabilité interne (—)."
        ),
        "horizon": "Horizon",
        "share_xlabel": "Part de l'assemblage (%)",
        "spread_note": (
            "Méthode RGZM, appliquée À L'INTÉRIEUR de chaque groupe : chaque tesson est "
            "une observation valuée par le rang de son sous-type dans son groupe (ordre "
            "1..k). variance = STDDEV_SAMP des rangs · qualité q = exp(−CV), CV = s/|x̄|. "
            "q → 1 = groupe concentré sur peu de sous-types ; q → 0 = dispersé sur la "
            "séquence. n = tessons du groupe. Schrägrandteller n'a qu'un sous-type "
            "(variance 0, q = 1 si présent)."
        ),
    },
}

# Display names for the three service GROUPS (used by the variability chart).
GROUP_DISPLAY = {
    "en": {
        "Schrägrandteller": "Oblique-rim plate",
        "Service I": "Service I",
        "Service II": "Service II",
    },
    "fr": {
        "Schrägrandteller": "Assiette à bord oblique",
        "Service I": "Service I",
        "Service II": "Service II",
    },
}

# Signature tint for each group's column header (echoes the timeline hues).
GROUP_TINT = {
    "Schrägrandteller": "#1f77b4",  # blue
    "Service I": "#cf2f26",         # red
    "Service II": "#2a924a",        # green
}

# Findspots with fewer than this many sherds are flagged as low-reliability.
LOW_N = 20

# ---------------------------------------------------------------------------
# Timeline row order — manual overrides
# The timeline sorts by (estimatedstart, estimatedend, label). Where that
# ordering should be overridden for presentation, list the findspot here
# together with the findspot it must appear DIRECTLY BELOW in the figure.
# The bar's position on the time axis is unaffected — only its row changes.
# To move another findspot, add one line: "<findspot>": "<findspot above it>".
# ---------------------------------------------------------------------------
TIMELINE_ROW_BELOW = {
    # Oberaden should sit between "Asberg, Lager" and "Zürich, Lindenhof"
    # rather than between "Asberg, Lagerdorf" and "Rödgen".
    "Oberaden": "Asberg, Lager",
}

# ---------------------------------------------------------------------------
# Chronological horizons
# Assign each findspot to a horizon (1 = latest … 5 = earliest). This is the
# single place to edit: to MOVE a findspot to another horizon, just change its
# number below. Everything downstream (aggregation, figure, CSV, the per-horizon
# findspot list) is derived from this table automatically.
# Membership uses the timeline (TTL) labels; xlsx-only spellings are resolved
# via PERCENT_LABEL_CORRECTIONS.
# ---------------------------------------------------------------------------
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

# Derived: findspot -> horizon, and horizon -> ordered member list.
HORIZON_OF = dict(FINDSPOT_HORIZON)
HORIZONS: dict[int, list[str]] = {}
for _label, _h in FINDSPOT_HORIZON.items():
    HORIZONS.setdefault(_h, []).append(_label)
HORIZONS = {h: HORIZONS[h] for h in sorted(HORIZONS)}  # order 1..5


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

# Era suffixes for year tick labels, per language.
ERA_LABELS = {
    "en": {"bc": "BC", "ad": "AD"},   # 15 BC / AD 9
    "fr": {"bc": "av. J.-C.", "ad": "apr. J.-C."},  # 15 av. J.-C. / 9 apr. J.-C.
}

# CSV outputs.
PERCENT_CSV = OUTPUT_DIR / "service_percentages.csv"
GROUP_VAR_CSV = OUTPUT_DIR / "service_group_variability.csv"


def build_service_colours(service_cols: list[str]) -> list:
    """Return one RGBA colour per service column, grouped by type family.

    Schrägrandteller is a fixed blue. The Service I family (Ia/Ib/Ic, in column
    order) is spread across a red gradient, and the Service II family across a
    green gradient, so related types read as shades of one hue while the three
    families stay clearly distinct.
    """
    from matplotlib.colors import to_rgba

    red_cmap = matplotlib.colormaps[RED_CMAP]
    green_cmap = matplotlib.colormaps[GREEN_CMAP]

    # Check "Service II" before "Service I": the former also startswith the latter.
    n_red = sum(
        1 for c in service_cols
        if c.startswith("Service I") and not c.startswith("Service II")
    )
    n_green = sum(1 for c in service_cols if c.startswith("Service II"))

    def _ramp(cmap, lo, hi, k, total):
        """k-th of `total` samples across [lo, hi] of a colormap (light→dark)."""
        if total <= 1:
            return cmap((lo + hi) / 2)
        return cmap(lo + (hi - lo) * k / (total - 1))

    colours, ri, gi = [], 0, 0
    for c in service_cols:
        if c.startswith("Schrägrandteller"):
            colours.append(to_rgba(BLUE_COLOUR))
        elif c.startswith("Service II"):
            colours.append(_ramp(green_cmap, *GREEN_RANGE, gi, n_green))
            gi += 1
        elif c.startswith("Service I"):
            colours.append(_ramp(red_cmap, *RED_RANGE, ri, n_red))
            ri += 1
        else:  # unexpected column name — fall back to neutral
            colours.append(to_rgba(NO_MATCH_COLOUR))
    return colours


# ==============================================================================
# SECTION 3 · Copied helpers (verbatim from alligator_to_clean_rdf.py)
# ==============================================================================
# --- copied from Section 5 -------------------------------------------------
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

        # Apply known TTL label corrections (see TTL_LABEL_CORRECTIONS in Section 2)
        label_str = TTL_LABEL_CORRECTIONS.get(label_str, label_str)

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


# --- copied from Section 16 ------------------------------------------------
def _format_year_label(year: float) -> str:
    """Return a human-readable year label, e.g. '15 BC' or 'AD 9'."""
    y = round(year)
    return f"{abs(y)} BC" if y < 0 else f"AD {y}"


def _year_label(year: float, lang: str = "en") -> str:
    """Localised year label, e.g. '15 BC' / '15 av. J.-C.' or 'AD 9' / '9 apr. J.-C.'."""
    era = ERA_LABELS.get(lang, ERA_LABELS["en"])
    y = round(year)
    if y < 0:
        return f"{abs(y)} {era['bc']}"
    return f"{era['ad']} {y}" if lang == "en" else f"{y} {era['ad']}"


# ==============================================================================
# SECTION 3a · Workbook reader (inlined from read_arretine_services.py)
# ==============================================================================
# These three functions were previously imported from the sibling script. They
# are inlined here so events_timeline_by_service.py runs with no local-module
# dependency, regardless of the working directory VS Code launches it from.


def load_service_matrix(xlsx_path: Path = XLSX_FILE) -> pd.DataFrame:
    """Read columns B-J and return them as a tidy abundance matrix.

    The returned DataFrame is indexed by the findspot label (column A) and
    holds one column per Arretine service type (columns B-J). Counts are stored
    as pandas' nullable ``Int64`` so that empty cells stay as missing values
    (``<NA>``) rather than being silently coerced to ``0.0`` floats -- keeping
    "absent" and "zero" distinguishable for whoever uses the matrix next.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {xlsx_path}\n"
            f"Expected it at root/src/. Check the file name and location."
        )

    # Read the whole sheet first, then slice explicitly. header=0 uses row 1.
    raw = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME, header=0)

    # Column A -> findspot labels (the identity key for every row).
    labels = raw.iloc[:, 0].astype("string").str.strip()

    # Columns B-J -> the service-type counts.
    services = raw.iloc[:, SERVICE_COL_START:SERVICE_COL_END].copy()

    # Tidy the header labels: strip whitespace and collapse internal doubles.
    services.columns = [" ".join(str(col).split()) for col in services.columns]

    # Attach the findspot labels as a named index.
    services.index = pd.Index(labels, name=LABEL_COLUMN)

    # Drop any fully empty trailing rows the sheet may carry beyond the data.
    services = services.loc[services.index.notna()]
    services = services.dropna(how="all")

    # Store counts as nullable integers (missing stays missing, no float drift).
    services = services.astype("Int64")

    return services


def as_seriation_input(services: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with missing counts filled as 0 for numeric downstream use."""
    return services.fillna(0).astype("int64")


def compute_service_percentages(
    services: pd.DataFrame, decimals: int = 2
) -> pd.DataFrame:
    """Return the per-findspot percentage share of each service type (B-J).

    For every findspot (row) the nine service-type counts are expressed as a
    percentage of that findspot's own total, so each row sums to 100. Missing
    counts are treated as an explicit absence (0). Zero-sum rows (guarded for
    safety) yield all-zero rows rather than a division-by-zero error.
    """
    counts = as_seriation_input(services)          # dense integer matrix, NA -> 0
    row_totals = counts.sum(axis=1)                # total sherds per findspot

    # Divide each count by its row total; guard against any zero-sum rows.
    percentages = counts.div(row_totals.replace(0, pd.NA), axis=0) * 100
    percentages = percentages.fillna(0.0)

    if decimals is not None:
        percentages = percentages.round(decimals)

    return percentages


# ==============================================================================
# SECTION 4 · Percentage lookup
# ==============================================================================
def load_service_percentages():
    """Return the per-findspot service-type percentage matrix (rows sum to 100).

    Wrapper around the inlined reader so the percentage source stays in one
    place. The returned DataFrame is indexed by findspot label with one column
    per service type (columns B-J of the workbook).
    """
    services = load_service_matrix()
    return compute_service_percentages(services, decimals=None)  # exact, unrounded


def write_percentages_csv(pct_df, csv_path: Path, decimals: int = 2):
    """Write the per-findspot percentage matrix to CSV.

    The findspot label becomes the first column ("findspot"); the remaining
    columns are the nine service types (German workbook names). UTF-8 with BOM
    so Excel on Windows renders the umlauts correctly.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = pct_df.round(decimals).copy()
    out.index.name = "findspot"
    out.to_csv(csv_path, encoding="utf-8-sig")
    print(f"✓ Percentages CSV saved: {csv_path}  ({out.shape[0]} findspots)")


def _match_percentages(label: str, pct_df):
    """Return the percentage row for a findspot label, or None if unmatched.

    Tries, in order: the explicit PERCENT_LABEL_CORRECTIONS bridge, an exact
    index match, then a whitespace/case-insensitive match. Returning None lets
    the caller flag the findspot rather than silently colouring it wrongly.
    """
    lookup = PERCENT_LABEL_CORRECTIONS.get(label, label)

    if lookup in pct_df.index:
        return pct_df.loc[lookup]

    # Normalised fallback: collapse whitespace and ignore case.
    def _norm(s: str) -> str:
        return " ".join(str(s).split()).casefold()

    target = _norm(lookup)
    for idx in pct_df.index:
        if _norm(idx) == target:
            return pct_df.loc[idx]

    return None


# ==============================================================================
# SECTION 5 · Timeline coloured by service composition
# ==============================================================================
# Adapted from plot_alligator_events_timeline in alligator_to_clean_rdf.py.
# Bar positions and axis styling are preserved; the bar *fill* is replaced by a
# nine-segment proportional stack, and the in-bar label is the plain findspot
# name (the "-->nfsn,nfen" suffix is dropped).


def plot_events_timeline_by_service(events: dict, pct_df, output_path: Path, lang="en"):
    """Draw the individual-site timeline with bars split by service percentage.

    Parameters
    ----------
    events      : dict            Output of ``load_alligator_events``.
    pct_df      : pandas.DataFrame Per-findspot percentages (columns B-J).
    output_path : Path            Destination JPEG file (a .svg twin is saved too).
    lang        : str             UI language for legend / axis labels ("en"/"fr").
    """
    if not events:
        print("  ⚠ No events to plot — skipping service timeline.")
        return

    s = STRINGS.get(lang, STRINGS["en"])
    service_label = SERVICE_LABELS.get(lang, SERVICE_LABELS["en"])

    service_cols = list(pct_df.columns)  # nine service types, in column order
    service_colours = build_service_colours(service_cols)

    # --- Collect and sort events by estimatedstart, then estimatedend ---
    rows = []
    unmatched = []
    for label, ev in events.items():
        try:
            start = float(ev["estimatedstart"])
            end = float(ev["estimatedend"])
        except (ValueError, TypeError):
            continue

        pct_row = _match_percentages(label, pct_df)
        if pct_row is None:
            unmatched.append(label)

        rows.append(
            {
                "label": label,          # plain findspot name — no "-->" suffix
                "start": start,
                "end": end,
                "pct": pct_row,          # pandas Series or None
            }
        )

    # Same ordering as the original: cluster groups fall together visually.
    rows.sort(key=lambda r: (r["start"], r["end"], r["label"]))

    # Apply the manual row overrides (see TIMELINE_ROW_BELOW in Section 2).
    # `rows` runs bottom → top, so "directly below X in the figure" means
    # "directly before X in this list".
    for label, anchor in TIMELINE_ROW_BELOW.items():
        moved = next((r for r in rows if r["label"] == label), None)
        if moved is None:
            continue
        rows.remove(moved)
        anchor_idx = next(
            (i for i, r in enumerate(rows) if r["label"] == anchor), None
        )
        if anchor_idx is None:
            print(f"  ⚠ Row override skipped: anchor {anchor!r} not in timeline.")
            rows.append(moved)          # put it back rather than lose the bar
            rows.sort(key=lambda r: (r["start"], r["end"], r["label"]))
            continue
        rows.insert(anchor_idx, moved)

    n = len(rows)

    if unmatched:
        print(
            f"  ⚠ {len(unmatched)} findspot(s) had no percentage match and are "
            f"drawn in neutral grey. Add them to PERCENT_LABEL_CORRECTIONS:"
        )
        for lbl in unmatched:
            print(f"      · {lbl!r}")

    bar_height = 0.55
    fig_h = max(6, n * 0.42 + 2)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, row in enumerate(rows):
        duration = row["end"] - row["start"] if row["end"] != row["start"] else 0.3
        x0 = row["start"]

        if row["pct"] is None:
            # No composition data → single neutral bar.
            ax.barh(
                i, duration, left=x0, height=bar_height,
                color=NO_MATCH_COLOUR, edgecolor="#00000018",
                linewidth=0.4, align="center", zorder=2,
            )
        else:
            # Lay the nine service segments left-to-right, widths ∝ percentage.
            seg_left = x0
            for col, colour in zip(service_cols, service_colours):
                share = float(row["pct"].get(col, 0.0) or 0.0)
                if share <= 0:
                    continue
                seg_w = duration * share / 100.0
                ax.add_patch(
                    plt.Rectangle(
                        (seg_left, i - bar_height / 2),
                        seg_w,
                        bar_height,
                        facecolor=colour,
                        edgecolor="none",
                        zorder=2,
                    )
                )
                seg_left += seg_w
            # Thin outline around the whole bar so short bars stay visible.
            ax.add_patch(
                plt.Rectangle(
                    (x0, i - bar_height / 2),
                    duration,
                    bar_height,
                    fill=False,
                    edgecolor="#00000022",
                    linewidth=0.4,
                    zorder=2.1,
                )
            )

        # Findspot name in the left margin (black), linked to the bar by a
        # thin horizontal leader line running along the row.
        ax.annotate(
            row["label"],
            xy=(x0, i),
            xycoords="data",                       # arrow tip at the bar's start
            xytext=(-0.012, i),
            textcoords=ax.get_yaxis_transform(),   # x: axes fraction, y: data
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#000000",
            clip_on=False,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-",
                color="#999999",
                linewidth=0.6,
                shrinkA=3,
                shrinkB=2,
            ),
            zorder=4,
        )

        # Subtle separator between cluster groups (same rule as the original).
        if i > 0 and (
            rows[i]["start"] != rows[i - 1]["start"]
            or rows[i]["end"] != rows[i - 1]["end"]
        ):
            ax.axhline(i - 0.5, color="#cccccc", linewidth=0.6, linestyle="--", zorder=1)

    # --- Axes (identical styling to the original) ---
    all_starts = [r["start"] for r in rows]
    all_ends = [r["end"] for r in rows]
    x_min = min(all_starts) - 2
    x_max = max(all_ends) + 2

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks([])

    x_ticks = list(range(int(x_min), int(x_max) + 1, 5))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [_year_label(t, lang) for t in x_ticks],
        rotation=45, ha="right", fontsize=8, color="#333333",
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="x", which="minor", length=2, color="#cccccc")
    ax.tick_params(axis="x", which="major", length=5, color="#aaaaaa")
    ax.grid(axis="x", which="major", color="#eeeeee", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    # --- Legend: one swatch per service type (localised labels) ---
    legend_patches = [
        mpatches.Patch(color=colour, label=service_label.get(col, col))
        for col, colour in zip(service_cols, service_colours)
    ]
    if unmatched:
        legend_patches.append(mpatches.Patch(color=NO_MATCH_COLOUR, label=s["no_data"]))
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=13,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
        title=s["legend_title"],
        title_fontsize=14,
        handlelength=2.2,
        handleheight=1.4,
        labelspacing=0.7,
        borderpad=1.0,
    )

    # Title intentionally omitted.
    ax.set_xlabel(s["xlabel"], color="#333333", fontsize=9)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output_path), dpi=300, format="jpeg", bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        str(output_path.with_suffix(".svg")),
        format="svg", bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    print(f"✓ Service-composition timeline saved: {output_path}")


# ==============================================================================
# SECTION 5b · Service-group variability & quality (dating-independent)
# ==============================================================================
# For each findspot and each of the three service GROUPS (Schrägrandteller /
# Service I / Service II) we measure how evenly the group's sub-types are
# represented, independent of chronology:
#   · variability = std. dev. of the within-group sub-type shares (normalised to
#     100 % within the group). 0 = perfectly even; larger = one sub-type dominates.
#   · quality     = Pielou's evenness J = H / ln(k) of those shares, in [0, 1].
#     1 = sub-types evenly represented (consistent); 0 = dominated by one.
# Schrägrandteller has a single sub-type, so both are undefined (shown as "—").


def _group_members(service_cols: list[str]) -> dict[str, list[str]]:
    """Map each of the three service groups to its member columns (ordered)."""
    groups: dict[str, list[str]] = {
        "Schrägrandteller": [],
        "Service I": [],
        "Service II": [],
    }
    for c in service_cols:
        if c.startswith("Schrägrandteller"):
            groups["Schrägrandteller"].append(c)
        elif c.startswith("Service II"):  # check II before I
            groups["Service II"].append(c)
        elif c.startswith("Service I"):
            groups["Service I"].append(c)
    return groups


def _rgzm_variance_quality(group_counts: np.ndarray, positions: np.ndarray):
    """RGZM variance/quality over the group-rank of each sherd.

    Following Allard Mees' RGZM method (variance_quality_from_sql), every sherd
    is one observation whose value is the chronological rank of its service
    group (``positions``). Over the N sherds of a horizon:

        x̄  = Σ pos_g · G_g / N                          (mean rank, AVG)
        s  = sqrt( Σ G_g · (pos_g − x̄)² / (N − 1) )     (STDDEV_SAMP, ddof=1)
        CV = s / |x̄|                                    (coefficient of variation)
        q  = exp(−CV)                                   (quality, in (0, 1])

    q → 1 : the assemblage is chronologically concentrated on one rank
            (tightly defined); q → 0 : it is spread across the group sequence.
    Returns (mean, s, cv, q, N). Undefined (nan) when N < 2 or x̄ = 0.
    """
    G = np.asarray(group_counts, dtype=float)
    N = float(G.sum())
    if N < 2:
        return (np.nan, np.nan, np.nan, np.nan, int(N))
    mean = float((positions * G).sum() / N)                 # AVG
    var_samp = float((G * (positions - mean) ** 2).sum() / (N - 1))  # VAR_SAMP
    s = float(np.sqrt(var_samp))                            # STDDEV_SAMP
    if mean == 0:
        return (mean, s, np.nan, np.nan, int(N))
    cv = s / abs(mean)
    q = float(np.exp(-cv))
    return (mean, s, cv, q, int(N))


def compute_within_group_rgzm(counts_df):
    """Per horizon and per GROUP: RGZM variance/quality *within* the group.

    Within each group the sub-types are the observations: every sherd is valued
    by the rank of its sub-type inside its own group (column order → 1..k). The
    RGZM measures are then computed over those within-group ranks (see
    ``_rgzm_variance_quality``):
        s = STDDEV_SAMP(sub-type ranks)      (variance within the group)
        q = exp(−CV), CV = s / |x̄|           (quality within the group)
    q → 1: the group's material is concentrated in few sub-types; q → 0: it is
    spread across the group's sub-type sequence.

    Returns (var_df, qual_df, ncell_df, n_horizon) indexed by horizon; the first
    three have one column per group.
    """
    groups = _group_members(list(counts_df.columns))
    group_names = list(groups)
    horizons = sorted(set(HORIZON_OF.values()))
    counts0 = counts_df.fillna(0)

    var_df = pd.DataFrame(index=horizons, columns=group_names, dtype=float)
    qual_df = pd.DataFrame(index=horizons, columns=group_names, dtype=float)
    ncell_df = pd.DataFrame(index=horizons, columns=group_names, dtype="int64")
    n_horizon = pd.Series(index=horizons, dtype="int64")

    for h in horizons:
        members = [f for f in counts0.index if resolve_horizon(f) == h]
        pooled = counts0.loc[members].sum(axis=0)
        n_horizon[h] = int(pooled.sum())
        for g in group_names:
            sub = pooled[groups[g]].to_numpy(dtype=float)   # sub-type counts
            ranks = np.arange(1, len(sub) + 1, dtype=float)  # within-group ranks
            _, s, _, q, N = _rgzm_variance_quality(sub, ranks)
            var_df.loc[h, g] = s
            qual_df.loc[h, g] = q
            ncell_df.loc[h, g] = int(N)

    for df in (var_df, qual_df, ncell_df):
        df.index.name = "horizon"
    n_horizon.index.name = "horizon"
    return var_df, qual_df, ncell_df, n_horizon


def write_group_spread_csv(var_df, qual_df, ncell_df, csv_path: Path):
    """Write within-group variance, quality and n per horizon per group to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(index=var_df.index)
    out.index.name = var_df.index.name or "horizon"
    for g in var_df.columns:
        out[f"{g} — n"] = ncell_df[g].astype(int)
        out[f"{g} — variance (STDDEV_SAMP)"] = var_df[g].round(3)
        out[f"{g} — quality exp(-CV)"] = qual_df[g].round(3)
    out.to_csv(csv_path, encoding="utf-8-sig")
    print(f"✓ Within-group variance/quality CSV saved: {csv_path}  ({out.shape[0]} rows)")


def plot_within_group_heatmap(
    var_df, qual_df, ncell_df, output_path: Path, lang="en"
):
    """Paired heatmap of the within-group variance (left) and quality (right).

    Rows are the five horizons (Horizon 1 at the top), columns the three service
    groups. Left panel colours the within-group variance s (STDDEV_SAMP over the
    group's sub-type ranks); right panel the within-group quality q = exp(−CV).
    Each cell prints the value and, in small grey, the group's sherd count n.
    Undefined cells (group absent, or n < 2) show "—" in light grey.
    """
    s = STRINGS.get(lang, STRINGS["en"])
    gdisp = GROUP_DISPLAY.get(lang, GROUP_DISPLAY["en"])

    groups = list(var_df.columns)
    horizons = list(var_df.index)
    n_rows = len(horizons)
    col_labels = [gdisp.get(g, g) for g in groups]

    var_arr = np.ma.masked_invalid(var_df.to_numpy(dtype=float))
    qual_arr = np.ma.masked_invalid(qual_df.to_numpy(dtype=float))

    cmap_var = matplotlib.colormaps["cividis"].copy()
    cmap_qual = matplotlib.colormaps["RdYlGn"].copy()  # red = low, green = high
    cmap_var.set_bad("#eeeeee")
    cmap_qual.set_bad("#eeeeee")

    fig_h = max(3.6, n_rows * 0.72 + 2.4)
    fig, (ax_v, ax_q) = plt.subplots(
        1, 2, figsize=(12, fig_h), sharey=True,
        gridspec_kw={"wspace": 0.08}, layout="constrained",
    )
    fig.patch.set_facecolor("white")

    def _draw(ax, arr, cmap, vmin, vmax, title, fmt, source_df):
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(col_labels, fontsize=12)
        for tick, g in zip(ax.get_xticklabels(), groups):
            tick.set_color(GROUP_TINT.get(g, "#000000"))
            tick.set_fontweight("bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=24, color="#111111")
        ax.set_xlim(-0.5, len(groups) - 0.5)
        norm = im.norm
        for i in range(n_rows):
            for j, g in enumerate(groups):
                v = source_df.iat[i, j]
                ng = int(ncell_df.iat[i, j])
                if pd.isna(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=13, color="#888888")
                    continue
                r, gg, b, _ = cmap(norm(v))
                lum = 0.299 * r + 0.587 * gg + 0.114 * b
                tcol = "#111111" if lum > 0.55 else "#ffffff"
                ax.text(j, i - 0.13, fmt(v), ha="center", va="center",
                        fontsize=13, color=tcol)
                ax.text(j, i + 0.22, f"n={ng}", ha="center", va="center",
                        fontsize=8, color=tcol, alpha=0.8)
        ax.set_xticks(np.arange(-0.5, len(groups), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, location="bottom")
        cbar.ax.tick_params(labelsize=9)
        return im

    var_vmax = float(np.nanmax(var_df.to_numpy(dtype=float))) or 1.0
    _draw(ax_v, var_arr, cmap_var, 0.0, var_vmax, s["var_panel"],
          lambda v: f"{v:.2f}", var_df)
    _draw(ax_q, qual_arr, cmap_qual, 0.0, 1.0, s["qual_panel"],
          lambda v: f"{v:.2f}", qual_df)

    ax_v.set_yticks(range(n_rows))
    ax_v.set_yticklabels([f"{s['horizon']} {h}" for h in horizons],
                         fontsize=11, fontweight="bold")

    foot_fs = 9
    wrap_chars = max(80, int(fig.get_figwidth() * 72 / (foot_fs * 0.55)))
    note_lines = [textwrap.fill(s["spread_note"], width=wrap_chars), ""]
    for h in horizons:
        prefix = f"{s['horizon']} {h}: "
        members = " · ".join(HORIZONS.get(h, []))
        note_lines.append(
            textwrap.fill(members, width=wrap_chars,
                          initial_indent=prefix,
                          subsequent_indent=" " * len(prefix))
        )
    fig.text(0.01, -0.02, "\n".join(note_lines),
             fontsize=foot_fs, color="#555555", ha="left", va="top",
             linespacing=1.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, format="jpeg",
                bbox_inches="tight", facecolor="white")
    fig.savefig(str(output_path.with_suffix(".svg")), format="svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Within-group heatmap saved: {output_path}")


# ==============================================================================
# SECTION 6 · Main Entry Point
# ==============================================================================
def main():
    """Load data, write CSVs, and render EN + FR figures."""
    events = load_alligator_events(TTL_FILE)

    counts_df = load_service_matrix()
    pct_df = compute_service_percentages(counts_df, decimals=None)

    # --- CSV deliverables ---
    write_percentages_csv(pct_df, PERCENT_CSV)

    var_df, qual_df, ncell_df, n_horizon = compute_within_group_rgzm(counts_df)
    write_group_spread_csv(var_df, qual_df, ncell_df, GROUP_VAR_CSV)

    # --- Timeline by service composition (EN + FR) ---
    plot_events_timeline_by_service(
        events, pct_df, OUTPUT_DIR / "events_timeline_by_service_en.jpg", lang="en"
    )
    plot_events_timeline_by_service(
        events, pct_df, OUTPUT_DIR / "events_timeline_by_service_fr.jpg", lang="fr"
    )

    # --- Within-group variance/quality per horizon per group (EN + FR) ---
    plot_within_group_heatmap(
        var_df, qual_df, ncell_df,
        OUTPUT_DIR / "service_group_variability_en.jpg", lang="en"
    )
    plot_within_group_heatmap(
        var_df, qual_df, ncell_df,
        OUTPUT_DIR / "service_group_variability_fr.jpg", lang="fr"
    )


if __name__ == "__main__":
    main()