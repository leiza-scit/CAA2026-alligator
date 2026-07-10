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

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — no display required
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
LABEL_COLUMN = "findspot"  # column A carries no header; we name it explicitly
SERVICE_COL_START = 1  # column B, 0-based
SERVICE_COL_END = 10  # column J inclusive -> slice stop is exclusive

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
BLUE_COLOUR = "#1f77b4"  # Schrägrandteller
RED_CMAP = "Reds"  # Service I family
GREEN_CMAP = "Greens"  # Service II family
RED_RANGE = (0.35, 0.90)  # sampled span of the Reds colormap (avoids white)
GREEN_RANGE = (0.45, 0.88)  # sampled span of the Greens colormap

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
    },
    "fr": {
        "legend_title": "Type de service (part de l'assemblage)",
        "xlabel": "Année",
        "no_data": "pas de données",
    },
}

# Era suffixes for year tick labels, per language.
ERA_LABELS = {
    "en": {"bc": "BC", "ad": "AD"},  # 15 BC / AD 9
    "fr": {"bc": "av. J.-C.", "ad": "apr. J.-C."},  # 15 av. J.-C. / 9 apr. J.-C.
}

# CSV of the computed per-findspot service percentages.
PERCENT_CSV = OUTPUT_DIR / "service_percentages.csv"


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
        1
        for c in service_cols
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
    counts = as_seriation_input(services)  # dense integer matrix, NA -> 0
    row_totals = counts.sum(axis=1)  # total sherds per findspot

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
                "label": label,  # plain findspot name — no "-->" suffix
                "start": start,
                "end": end,
                "pct": pct_row,  # pandas Series or None
            }
        )

    # Same ordering as the original: cluster groups fall together visually.
    rows.sort(key=lambda r: (r["start"], r["end"], r["label"]))
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
                i,
                duration,
                left=x0,
                height=bar_height,
                color=NO_MATCH_COLOUR,
                edgecolor="#00000018",
                linewidth=0.4,
                align="center",
                zorder=2,
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
            xycoords="data",  # arrow tip at the bar's start
            xytext=(-0.012, i),
            textcoords=ax.get_yaxis_transform(),  # x: axes fraction, y: data
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
            ax.axhline(
                i - 0.5, color="#cccccc", linewidth=0.6, linestyle="--", zorder=1
            )

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
        rotation=45,
        ha="right",
        fontsize=8,
        color="#333333",
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
        format="svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"✓ Service-composition timeline saved: {output_path}")


# ==============================================================================
# SECTION 6 · Main Entry Point
# ==============================================================================
def main():
    """Load events + percentages, write the CSV, and render EN + FR timelines."""
    events = load_alligator_events(TTL_FILE)
    pct_df = load_service_percentages()

    # Computed percentages as a CSV deliverable.
    write_percentages_csv(pct_df, PERCENT_CSV)

    # One figure per language (JPG + SVG each), mirroring alligator_to_clean_rdf.py.
    plot_events_timeline_by_service(
        events, pct_df, OUTPUT_DIR / "events_timeline_by_service_en.jpg", lang="en"
    )
    plot_events_timeline_by_service(
        events, pct_df, OUTPUT_DIR / "events_timeline_by_service_fr.jpg", lang="fr"
    )


if __name__ == "__main__":
    main()
