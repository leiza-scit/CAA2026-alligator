"""Assign a new assemblage to a chronological horizon from its composition.

The variance/quality note reads the sequence in one direction: given a horizon,
what are its proportions? This script reads it in the other direction, which is
the question excavators actually ask. Given a bag of sherds counted by service
type, which of the five horizons does it belong to, and how sure is that?

The method is the standard assignment of a composition to one of several
reference distributions. What is new here is only the reference: the five
horizon profiles come out of the Alligator seriation and are therefore
available in this form for the first time.

Method
------
Five categories are used: Schrägrandteller, Service Ia, Ib, Ic and Service II.
Cup and plate of the same stage are merged, exactly as in the rank assignment
of the variance note — a change of form is not a step in time. Stages Ia to Ic
carry most of the discriminating power between neighbouring horizons; the three
groups on their own (the Figure 20 view) are demonstrably not enough.

Four steps, each exposed as its own function so the arithmetic can be followed
and checked by hand:

    1. proportions -> counts               scale_to_counts()
    2. counts -> log-likelihood per horizon  log_likelihood()
    3. log-likelihood -> probability         posterior_plugin()
    4. the same, allowing for the fact that
       the reference is itself a sample      posterior_dirichlet()
    5. does the winner fit at all?           goodness_of_fit()

Steps 1 to 4 are relative: the five probabilities always sum to 100 %, so a
mixed context or an assemblage from outside the seriated area is still handed a
horizon, and handed it confidently. Step 5 is the absolute question, and
chi-square is the right tool for it — the same chi-square, in its
likelihood-ratio form G², that steps 2 and 3 already compute under another name.

Step 4 is the honest one and the default for every reported figure. Step 3 is
kept because it can be reproduced with a pocket calculator, which matters for a
method that is meant to be used rather than believed.

Validation
----------
``leave_one_out()`` removes each of the 44 seriated findspots from its own
horizon, rebuilds the reference from the remaining 43 and then treats it as if
it were a new find. It recovers the seriation's own horizon in 34 cases and
lands in the correct or an immediately neighbouring horizon in all 44.

Inputs
------
* root/src/ArretineDatedSitesServicesI_II.xlsx   (service-type counts, B-J)
* root/py/horizons.py                            (findspot -> horizon)

Outputs (root/output/)
----------------------
* horizon_reference_profiles.csv   pooled counts and proportions per horizon
* horizon_assignment_examples.csv  the worked examples, both methods, three n
* horizon_assignment_loo.csv       leave-one-out result for every findspot

Usage
-----
    python py/horizon_assignment.py
    python py/horizon_assignment.py --assign "Ib=13,Ic=27,II=60" --n 100
    python py/horizon_assignment.py --assign "Ib=6,Ic=24,II=70" --n 100 --steps
    python py/horizon_assignment.py --no-write

It is registered in py/main.py as the "assign" step, so a full pipeline run
produces the three CSVs alongside everything else:

    python py/main.py --only assign
"""

from __future__ import annotations

# ==============================================================================
# SECTION 1 · Imports
# ==============================================================================

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# The sys.path line makes the sibling modules resolvable no matter which working
# directory the script is launched from (VS Code often uses the repository root).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wd_paths  # noqa: E402
from horizons import (  # noqa: E402  (shared horizon definition)
    HORIZON_NUMBERS,
    resolve_horizon,
)


# ==============================================================================
# SECTION 2 · Configuration
# ==============================================================================

XLSX_FILE = wd_paths.SRC / "ArretineDatedSitesServicesI_II.xlsx"
SHEET_NAME = "ArretineDatedSitesServicesI_II"
SERVICE_COL_START = 1          # column B, 0-based
SERVICE_COL_END = 10           # column J inclusive -> slice stop is exclusive

# The five categories, in sequence order. Cup and plate of one stage are merged;
# see the module docstring for why.
CATEGORIES = ["Schrägrandteller", "Ia", "Ib", "Ic", "Service II"]

# Workbook column (whitespace-collapsed) -> category. Anything not listed here
# would be dropped silently, so the reader checks the mapping is exhaustive.
CATEGORY_OF_COLUMN = {
    "Schrägrandteller": "Schrägrandteller",
    "Service Ia Tasse": "Ia",
    "Service Ia Teller": "Ia",
    "Service Ib Tasse": "Ib",
    "Service Ib Teller": "Ib",
    "Service Ic Tasse": "Ic",
    "Service Ic Teller": "Ic",
    "Service II Tasse": "Service II",
    "Service II Teller": "Service II",
}

# Dirichlet concentration added to every reference count in step 4. One is the
# uniform (Laplace) prior: weakly informative, and it keeps an empty reference
# category from excluding a horizon with probability exactly zero.
DIRICHLET_ALPHA = 1.0

# Expected counts below this make the chi-square distribution an unsafe
# reference for the fit test. The rare stages here fall well under it, which is
# exactly why the fit test is reported with a flag rather than as a bare p-value.
MIN_EXPECTED = 5.0

# Below this p-value the leading horizon is reported as not fitting.
FIT_ALPHA = 0.05

# The worked examples of the accompanying note. Proportions in per cent.
EXAMPLES = {
    "A": {"label": "13 / 27 / 60", "shares": {"Ib": 13, "Ic": 27, "Service II": 60}},
    "B": {"label": "6 / 24 / 70", "shares": {"Ib": 6, "Ic": 24, "Service II": 70}},
    "C": {"label": "2 / 8 / 90", "shares": {"Ib": 2, "Ic": 8, "Service II": 90}},
}
EXAMPLE_SIZES = (30, 100, 300)

# A deliberately mixed assemblage: a late context carrying early residual
# material. Every horizon is a poor fit for it, which is precisely what the fit
# test is meant to catch and what the horizon probabilities cannot say.
FIT_DEMO = {
    "label": "5 / 5 / 8 / 82 (with Ia)",
    "shares": {"Ia": 5, "Ib": 5, "Ic": 8, "Service II": 82},
}

# Confidence thresholds for "how many sherds do you need?", and the largest
# assemblage worth asking about. Beyond roughly the size of the reference the
# answer describes the reference rather than the find — see the note, ch. 8.2.
THRESHOLDS = (0.90, 0.95, 0.99)
MAX_SHERDS_SEARCHED = 1000

REFERENCE_CSV = wd_paths.OUTPUT / "horizon_reference_profiles.csv"
EXAMPLES_CSV = wd_paths.OUTPUT / "horizon_assignment_examples.csv"
LOO_CSV = wd_paths.OUTPUT / "horizon_assignment_loo.csv"


# ==============================================================================
# SECTION 3 · Reference profiles
# ==============================================================================
# The workbook reader is inlined rather than imported, following the pattern of
# events_timeline_by_service.py: this script then has no local-module
# dependency beyond the two shared helpers and runs from any directory.


def load_findspot_counts(xlsx_path: Path = XLSX_FILE) -> pd.DataFrame:
    """Read columns B-J and fold them into the five chronological categories.

    Returns a findspot × category matrix of integer counts. Empty cells are an
    explicit absence and become 0; unlike the abundance matrix used by the
    figures, nothing downstream here needs to tell "absent" from "zero".
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {xlsx_path}\n"
            f"Expected it at root/src/. Check the file name and location."
        )

    raw = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME, header=0)

    labels = raw.iloc[:, 0].astype("string").str.strip()
    services = raw.iloc[:, SERVICE_COL_START:SERVICE_COL_END].copy()
    services.columns = [" ".join(str(col).split()) for col in services.columns]
    services.index = pd.Index(labels, name="findspot")
    services = services.loc[services.index.notna()].dropna(how="all")
    services = services.fillna(0).astype("int64")

    unmapped = [c for c in services.columns if c not in CATEGORY_OF_COLUMN]
    if unmapped:
        raise ValueError(
            "Workbook columns not covered by CATEGORY_OF_COLUMN: "
            + ", ".join(unmapped)
            + "\nAdd them there rather than letting them vanish from the totals."
        )

    folded = pd.DataFrame(0, index=services.index, columns=CATEGORIES, dtype="int64")
    for column, category in CATEGORY_OF_COLUMN.items():
        if column in services.columns:
            folded[category] += services[column]

    return folded


def pool_by_horizon(findspots: pd.DataFrame) -> pd.DataFrame:
    """Sum the findspot counts into one row per horizon.

    Findspots the correspondence analysis could not place carry no horizon and
    are left out; the difference between the workbook total and the total here
    is exactly their material (see FINDSPOT_EXCLUSIONS in horizons.py).
    """
    horizon = findspots.index.map(resolve_horizon)
    assigned = findspots[pd.notna(horizon)].copy()
    assigned["horizon"] = [h for h in horizon if pd.notna(h)]

    pooled = assigned.groupby("horizon")[CATEGORIES].sum()
    pooled.index = pooled.index.astype(int)
    return pooled.reindex(HORIZON_NUMBERS).fillna(0).astype("int64")


def reference_profiles(pooled: pd.DataFrame) -> pd.DataFrame:
    """Turn pooled counts into the proportions p(h) the assignment runs on."""
    totals = pooled.sum(axis=1)
    return pooled.div(totals, axis=0)


# ==============================================================================
# SECTION 4 · The four calculation steps
# ==============================================================================


def scale_to_counts(shares: dict, n: int) -> dict:
    """Step 1 — turn percentage shares into whole sherd counts summing to n.

    Percentages carry no sample size, and sample size is the whole question:
    60 % from 3 sherds says almost nothing, 60 % from 600 says a great deal.
    Largest-remainder rounding keeps the total at exactly n, so the example
    tables are internally consistent even at n = 30.
    """
    exact = {cat: shares.get(cat, 0) / 100 * n for cat in CATEGORIES}
    counts = {cat: int(exact[cat]) for cat in CATEGORIES}

    shortfall = n - sum(counts.values())
    by_remainder = sorted(CATEGORIES, key=lambda c: -(exact[c] - counts[c]))
    for cat in by_remainder[:shortfall]:
        counts[cat] += 1

    return counts


def log_likelihood(counts: dict, profiles: pd.DataFrame) -> dict:
    """Step 2 — ln L(h) for every horizon, treating the bag as a draw from it.

    ln L(h) = Σ x_i · ln p(h)_i. Categories the find has nothing in contribute
    no term, but their absence still tells: the proportions sum to one, so a
    horizon carrying much Ia has correspondingly less mass left for everything
    else and is penalised automatically when Ia is missing.

    The multinomial coefficient is identical for all five horizons and cancels
    in step 3, so it is omitted. A horizon whose reference is empty in a
    category the find does have is impossible under this step and returns
    negative infinity; step 4 softens that to "very unlikely".
    """
    result = {}
    for horizon in profiles.index:
        total = 0.0
        for category, observed in counts.items():
            if observed <= 0:
                continue
            p = float(profiles.at[horizon, category])
            if p <= 0:
                total = -math.inf
                break
            total += observed * math.log(p)
        result[int(horizon)] = total
    return result


def _normalise(log_values: dict) -> dict:
    """Turn log-scale scores into probabilities in per cent, summing to 100.

    Subtracting the maximum before exponentiating is what keeps this from
    underflowing: the differences are all that matter, and they are small.
    """
    best = max(log_values.values())
    weights = {
        h: (math.exp(v - best) if v != -math.inf else 0.0)
        for h, v in log_values.items()
    }
    total = sum(weights.values())
    return {h: 100 * w / total for h, w in weights.items()}


def posterior_plugin(counts: dict, profiles: pd.DataFrame) -> dict:
    """Step 3 — probability per horizon, with a uniform prior over horizons.

    P(h | find) = L(h) / Σ L(h'). "Plug-in" because the reference proportions
    are plugged in as if exactly known. They are not; see step 4.
    """
    return _normalise(log_likelihood(counts, profiles))


def posterior_dirichlet(
    counts: dict, pooled: pd.DataFrame, alpha: float = DIRICHLET_ALPHA
) -> dict:
    """Step 4 — the same, allowing for the reference being a sample too.

    Horizon 4 rests on 476 sherds, horizon 2 on 2,706; a proportion of 6.7 %
    from the former is far less certain than the same figure from the latter.
    The Dirichlet-multinomial replaces each fixed proportion with the whole
    range of proportions the reference counts are compatible with, which damps
    the result exactly where the plug-in version is most overconfident.

    Its one trap: as n approaches the size of the reference itself, the answer
    starts to report the reference's uncertainty rather than the find. Keep
    assemblages well below the reference totals — see the note, ch. 8.2.
    """
    n = sum(counts.values())
    result = {}
    for horizon in pooled.index:
        a = {cat: float(pooled.at[horizon, cat]) + alpha for cat in CATEGORIES}
        A = sum(a.values())
        total = math.lgamma(A) - math.lgamma(A + n)
        for category in CATEGORIES:
            observed = counts.get(category, 0)
            total += math.lgamma(a[category] + observed) - math.lgamma(a[category])
        result[int(horizon)] = total
    return _normalise(result)


def assign(counts: dict, pooled: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    """Run both methods over one assemblage and return them side by side."""
    return {
        "plugin": posterior_plugin(counts, profiles),
        "dirichlet": posterior_dirichlet(counts, pooled),
    }


# ==============================================================================
# SECTION 5 · Goodness of fit
# ==============================================================================


def _regularised_gamma_p(a: float, x: float) -> float:
    """Series expansion of the lower regularised incomplete gamma P(a, x)."""
    term = total = 1.0 / a
    ap = a
    for _ in range(1000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _regularised_gamma_q(a: float, x: float) -> float:
    """Continued fraction for the upper regularised incomplete gamma Q(a, x)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if b != 0.0 else 1.0 / tiny
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi_square_sf(x: float, df: int) -> float:
    """Return P(chi-square with df degrees of freedom exceeds x).

    Written out rather than taken from scipy, which nothing else in this
    pipeline needs and which would be a heavy dependency for one distribution
    function. It agrees with ``scipy.stats.chi2.sf`` to ten significant figures
    across the range that arises here. The series is used below the mode of the
    distribution and the continued fraction above it, each where it converges
    quickly.
    """
    if df <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    if math.isinf(x):
        return 0.0
    a, y = df / 2.0, x / 2.0
    if y < a + 1.0:
        return 1.0 - _regularised_gamma_p(a, y)
    return _regularised_gamma_q(a, y)


def expected_counts(counts: dict, profiles: pd.DataFrame, horizon: int) -> dict:
    """Return the counts this horizon predicts for an assemblage of this size."""
    n = sum(counts.values())
    return {cat: n * float(profiles.at[horizon, cat]) for cat in CATEGORIES}


def g_squared(counts: dict, profiles: pd.DataFrame, horizon: int) -> float:
    """Return the likelihood-ratio form of chi-square, 2 Σ O · ln(O / E).

    Included because it makes a relation visible that is otherwise only
    assertable: the difference in G² between two horizons is exactly twice the
    difference in their log-likelihoods, so normalising exp(-G²/2) over the five
    horizons reproduces posterior_plugin() to the last digit. Pearson's
    chi-square is the second-order approximation of this same quantity, which is
    why it ranks the horizons identically but scores them differently — and why
    the exact likelihood is preferred here, since it costs nothing extra.
    """
    expected = expected_counts(counts, profiles, horizon)
    total = 0.0
    for category in CATEGORIES:
        observed = counts.get(category, 0)
        if observed <= 0:
            continue
        if expected[category] <= 0:
            return math.inf
        total += 2 * observed * math.log(observed / expected[category])
    return total


def goodness_of_fit(counts: dict, profiles: pd.DataFrame, horizon: int) -> dict:
    """Test an assemblage against one horizon with Pearson's chi-square.

    Categories the horizon cannot produce at all are dropped from the sum; if
    the assemblage has sherds in one of them, the fit is impossible and the
    statistic is infinite. Degrees of freedom are the number of contributing
    categories minus one.

    The ``reliable`` flag is not decoration. The chi-square distribution is a
    large-sample approximation needing expected counts of roughly five per cell,
    and the rare stages here — the very ones carrying the chronological signal —
    fall well below that. Where the flag is False the statistic still orders the
    horizons sensibly, but its p-value should not be quoted as one.
    """
    expected = expected_counts(counts, profiles, horizon)
    contributions = {}
    statistic = 0.0
    cells = 0
    smallest = math.inf
    impossible = False

    for category in CATEGORIES:
        observed = counts.get(category, 0)
        e = expected[category]
        if e <= 0:
            if observed > 0:
                impossible = True
            continue
        cells += 1
        smallest = min(smallest, e)
        contribution = (observed - e) ** 2 / e
        statistic += contribution
        contributions[category] = {
            "observed": observed,
            "expected": e,
            "contribution": contribution,
        }

    if impossible:
        statistic = math.inf

    df = max(cells - 1, 1)
    p = chi_square_sf(statistic, df)

    return {
        "horizon": int(horizon),
        "chi_square": statistic,
        "df": df,
        "p": p,
        "fits": p >= FIT_ALPHA,
        "reliable": smallest >= MIN_EXPECTED and not impossible,
        "smallest_expected": smallest,
        "contributions": contributions,
    }


def _format_p(p: float) -> str:
    """Format a p-value for a table cell, without pretending to precision."""
    return f"{p:.4f}" if p >= 0.0001 else "< 0.0001"


def _format_p_clause(p: float) -> str:
    """Format a p-value as a clause, so "p < 0.0001" does not read "p = < ..."."""
    return f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"


def _format_statistic(x: float) -> str:
    """Format a chi-square value, keeping an impossible fit legible."""
    return "inf" if math.isinf(x) else f"{x:.2f}"

# ==============================================================================
# SECTION 6 · Derivation, printed step by step
# ==============================================================================


def print_derivation(counts: dict, pooled: pd.DataFrame, profiles: pd.DataFrame):
    """Print the arithmetic of steps 2 to 4 in full, for checking by hand.

    This is the console form of tables 4 to 6 of the accompanying note. Every
    number can be reproduced with a calculator from the columns shown.
    """
    present = [c for c in CATEGORIES if counts.get(c, 0) > 0]
    n = sum(counts.values())

    print()
    print(f"Assemblage: n = {n}   " + "  ".join(f"{c} = {counts[c]}" for c in present))

    print()
    print("Step 2 · log-likelihood per horizon")
    header = "  horizon " + "".join(f"{c + ' term':>26}" for c in present) + f"{'ln L':>14}"
    print(header)
    log_values = log_likelihood(counts, profiles)
    for horizon in profiles.index:
        row = f"  H{int(horizon):<7}"
        for category in present:
            p = float(profiles.at[horizon, category])
            term = counts[category] * math.log(p) if p > 0 else -math.inf
            row += f"{counts[category]}·ln {p:.4f} = {term:9.3f}".rjust(26)
        row += f"{log_values[int(horizon)]:14.3f}"
        print(row)

    best = max(log_values.values())
    print()
    print("Step 3 · normalised over the five horizons (uniform prior)")
    print(f"  {'horizon':<10}{'ln L':>12}{'ln L - max':>14}{'L / L(max)':>16}{'P(h)':>10}")
    plugin = posterior_plugin(counts, profiles)
    for horizon in sorted(plugin):
        delta = log_values[horizon] - best
        ratio = math.exp(delta) if delta > -700 else 0.0
        print(
            f"  H{horizon:<9}{log_values[horizon]:12.3f}{delta:14.3f}"
            f"{ratio:16.8f}{plugin[horizon]:9.2f}%"
        )

    print()
    print("Step 4 · allowing for the uncertainty of the reference")
    dirichlet = posterior_dirichlet(counts, pooled)
    for horizon in sorted(dirichlet):
        print(f"  H{horizon:<9}{dirichlet[horizon]:9.2f}%")

    leader = max(dirichlet, key=dirichlet.get)
    runner_up = sorted(dirichlet, key=dirichlet.get, reverse=True)[1]
    print()
    print(
        f"  → horizon {leader} at {dirichlet[leader]:.1f} %, "
        f"next best horizon {runner_up} at {dirichlet[runner_up]:.1f} %"
    )

    print()
    print(f"Step 5 · does horizon {leader} fit at all?")
    fit = goodness_of_fit(counts, profiles, leader)
    print(f"  {'category':<18}{'observed':>10}{'expected':>10}{'chi2 term':>12}")
    for category, part in fit["contributions"].items():
        print(
            f"  {category:<18}{part['observed']:>10}{part['expected']:>10.2f}"
            f"{part['contribution']:>12.2f}"
        )
    print(
        f"  chi-square = {_format_statistic(fit['chi_square'])}, "
        f"df = {fit['df']}, {_format_p_clause(fit['p'])}"
    )
    print()
    if fit["fits"]:
        print("  The observed counts are consistent with this horizon.")
    else:
        print(
            "  The observed counts are NOT consistent with this horizon. Since "
            "it is the best\n  of the five, no horizon fits: expect a mixed "
            "context, residual material, or a\n  site outside the seriated "
            "area. This is the one thing the percentages above\n  cannot tell "
            "you — they always sum to 100 % and always name a winner."
        )
    if not fit["reliable"]:
        print(
            f"  Caution: the smallest expected count is "
            f"{fit['smallest_expected']:.2f}, below the customary {MIN_EXPECTED:g}. "
            f"The\n  statistic still orders the horizons sensibly, but treat the "
            f"p-value as indicative."
        )


# ==============================================================================
# SECTION 7 · Leave-one-out validation
# ==============================================================================


def leave_one_out(findspots: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    """Re-assign every seriated findspot against a reference built without it.

    Leaving the findspot in its own reference would let it vote for itself,
    which for a large findspot such as Neuss all but guarantees a hit and
    proves nothing. Removing it first turns the exercise into a genuine test:
    each findspot is treated exactly as a new find would be.

    Returns one row per findspot with the seriation's horizon, the horizon the
    model picks, both probabilities and the assemblage size.
    """
    rows = []
    for label, counts_row in findspots.iterrows():
        assigned = resolve_horizon(label)
        if assigned is None:
            continue

        counts = {cat: int(counts_row[cat]) for cat in CATEGORIES}
        reduced = pooled.copy()
        for category in CATEGORIES:
            reduced.at[assigned, category] -= counts[category]

        posterior = posterior_dirichlet(counts, reduced)
        predicted = max(posterior, key=posterior.get)

        # The fit is tested against the same reduced reference, so a find-spot
        # is never judged against a profile it helped to define.
        fit = goodness_of_fit(counts, reference_profiles(reduced), predicted)

        rows.append({
            "findspot": label,
            "n": sum(counts.values()),
            "horizon_seriation": assigned,
            "horizon_model": predicted,
            "p_seriation": round(posterior[assigned], 4),
            "p_model": round(posterior[predicted], 4),
            "hit": predicted == assigned,
            "offset": predicted - assigned,
            "chi_square": round(fit["chi_square"], 3),
            "df": fit["df"],
            "p_fit": round(fit["p"], 5),
            "fits": fit["fits"],
        })

    return pd.DataFrame(rows).sort_values(
        ["horizon_seriation", "findspot"]
    ).reset_index(drop=True)


def summarise_leave_one_out(loo: pd.DataFrame) -> dict:
    """Reduce the leave-one-out table to the three figures worth reporting."""
    total = len(loo)
    exact = int(loo["hit"].sum())
    adjacent = int((loo["offset"].abs() <= 1).sum())
    misfits = int((~loo["fits"]).sum())
    return {
        "total": total,
        "exact": exact,
        "exact_pct": 100 * exact / total,
        "adjacent": adjacent,
        "adjacent_pct": 100 * adjacent / total,
        "misfits": misfits,
    }


# ==============================================================================
# SECTION 8 · How many sherds are needed
# ==============================================================================


def sherds_needed(
    shares: dict,
    pooled: pd.DataFrame,
    thresholds=THRESHOLDS,
    limit: int = MAX_SHERDS_SEARCHED,
) -> dict:
    """Smallest assemblage at which the leading horizon passes each threshold.

    Proportions are held fixed and only n is varied, so this answers the
    practical question: how much determinable material must a context yield
    before working it up is worth the effort? Returns None where the threshold
    is not reached within ``limit`` sherds.
    """
    found = {}
    for n in range(5, limit + 1):
        counts = scale_to_counts(shares, n)
        posterior = posterior_dirichlet(counts, pooled)
        leader = max(posterior, key=posterior.get)
        for threshold in thresholds:
            if threshold not in found and posterior[leader] >= threshold * 100:
                found[threshold] = {"n": n, "horizon": leader}
    return {t: found.get(t) for t in thresholds}


# ==============================================================================
# SECTION 9 · Output writers
# ==============================================================================


def write_reference_csv(pooled: pd.DataFrame, csv_path: Path = REFERENCE_CSV):
    """Write the pooled counts and proportions — the reference of the method."""
    profiles = reference_profiles(pooled)
    out = pooled.copy()
    out.columns = [f"count_{c}" for c in pooled.columns]
    out.insert(0, "n_total", pooled.sum(axis=1))
    for category in CATEGORIES:
        out[f"pct_{category}"] = (profiles[category] * 100).round(4)
    out.index.name = "horizon"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, encoding="utf-8-sig")
    print(f"✓ Reference profiles CSV saved: {csv_path}  ({len(out)} horizons)")


def write_examples_csv(
    pooled: pd.DataFrame, profiles: pd.DataFrame, csv_path: Path = EXAMPLES_CSV
):
    """Write every worked example at every sample size, under both methods."""
    rows = []
    for key, example in EXAMPLES.items():
        for n in EXAMPLE_SIZES:
            counts = scale_to_counts(example["shares"], n)
            both = assign(counts, pooled, profiles)
            for method, posterior in both.items():
                row = {
                    "example": key,
                    "shares": example["label"],
                    "n": n,
                    "method": method,
                    "counts": " / ".join(
                        str(counts[c]) for c in CATEGORIES if counts[c] > 0
                    ),
                }
                for horizon in sorted(posterior):
                    row[f"p_H{horizon}"] = round(posterior[horizon], 4)
                leader = max(posterior, key=posterior.get)
                fit = goodness_of_fit(counts, profiles, leader)
                row["leader"] = leader
                row["chi_square"] = round(fit["chi_square"], 3)
                row["df"] = fit["df"]
                row["p_fit"] = round(fit["p"], 5)
                row["fits"] = fit["fits"]
                rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ Worked examples CSV saved: {csv_path}  ({len(rows)} rows)")


def write_loo_csv(loo: pd.DataFrame, csv_path: Path = LOO_CSV):
    """Write the leave-one-out result for every seriated findspot."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    loo.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ Leave-one-out CSV saved: {csv_path}  ({len(loo)} findspots)")


# ==============================================================================
# SECTION 10 · Console report
# ==============================================================================


def report(pooled: pd.DataFrame, profiles: pd.DataFrame, loo: pd.DataFrame):
    """Print the reference table, the three examples and the validation."""
    print()
    print("Reference profiles (pooled sherd counts, per cent of horizon)")
    print("  horizon" + "".join(f"{c:>18}" for c in CATEGORIES) + f"{'n':>9}")
    for horizon in profiles.index:
        row = f"  H{int(horizon):<6}"
        for category in CATEGORIES:
            row += f"{profiles.at[horizon, category] * 100:17.2f}%"
        row += f"{int(pooled.loc[horizon].sum()):9d}"
        print(row)

    print()
    print("Worked examples (Dirichlet method, per cent)")
    print(
        f"  {'ex':<4}{'shares':<16}{'n':>5}"
        + "".join(f"{'H' + str(h):>9}" for h in HORIZON_NUMBERS)
    )
    for key, example in EXAMPLES.items():
        for n in EXAMPLE_SIZES:
            counts = scale_to_counts(example["shares"], n)
            posterior = posterior_dirichlet(counts, pooled)
            row = f"  {key:<4}{example['label']:<16}{n:>5}"
            for horizon in HORIZON_NUMBERS:
                value = posterior[horizon]
                row += f"{value:8.2f}" if value >= 0.01 else f"{'<0.01':>8}"
                row += " "
            print(row)

    print()
    print("Sherds needed to pass a confidence threshold (proportions held fixed)")
    print(f"  {'ex':<4}{'horizon':<10}" + "".join(f"{int(t * 100):>8}%" for t in THRESHOLDS))
    for key, example in EXAMPLES.items():
        needed = sherds_needed(example["shares"], pooled)
        leader = next((v["horizon"] for v in needed.values() if v), None)
        row = f"  {key:<4}H{leader if leader else '?':<9}"
        for threshold in THRESHOLDS:
            hit = needed[threshold]
            row += f"{hit['n'] if hit else '>' + str(MAX_SHERDS_SEARCHED):>9}"
        print(row)

    print()
    print("Goodness of fit of the leading horizon (n = 100)")
    print(
        f"  {'ex':<5}{'shares':<26}{'horizon':>8}{'chi2':>10}{'df':>4}"
        f"{'p':>11}   verdict"
    )
    fit_cases = [(k, EXAMPLES[k]["label"], EXAMPLES[k]["shares"]) for k in EXAMPLES]
    fit_cases.append(("mix", FIT_DEMO["label"], FIT_DEMO["shares"]))
    for key, label, shares in fit_cases:
        counts = scale_to_counts(shares, 100)
        posterior = posterior_dirichlet(counts, pooled)
        leader = max(posterior, key=posterior.get)
        fit = goodness_of_fit(counts, profiles, leader)
        verdict = "consistent" if fit["fits"] else "no horizon fits — mixed?"
        print(
            f"  {key:<5}{label:<26}{'H' + str(leader):>8}"
            f"{_format_statistic(fit['chi_square']):>10}{fit['df']:>4}"
            f"{_format_p(fit['p']):>11}   {verdict}"
        )
    print()
    print(
        "  The last row is a late context carrying early residual material. The "
        "horizon\n  probabilities give it H4 at 93.9 %, apparently decisively; "
        "only the fit test\n  reveals that no horizon accounts for it. Expected "
        "counts of the rare stages\n  are far below 5, so read these p-values as "
        "indicative rather than exact."
    )

    stats = summarise_leave_one_out(loo)
    print()
    print("Leave-one-out validation")
    print(
        f"  exact horizon recovered: {stats['exact']}/{stats['total']} "
        f"({stats['exact_pct']:.1f} %)"
    )
    print(
        f"  correct or neighbouring: {stats['adjacent']}/{stats['total']} "
        f"({stats['adjacent_pct']:.1f} %)"
    )
    small = loo[loo["n"] <= 100]
    large = loo[loo["n"] > 100]
    print(
        f"  failing the fit test:    {stats['misfits']}/{stats['total']} at "
        f"p < {FIT_ALPHA:g}  "
        f"({int((~small['fits']).sum())}/{len(small)} up to 100 sherds, "
        f"{int((~large['fits']).sum())}/{len(large)} above)"
    )
    print()
    print(
        "  That the larger find-spots are the ones failing is expected rather "
        "than alarming.\n  The fit test asks whether an assemblage is a random "
        "draw from the pooled horizon,\n  and no real site is: find-spots within "
        "one horizon vary among themselves more than\n  multinomial sampling "
        "allows. With enough sherds that surplus variation becomes\n  "
        "detectable, so the test works as a screening instrument for assemblages "
        "of moderate\n  size. For a large find-spot a high chi-square means "
        "\"more variable than the pooled\n  profile\", not \"assigned to the wrong "
        "horizon\" — the remedy would be an\n  over-dispersed fit test built on "
        "the same Dirichlet-multinomial as step 4."
    )
    misses = loo[~loo["hit"]]
    if not misses.empty:
        print()
        print("  findspots the model places elsewhere:")
        print(
            f"    {'findspot':<32}{'seriation':>10}{'model':>8}{'n':>6}"
            f"{'P(ser.)':>10}{'P(mod.)':>10}"
        )
        for _, row in misses.iterrows():
            print(
                f"    {row['findspot']:<32}{'H' + str(row['horizon_seriation']):>10}"
                f"{'H' + str(row['horizon_model']):>8}{row['n']:>6}"
                f"{row['p_seriation']:>9.1f}%{row['p_model']:>9.1f}%"
            )
        print()
        print(
            "  A miss is not automatically an error. Small assemblages are "
            "expected to wander,\n  and a near-tie is a genuine tie. A miss "
            "that is both clear and well supported is\n  worth a second look "
            "at the horizon assignment or at the context itself."
        )


# ==============================================================================
# SECTION 11 · Main entry point
# ==============================================================================


def parse_shares(text: str) -> dict:
    """Parse "Ib=13,Ic=27,II=60" into a share dictionary over CATEGORIES.

    Category names are matched loosely so the switch can be typed quickly:
    "II", "Service II" and "service ii" all reach the same category.
    """
    aliases = {c.casefold(): c for c in CATEGORIES}
    aliases.update({
        "ii": "Service II",
        "serviceii": "Service II",
        "srt": "Schrägrandteller",
        "schraegrandteller": "Schrägrandteller",
    })

    shares = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValueError(f"Expected name=value pairs, got: {part!r}")
        name, value = part.split("=", 1)
        key = "".join(name.split()).casefold()
        if key not in aliases:
            raise ValueError(
                f"Unknown category {name.strip()!r}. "
                f"Expected one of: {', '.join(CATEGORIES)}"
            )
        shares[aliases[key]] = float(value)

    total = sum(shares.values())
    if not math.isclose(total, 100.0, abs_tol=0.5):
        print(f"note: shares sum to {total:g} %, rescaling to 100 %")
        shares = {k: v / total * 100 for k, v in shares.items()}

    return shares


def main():
    parser = argparse.ArgumentParser(
        description="Assign an assemblage to a chronological horizon."
    )
    parser.add_argument(
        "--assign", metavar="SHARES",
        help='percentage shares of a new find, e.g. "Ib=13,Ic=27,II=60"'
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="number of sherds behind those shares (default: 100)"
    )
    parser.add_argument(
        "--steps", action="store_true",
        help="print the arithmetic of every step, not just the result"
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print only, write no CSVs"
    )
    args = parser.parse_args()

    print("Horizon assignment from service-type composition")
    print(f"  workbook: {XLSX_FILE}")

    findspots = load_findspot_counts()
    pooled = pool_by_horizon(findspots)
    profiles = reference_profiles(pooled)
    print(f"  reference: {int(pooled.to_numpy().sum())} sherds in "
          f"{len(pooled)} horizons")

    # --- A single assemblage, supplied on the command line ---
    if args.assign:
        shares = parse_shares(args.assign)
        counts = scale_to_counts(shares, args.n)
        if args.steps:
            print_derivation(counts, pooled, profiles)
        else:
            posterior = posterior_dirichlet(counts, pooled)
            print()
            for horizon in sorted(posterior):
                value = posterior[horizon]
                shown = f"{value:6.2f} %" if value >= 0.01 else "< 0.01 %"
                print(f"  H{horizon}  {shown}")

            leader = max(posterior, key=posterior.get)
            fit = goodness_of_fit(counts, profiles, leader)
            print()
            print(
                f"  fit of H{leader}: chi-square = "
                f"{_format_statistic(fit['chi_square'])}, df = {fit['df']}, "
                f"{_format_p_clause(fit['p'])}"
                + ("" if fit["fits"] else "  → no horizon fits well")
            )
        return

    # --- The full report ---
    loo = leave_one_out(findspots, pooled)
    report(pooled, profiles, loo)

    if not args.no_write:
        print()
        write_reference_csv(pooled)
        write_examples_csv(pooled, profiles)
        write_loo_csv(loo)


if __name__ == "__main__":
    main()