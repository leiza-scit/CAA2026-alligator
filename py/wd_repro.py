#!/usr/bin/env python3
"""Make matplotlib output byte-reproducible. Import this; that is all.

Two things in a matplotlib figure change on every run even when the figure does
not: the creation timestamp written into the SVG metadata, and the ids of the
`<path>` elements defined for ticks and markers, which are hashes salted with a
random value. Between them they rewrite dozens of lines per file, so every
rebuild shows a diff on every figure.

That matters beyond tidiness. A file that always shows up as modified trains
everyone to skip its diff, and once nobody reads the diff, a real change to a
published figure passes unnoticed. Reproducible output means a diff appears
exactly when something actually changed - the only state in which a diff is
worth reading.

Usage - one line, anywhere among the imports:

    import matplotlib.pyplot as plt
    import wd_repro            # noqa: F401  (imported for its effect)

No call, no ordering rule, and no change at the savefig call sites.

Why it is safe to do this on import
-----------------------------------
The obvious implementation - setting ``plt.rcParams["svg.hashsalt"]`` inside
each script - has a trap: ``plt.rcdefaults()`` resets it along with everything
else, and this family's convention is that every script calls ``rcdefaults()``
before applying its own styling. The salt would be silently discarded and the
ids random again, with nothing to show for it.

Changing ``matplotlib.rcParamsDefault`` instead means ``rcdefaults()`` restores
*this* value rather than a random one, so the setting survives the reset and the
order of the two no longer matters. Verified both ways round.

The date comes from the SOURCE_DATE_EPOCH environment variable, which matplotlib
reads when it writes a file rather than when it is imported - which is why this
works from a module imported at any point, and why no savefig call has to change.

This module deliberately does NOT import matplotlib.pyplot or select a backend;
it touches one dictionary and one environment variable. Importing it from a
script that never draws anything costs nothing.
"""

from __future__ import annotations

import os

# Any fixed string works; the salt has to be stable, not secret. One value across
# the family means the same figure built in two repositories gets the same ids.
HASHSALT = "wdttest"

# The timestamp stamped into SVG metadata. 0 (1970-01-01) is the convention from
# the reproducible-builds project: an obviously synthetic date, so nobody mistakes
# it for the date the figure was really made. Set it to a release date instead if
# you would rather the file carry a meaningful one - any fixed value works.
SOURCE_DATE_EPOCH = "0"


def deterministic(hashsalt: str = HASHSALT,
                  source_date_epoch: str = SOURCE_DATE_EPOCH) -> None:
    """Pin the two sources of run-to-run noise in matplotlib output.

    Called once on import; exposed so a script can be explicit about it, or pass
    a different salt or date. An existing SOURCE_DATE_EPOCH in the environment is
    respected rather than overwritten, so a CI system that already sets one stays
    in charge.
    """
    import matplotlib

    os.environ.setdefault("SOURCE_DATE_EPOCH", source_date_epoch)
    # The default, not just the current value: plt.rcdefaults() restores the
    # defaults, so setting it here is what makes the salt survive that call.
    matplotlib.rcParamsDefault["svg.hashsalt"] = hashsalt
    matplotlib.rcParams["svg.hashsalt"] = hashsalt


deterministic()


# ---------------------------------------------------------------------------
# Spreadsheets (openpyxl)
# ---------------------------------------------------------------------------
# An .xlsx is a ZIP archive, and it carries the clock in two places: every entry
# in the archive has a modification time, and docProps/core.xml records
# dcterms:created and dcterms:modified. Both change on every run, so a workbook
# rebuilt from unchanged data still shows up as modified.
#
# Unlike the matplotlib settings above this cannot be fixed by an import: there
# is no default to change, the timestamps are written by openpyxl and by Python's
# zipfile as the file is saved. So the save itself has to go through here.

# The fixed timestamp written into the workbook. Chosen to match the intent of
# SOURCE_DATE_EPOCH above: obviously synthetic, so nobody reads it as the moment
# the file was really produced.
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)   # ZIP's own epoch; earlier is unstorable


def _normalise_archive(path, timestamp=FIXED_TIMESTAMP):
    """Rewrite a ZIP archive with a fixed modification time on every entry.

    Order, names and compression are preserved exactly; only the clock changes.
    ``docProps/core.xml`` is patched as well, because openpyxl overwrites
    dcterms:modified with the current time as it saves - whatever the workbook's
    properties said a moment earlier. Done as a rewrite rather than by patching
    openpyxl, so it keeps working across library versions.
    """
    import re
    import shutil
    import zipfile
    from pathlib import Path

    stamp = ("%04d-%02d-%02dT%02d:%02d:%02dZ" % timestamp).encode("utf-8")

    def pin_dates(name, payload):
        if name != "docProps/core.xml":
            return payload
        for field in (b"created", b"modified"):
            payload = re.sub(
                rb"(<dcterms:" + field + rb"[^>]*>)[^<]*(</dcterms:" + field + rb">)",
                rb"\g<1>" + stamp + rb"\g<2>", payload)
        return payload

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".repro")
    with zipfile.ZipFile(path) as source:
        entries = [(info, pin_dates(info.filename, source.read(info.filename)))
                   for info in source.infolist()]
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as out:
        for info, payload in entries:
            fixed = zipfile.ZipInfo(info.filename, date_time=timestamp)
            fixed.compress_type = info.compress_type
            fixed.external_attr = info.external_attr
            out.writestr(fixed, payload)
    shutil.move(str(temporary), str(path))


def save_workbook(workbook, path, timestamp=FIXED_TIMESTAMP):
    """Save an openpyxl workbook so that identical data gives an identical file.

    Use in place of ``workbook.save(path)``. It pins the document properties and
    then rewrites the archive with fixed entry timestamps.
    """
    import datetime

    fixed = datetime.datetime(*timestamp)
    workbook.properties.created = fixed
    workbook.properties.modified = fixed
    workbook.save(path)
    _normalise_archive(path, timestamp)
    return path
