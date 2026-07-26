# Figure: Allen's interval relation between every ordered pair of horizons, as a
# matrix. The browser counterpart of output/horizon_allen_matrix_en.svg, and
# drawn to match it: the same relation palette, the same three families in the
# legend, the diagonal greyed out, axis labels carrying each horizon's interval.
#
# The palette is not re-invented here — ALLEN_COLOUR in py/viz/_prelude.py is
# copied from alligator_to_clean_rdf.py, so a relation is the same colour in
# this matrix as in the printed one.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

# The query names the relation as OWL-Time does, in CamelCase; the palette and
# the printed labels use Allen's own spelling.
SHORT = {
    "Before": ("before", "before"), "After": ("after", "after"),
    "Meets": ("meets", "meets"), "MetBy": ("metBy", "met-by"),
    "Overlaps": ("overlaps", "overlaps"),
    "OverlappedBy": ("overlappedBy", "ovlp-by"),
    "Contains": ("contains", "contains"), "During": ("during", "during"),
    "Starts": ("starts", "starts"), "StartedBy": ("startedBy", "started-by"),
    "Finishes": ("finishes", "finishes"),
    "FinishedBy": ("finishedBy", "finished-by"),
    "Equals": ("equals", "equals"),
}

FAMILIES = [
    ("Sequential (before / after / meets / met-by)", "#4a90d9",
     {"before", "after", "meets", "metBy"}),
    ("Overlapping (overlaps / overlapped-by)", "#f0a500",
     {"overlaps", "overlappedBy"}),
    ("Containing (contains / during / starts / finishes \u2026)", "#d94a4a",
     {"contains", "during", "starts", "startedBy", "finishes", "finishedBy"}),
    ("Equal", "#4caf50", {"equals"}),
]


def year(value):                                   # local, tolerates gYear text
    v = int(str(value).lstrip("+"))
    return f"{-v}BC" if v < 0 else f"AD{v}"


spans, cells = {}, {}
for r in rows:
    a, b = r["horizonA"], r["horizonB"]
    key, label = SHORT.get(r["relation"], (r["relation"], r["relation"]))
    cells[(a, b)] = {"key": key, "label": label,
                     "colour": ALLEN_COLOUR.get(key, "#888888")}
    if "beginA" in r and r["beginA"] is not None:
        spans[a] = f'{year(r["beginA"])}\u2013{year(r["endA"])}'

axis = sorted({h for pair in cells for h in pair})
head = "".join(
    f'<th><span>H{escape(h)}</span>'
    f'<em>{escape(spans.get(h, ""))}</em></th>' for h in axis)

body = ""
for a in axis:
    body += (f'<tr><th class="side"><span>H{escape(a)}</span>'
             f'<em>{escape(spans.get(a, ""))}</em></th>')
    for b in axis:
        if a == b:
            body += '<td class="same" title="the same horizon">&mdash;</td>'
            continue
        cell = cells.get((a, b))
        if cell is None:
            body += '<td class="same">&nbsp;</td>'
            continue
        tip = (f'H{a} ({spans.get(a, "?")}) {cell["label"]} '
               f'H{b} ({spans.get(b, "?")})')
        body += (f'<td style="background:{cell["colour"]};'
                 f'color:{ink_on(cell["colour"])}" title="{escape(tip)}">'
                 f'{escape(cell["label"])}</td>')
    body += "</tr>"

legend = "".join(
    f'<span><b style="background:{colour}"></b>{escape(name)}</span>'
    for name, colour, _members in FAMILIES
    if any(c["key"] in _members for c in cells.values())
) + '<span><b style="background:#e0e0e0"></b>Same horizon</span>'

style = """
  body{margin:0;font-family:sans-serif;padding:6px 4px 4px;color:#333}
  .axis{font-size:11px;color:#888;text-align:center;margin:0 0 6px}
  table{border-collapse:separate;border-spacing:3px;margin:0 auto}
  th{font-weight:600;font-size:11.5px;color:#444;padding:2px 4px}
  th span{display:block}
  th em{display:block;font-style:normal;font-size:9.5px;color:#999;
    font-weight:400}
  th.side{text-align:right}
  td{width:104px;height:52px;text-align:center;border-radius:4px;
    font-size:11.5px;font-weight:600;cursor:help}
  td.same{background:#e0e0e0;color:#aaa;font-weight:400;cursor:default}
  #leg{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;
    font-size:11px;color:#666;padding:.7rem 0 0}
  #leg span{display:flex;align-items:center;gap:5px}
  #leg b{width:11px;height:11px;border-radius:2px;display:inline-block}
"""

matrix = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{style}</style></head>
<body>
<p class="axis">row = horizon A &nbsp;&middot;&nbsp; column = horizon B
  &nbsp;&middot;&nbsp; read as &ldquo;A <em>relation</em> B&rdquo;</p>
<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>
<div id="leg">{legend}</div>
</body></html>"""

Frame(matrix, height=len(axis) * 58 + 150)
