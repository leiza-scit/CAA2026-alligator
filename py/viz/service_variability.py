# Figure: the RGZM within-group variance and quality per horizon and service
# group, under both rank readings side by side. The browser counterpart of
# output/service_group_variability_en.svg.
#
# The query returns N, Σ(rank·count) and Σ(rank²·count); the standard deviation
# and exp(−CV) are taken here, because rdflib's SPARQL has neither SQRT nor EXP.
# See rgzm() in py/viz/_prelude.py.
#
# Two readings, not one. Under the stage reading a cup and a plate of the same
# stage share a rank, which is what this project reports: a change of vessel
# form is not a chronological step. The column reading counts every sub-type as
# a step and is kept so the alternative can be re-checked. They rank the
# horizons the same way, which is the point of showing both.
#
# Cells whose group holds only one rank are drawn grey. Their s = 0 and q = 1
# follow from the rank assignment alone, whatever the sherds do, and a
# structurally determined constant should not read as a measurement.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

READINGS = [
    ("stage", "Stage reading", "stageSteps", "sumStage", "sumStageSq",
     "cup and plate of one stage share a rank \u2014 reported"),
    ("column", "Column reading", "columnSteps", "sumColumn", "sumColumnSq",
     "every sub-type its own step \u2014 kept for comparison"),
]

horizons, groups, cells = [], [], {}
for r in rows:
    horizon, group, n = r["horizon"], r["group"], int(r["sherds"])
    if horizon not in horizons:
        horizons.append(horizon)
    if group not in groups:
        groups.append(group)
    for key, _title, steps_field, sum_field, sumsq_field, _note in READINGS:
        s, q = rgzm(n, float(r[sum_field]), float(r[sumsq_field]))
        cells[(key, horizon, group)] = {
            "n": n, "s": s, "q": q,
            "measured": int(r[steps_field]) > 1,
        }

horizons.sort()
# Latest horizon at the top, as the printed figures do.
rows_order = list(reversed(horizons))

# Quality drives the colour: dark where the assemblage sits on one rank, light
# where it is spread. Only measured cells are shaded.
measured_q = [c["q"] for c in cells.values() if c["measured"]]
lo, hi = (min(measured_q), max(measured_q)) if measured_q else (0.0, 1.0)


def shade(q):
    """A green ramp over the measured range, dark = concentrated."""
    if hi == lo:
        t = 0.5
    else:
        t = (q - lo) / (hi - lo)
    top = (0, 68, 27)
    bottom = (229, 245, 224)
    rgb = [round(bottom[i] + (top[i] - bottom[i]) * t) for i in range(3)]
    return "#%02x%02x%02x" % tuple(rgb), ("#ffffff" if t > 0.45 else "#1a1a1a")


def block(key, title, note):
    head = "".join(
        f'<th title="{escape(group)}">{escape(group)}</th>' for group in groups)
    body = ""
    for horizon in rows_order:
        body += f'<tr><th class="h">H{escape(horizon)}</th>'
        for group in groups:
            cell = cells.get((key, horizon, group))
            if cell is None:
                body += '<td class="absent">&mdash;</td>'
                continue
            if cell["measured"]:
                fill, ink = shade(cell["q"])
                cls = ""
            else:
                fill, ink, cls = STRUCTURAL_GREY, "#666", " structural"
            tip = (f'{group}, horizon {horizon}: {cell["n"]} sherds, '
                   f's = {cell["s"]:.3f}, q = {cell["q"]:.3f}'
                   + ("" if cell["measured"]
                      else " \u2014 one rank only, so this follows from the "
                           "rank assignment rather than from the sherds"))
            body += (f'<td class="cell{cls}" style="background:{fill};'
                     f'color:{ink}" title="{escape(tip)}">'
                     f'<span class="q">{cell["q"]:.3f}</span>'
                     f'<span class="s">s {cell["s"]:.3f}</span>'
                     f'<span class="n">{cell["n"]} sherds</span></td>')
        body += "</tr>"
    return (f'<div class="block"><h3>{escape(title)}</h3>'
            f'<p class="note">{escape(note)}</p>'
            f'<table><thead><tr><th></th>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")

style = f"""
  body{{margin:0;font-family:sans-serif;padding:6px 4px 4px;color:#333}}
  .wrap{{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}}
  .block h3{{margin:0 0 2px;font-size:13px;font-weight:600}}
  .block .note{{margin:0 0 8px;font-size:11px;color:#888;max-width:34ch}}
  table{{border-collapse:separate;border-spacing:3px}}
  th{{font-size:11px;font-weight:600;color:#666;text-align:center;
    padding:0 4px;max-width:96px}}
  th.h{{text-align:right;color:#444}}
  td.cell{{width:96px;padding:5px 6px;border-radius:4px;text-align:center;
    line-height:1.25;cursor:help}}
  td.absent{{width:96px;text-align:center;color:#ccc;font-size:12px}}
  td.cell.structural{{border:1px dashed #b9b9b9}}
  .q{{display:block;font-size:14px;font-weight:600}}
  .s{{display:block;font-size:10px;opacity:.85}}
  .n{{display:block;font-size:9.5px;opacity:.7}}
  .key{{margin-top:10px;font-size:11px;color:#777;max-width:70ch;
    line-height:1.5}}
  .key b{{display:inline-block;width:10px;height:10px;border-radius:2px;
    background:{STRUCTURAL_GREY};border:1px dashed #b9b9b9;
    vertical-align:middle}}
"""

heatmap = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{style}</style></head>
<body>
<div class="wrap">
  {"".join(block(k, t, note) for k, t, _s, _a, _b, note in READINGS)}
</div>
<p class="key">
  Large figure is the quality <em>q</em> = exp(&minus;CV); below it the standard
  deviation <em>s</em> over the within-group ranks, and the number of sherds the
  two are computed from. Darker means the group's material sits on fewer ranks.
  <b></b> grey with a dashed border: the group holds a single rank, so
  <em>s</em> = 0 and <em>q</em> = 1 follow from the rank assignment rather than
  from the material. Hover any cell for the full figures.
</p>
</body></html>"""

Frame(heatmap, height=len(horizons) * 62 + 210)
