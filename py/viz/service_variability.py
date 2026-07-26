# Figure: the RGZM within-group variance and quality per horizon and service
# group. The browser counterpart of output/service_group_variability_en.svg and
# drawn to match it: two panels side by side, variance on RdYlGn_r from 0 to the
# largest value, quality on RdYlGn from 0 to 1, latest horizon at the top, a
# colourbar under each panel, and grey for the cells whose value follows from
# the rank assignment rather than from the sherds.
#
# One thing the printed figure cannot do is offered here instead of a second
# page: the rank reading can be switched. The stage reading is what the project
# reports — a cup and a plate of one stage share a rank, because a change of
# vessel form is not a chronological step. The sub-type reading counts every
# type as a step and is kept so the alternative can be re-checked; it orders the
# horizons the same way.
#
# The query returns N, Σ(rank·count) and Σ(rank²·count); the standard deviation
# and exp(−CV) are taken by rgzm() in py/viz/_prelude.py, because rdflib's
# SPARQL has neither SQRT nor EXP.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

READINGS = [
    ("stage", "ranks by stage only", "stageSteps", "sumStage", "sumStageSq"),
    ("column", "ranks by sub-type", "columnSteps", "sumColumn", "sumColumnSq"),
]

horizons, groups, data = [], [], {}
for r in rows:
    horizon, group, n = r["horizon"], r["group"], int(r["sherds"])
    if horizon not in horizons:
        horizons.append(horizon)
    if group not in groups:
        groups.append(group)
    for key, _note, steps_field, sum_field, sumsq_field in READINGS:
        s, q = rgzm(n, float(r[sum_field]), float(r[sumsq_field]))
        data[(key, horizon, group)] = {
            "n": n, "s": s, "q": q, "measured": int(r[steps_field]) > 1}

horizons.sort()

payload = json.dumps({
    "readings": {k: note for k, note, *_ in READINGS},
    "horizons": horizons,
    "groups": groups,
    # Keyed "reading|horizon|group", so switching the reading is a redraw
    # rather than a second query.
    "cells": {f"{k}|{h}|{g}": v for (k, h, g), v in data.items()},
    "vmax": {k: max((v["s"] for (rk, _h, _g), v in data.items()
                     if rk == k and v["measured"]), default=1.0)
             for k, *_ in READINGS},
    "rdylgn": RDYLGN,
    "grey": STRUCTURAL_GREY,
})

# The variance panel is scaled to its own largest value, so its bar is labelled
# with that; the quality panel always runs 0 to 1, as in the printed figure.
bars = {k: colourbar(RDYLGN[::-1], 0, v, width=250, fmt="{:.2f}")
        for k, v in json.loads(payload)["vmax"].items()}
bar_qual = colourbar(RDYLGN, 0, 1, width=250, fmt="{:.1f}")

style = """
  body{margin:0;font-family:sans-serif;padding:6px 4px 4px;color:#333}
  #ctrl{display:flex;gap:10px;align-items:center;font-size:12px;color:#666;
    padding:0 0 .6rem}
  #ctrl select{font-size:12px;padding:2px 6px;border-radius:4px;
    border:1px solid #ccc;background:#fff;color:#333}
  #ctrl em{font-style:normal;color:#999}
  .panels{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}
  .panel h3{margin:0 0 8px;font-size:12.5px;font-weight:700;text-align:center}
  table{border-collapse:separate;border-spacing:2px}
  th{font-size:11.5px;font-weight:700;padding:0 4px 4px;text-align:center}
  th.side{text-align:right;white-space:nowrap}
  td{width:104px;height:56px;text-align:center;line-height:1.25;cursor:help}
  .v{display:block;font-size:15px;font-weight:600}
  .n{display:block;font-size:9.5px;opacity:.65}
  .cbar{margin-top:8px;text-align:center}
  .cbar span{display:block;font-size:10px;color:#999;margin-bottom:2px}
  .foot{font-size:11px;color:#888;max-width:82ch;line-height:1.55;
    padding:.8rem 0 0}
  .foot b{display:inline-block;width:11px;height:11px;background:#eeeeee;
    border:1px solid #ddd;vertical-align:middle}
"""

script = """
(function () {
  var C = JSON.parse(document.getElementById("payload").textContent);
  var se = document.getElementById("reading");
  var ne = document.getElementById("note");

  // The group colours of the printed figure's header row.
  var HEAD = {"Oblique-rim plate": "#1f77b4", "Service I": "#d62728",
              "Service II": "#2ca02c"};

  function ramp(stops, t) {
    t = Math.min(Math.max(t, 0), 1);
    var span = t * (stops.length - 1);
    var i = Math.min(Math.floor(span), stops.length - 2), f = span - i;
    var out = "#", k, v;
    for (k = 1; k < 6; k += 2) {
      v = Math.round(parseInt(stops[i].substr(k, 2), 16)
        + (parseInt(stops[i + 1].substr(k, 2), 16)
           - parseInt(stops[i].substr(k, 2), 16)) * f);
      out += ("0" + v.toString(16)).slice(-2);
    }
    return out;
  }

  function ink(hex) {
    var r = parseInt(hex.substr(1, 2), 16), g = parseInt(hex.substr(3, 2), 16),
        b = parseInt(hex.substr(5, 2), 16);
    return (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? "#1a1a1a" : "#ffffff";
  }

  function panel(kind, reading) {
    var vmax = kind === "s" ? C.vmax[reading] : 1;
    var stops = kind === "s" ? C.rdylgn.slice().reverse() : C.rdylgn;
    var h = "<table><thead><tr><th></th>";
    C.groups.forEach(function (g) {
      h += '<th style="color:' + (HEAD[g] || "#444") + '">' + g + "</th>";
    });
    h += "</tr></thead><tbody>";
    C.horizons.slice().reverse().forEach(function (hz) {
      h += '<tr><th class="side">Horizon ' + hz + "</th>";
      C.groups.forEach(function (g) {
        var c = C.cells[reading + "|" + hz + "|" + g];
        if (!c) {
          h += '<td style="background:' + C.grey + ';color:#bbb">&mdash;</td>';
          return;
        }
        var v = kind === "s" ? c.s : c.q;
        var fill = c.measured ? ramp(stops, vmax ? v / vmax : 0) : C.grey;
        var tone = c.measured ? ink(fill) : "#888";
        var tip = g + ", horizon " + hz + ": " + c.n + " sherds, s = "
                + c.s.toFixed(3) + ", q = " + c.q.toFixed(3)
                + (c.measured ? "" : " \\u2014 one rank only, so this follows"
                   + " from the rank assignment rather than from the sherds");
        h += '<td style="background:' + fill + ';color:' + tone + '"'
           + ' title="' + tip + '"><span class="v">' + v.toFixed(2)
           + '</span><span class="n">n=' + c.n + "</span></td>";
      });
      h += "</tr>";
    });
    return h + "</tbody></table>";
  }

  function draw() {
    var reading = se.value;
    ne.textContent = C.readings[reading];
    document.getElementById("pv").innerHTML = panel("s", reading);
    document.getElementById("pq").innerHTML = panel("q", reading);
    Array.prototype.forEach.call(
      document.querySelectorAll("[data-bar]"), function (el) {
        el.style.display = el.getAttribute("data-bar") === reading ? "" : "none";
      });
  }

  se.addEventListener("change", draw);
  draw();
})();
"""

var_bars = "".join(f'<div data-bar="{k}">{bar}</div>' for k, bar in bars.items())

heatmap = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{style}</style></head>
<body>
<div id="ctrl">
  <label for="reading">Rank reading</label>
  <select id="reading">
    <option value="stage">Stage &mdash; reported</option>
    <option value="column">Sub-type &mdash; for comparison</option>
  </select>
  <em id="note"></em>
</div>
<div class="panels">
  <div class="panel">
    <h3>Within-group variance &nbsp;(STDDEV_SAMP of sub-type ranks)</h3>
    <div id="pv"></div>
    <div class="cbar"><span>variance</span>{var_bars}</div>
  </div>
  <div class="panel">
    <h3>Within-group quality &nbsp;(q = exp(&minus;CV))</h3>
    <div id="pq"></div>
    <div class="cbar"><span>quality</span>{bar_qual}</div>
  </div>
</div>
<p class="foot">
  Every sherd is one observation valued by the rank of its sub-type within its
  group. <b></b> grey: the group holds a single rank, so <em>s</em> = 0 and
  <em>q</em> = 1 follow from the rank assignment rather than from the material.
  Hover a cell for both figures and the sherd count.
</p>
<script id="payload" type="application/json">{payload}</script>
<script>{script}</script>
</body></html>"""

Frame(heatmap, height=len(horizons) * 62 + 250)
