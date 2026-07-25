# Figure: a Gantt-style timeline of every findspot still in use at, or first
# used after, the Clades Variana, with AD 9 marked. Ported from section 11 of
# notebook/clades_variana_temporal.ipynb.
#
# One deliberate change from the notebook version. There the colour map and the
# filter listed 'after', 'contains' and 'meets', which are not the relations
# this corpus actually contains: every bar that came out 'finishedBy' or
# 'startedBy' fell through to the grey default and could not be filtered for.
# Legend and filter are built from the data instead, so they cannot drift from
# it again.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

# This figure and the one above share a query. The notebook does the same: one
# frame of all findspots with their relation, and the qualifying subset taken
# from it in Python rather than asked for a second time.
sites = [{"id": r["site"],
          "label": r["findspot"],
          "start": int(r["begin"]),
          "end": int(r["end"]),
          "rel": r["relation"]}
         for r in rows if r["relation"] in CONTEMPORARY_OR_LATER]

# Relations present, in Allen's order rather than in order of appearance.
present = [rel for rel in ALLEN_ORDER if any(s["rel"] == rel for s in sites)]

# Bar fill: the relation colour at about a tenth of its strength, so the
# outline stays the thing that carries the meaning.
fill = {rel: ALLEN_COLOUR[rel] + "1f" for rel in present}
stroke = {rel: ALLEN_COLOUR[rel] for rel in present}

span_min = min(s["start"] for s in sites)
span_max = max(s["end"] for s in sites)
pad = max(2, round((span_max - span_min) * 0.06))

legend = "".join(
    f'<span><b style="background:{ALLEN_COLOUR[rel]}"></b>{rel} '
    f'({sum(1 for s in sites if s["rel"] == rel)})</span>'
    for rel in present)
options = "".join(f'<option value="{rel}">{rel}</option>' for rel in present)

payload = json.dumps({"sites": sites, "fill": fill, "stroke": stroke,
                      "min": span_min - pad, "max": span_max + pad,
                      "event": EVENT_YEAR, "eventLabel": EVENT_LABEL,
                      "eventColour": EVENT_COLOUR})

style = """
  body{margin:0;font-family:sans-serif;padding:6px 4px 4px}
  #ctrl{display:flex;gap:12px;flex-wrap:wrap;align-items:center;
    padding:.3rem 0}
  #ctrl label{font-size:12px;color:#666}
  #ctrl select{font-size:12px;padding:2px 6px;border-radius:4px;
    border:1px solid #ccc;background:#fff;color:#333}
  #leg{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#555;
    padding:.2rem 0 .3rem}
  #leg span{display:flex;align-items:center;gap:4px}
  #leg b{width:10px;height:10px;border-radius:2px;display:inline-block}
"""

script = """
(function () {
  var C = JSON.parse(document.getElementById("payload").textContent);
  var se = document.getElementById("sort");
  var fe = document.getElementById("filter");
  var ce = document.getElementById("chart");

  function yr(v) { return v < 0 ? (-v) + " BC" : "AD " + v; }

  function ticks(lo, hi) {
    var step = (hi - lo) > 60 ? 10 : 5, out = [], t;
    for (t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push(t);
    if (out.indexOf(C.event) < 0) out.push(C.event);
    return out.sort(function (a, b) { return a - b; });
  }

  function draw() {
    var s = se.value, f = fe.value;
    var d = f === "all" ? C.sites.slice()
                        : C.sites.filter(function (x) { return x.rel === f; });
    d.sort(function (a, b) {
      if (s === "start") return a.start - b.start || a.label.localeCompare(b.label);
      if (s === "end")   return a.end   - b.end   || a.label.localeCompare(b.label);
      if (s === "rel")   return a.rel.localeCompare(b.rel) || a.start - b.start;
      return a.label.localeCompare(b.label);
    });

    var LW = 200, RH = 21, RG = 4, MT = 38, MB = 36, CW = 480;
    var W = LW + CW + 24, H = MT + d.length * (RH + RG) + MB;
    function px(y) { return LW + (y - C.min) / (C.max - C.min) * CW; }

    var o = '<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '"'
          + ' height="' + H + '" style="font-family:sans-serif;overflow:visible">';

    ticks(C.min, C.max).forEach(function (t) {
      var p = px(t), ev = (t === C.event);
      o += '<line x1="' + p + '" y1="' + (MT - 4) + '" x2="' + p + '"'
         + ' y2="' + (H - MB) + '" stroke="' + (ev ? C.eventColour : "#ddd") + '"'
         + ' stroke-width="' + (ev ? 1.5 : 0.5) + '"'
         + (ev ? ' stroke-dasharray="4 3"' : "") + "/>";
      o += '<text x="' + p + '" y="' + (MT - 8) + '" text-anchor="middle"'
         + ' font-size="10" fill="' + (ev ? C.eventColour : "#999") + '">'
         + yr(t) + "</text>";
    });

    d.forEach(function (x, i) {
      var y = MT + i * (RH + RG);
      var x1 = px(x.start), x2 = px(x.end), bw = Math.max(x2 - x1, 3);
      var sc = C.stroke[x.rel] || "#888", fc = C.fill[x.rel] || "#eee";
      o += '<text x="' + (LW - 6) + '" y="' + (y + RH * 0.72) + '"'
         + ' text-anchor="end" font-size="11" fill="#444">' + x.label + "</text>";
      o += '<rect x="' + x1 + '" y="' + (y + 2) + '" width="' + bw + '"'
         + ' height="' + (RH - 4) + '" rx="3" fill="' + fc + '"'
         + ' stroke="' + sc + '" stroke-width="1.2"><title>' + x.label
         + " \\u00b7 " + yr(x.start) + "\\u2013" + yr(x.end) + " \\u00b7 "
         + x.rel + "</title></rect>";
      var wide = bw > 70;
      o += '<text x="' + (wide ? x1 + bw / 2 : x2 + 4) + '"'
         + ' y="' + (y + RH * 0.72) + '"'
         + ' text-anchor="' + (wide ? "middle" : "start") + '" font-size="10"'
         + ' fill="' + (wide ? sc : "#999") + '">'
         + yr(x.start) + "\\u2013" + yr(x.end) + "</text>";
    });

    var ex = px(C.event);
    o += '<rect x="' + (ex - 44) + '" y="' + (H - MB + 4) + '" width="88"'
       + ' height="16" rx="3" fill="#faece7" stroke="' + C.eventColour + '"'
       + ' stroke-width=".8"/>';
    o += '<text x="' + ex + '" y="' + (H - MB + 15) + '" text-anchor="middle"'
       + ' font-size="10" fill="' + C.eventColour + '">' + C.eventLabel
       + "</text>";
    o += "</svg>";
    ce.innerHTML = o;
  }

  se.addEventListener("change", draw);
  fe.addEventListener("change", draw);
  draw();
})();
"""

timeline = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{style}</style></head>
<body>
<div id="ctrl">
  <label for="sort">Sort by</label>
  <select id="sort">
    <option value="start">Start year</option>
    <option value="end">End year</option>
    <option value="rel">Allen relation</option>
    <option value="label">Findspot</option>
  </select>
  <label for="filter">Relation</label>
  <select id="filter">
    <option value="all">all ({len(sites)})</option>
    {options}
  </select>
</div>
<div id="leg">{legend}
  <span><b style="background:{EVENT_COLOUR};opacity:.7"></b>
    {EVENT_LABEL} {year(EVENT_YEAR)}</span>
</div>
<div id="chart"></div>
<script id="payload" type="application/json">{payload}</script>
<script>{script}</script>
</body></html>"""

Frame(timeline, height=38 + len(sites) * 25 + 36 + 110)
