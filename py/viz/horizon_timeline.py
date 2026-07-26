# Figure: the five chronological horizons as intervals, coloured by how many
# findspots each holds. The browser counterpart of output/horizon_timeline_en.svg
# and drawn to match it: latest horizon at the top, bar labelled inside where it
# fits and outside where it does not, YlOrRd sampled the same way the printed
# version samples it, colourbar from 1 to the largest horizon.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

horizons = [{
    "horizon": r["horizon"],
    "label": r["label"],
    "begin": int(str(r["begin"]).lstrip("+")),
    "end": int(str(r["end"]).lstrip("+")),
    "findspots": int(r["findspots"]),
} for r in rows]

most = max(h["findspots"] for h in horizons)

# The printed figure samples YlOrRd over 0.3 to 1.0 rather than the whole
# range: the pale end of the map is nearly white and a bar in it disappears.
for h in horizons:
    h["colour"] = ramp(YLORRD, 0.3 + 0.7 * h["findspots"] / most)
    h["ink"] = ink_on(h["colour"])

span_min = min(h["begin"] for h in horizons)
span_max = max(h["end"] for h in horizons)

payload = json.dumps({
    "horizons": sorted(horizons, key=lambda h: h["horizon"]),
    "min": span_min - 3, "max": span_max + 5,
})

bar = colourbar(YLORRD[3:], 1, most, width=200, fmt="{:.0f}")

style = """
  body{margin:0;font-family:sans-serif;padding:6px 4px 4px}
  #chart{overflow-x:auto}
  #foot{display:flex;align-items:flex-end;gap:10px;font-size:11px;color:#777;
    padding:.5rem 0 0}
"""

script = """
(function () {
  var C = JSON.parse(document.getElementById("payload").textContent);
  var ce = document.getElementById("chart");

  function yr(v) { return v < 0 ? (-v) + " BC" : "AD " + v; }

  var RH = 46, RG = 14, MT = 16, MB = 34, LW = 8, CW = 660, RW = 120;
  var d = C.horizons.slice().reverse();          // latest at the top
  var W = LW + CW + RW, H = MT + d.length * (RH + RG) + MB;
  function px(y) { return LW + (y - C.min) / (C.max - C.min) * CW; }

  var o = '<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '"'
        + ' height="' + H + '" style="font-family:sans-serif">';
  o += '<rect x="' + LW + '" y="' + MT + '" width="' + CW + '"'
     + ' height="' + (H - MT - MB) + '" fill="#f7f7f7"/>';

  var step = 5, t;
  for (t = Math.ceil(C.min / step) * step; t <= C.max; t += step) {
    var p = px(t);
    o += '<line x1="' + p + '" y1="' + MT + '" x2="' + p + '"'
       + ' y2="' + (H - MB) + '" stroke="#e3e3e3" stroke-width="1"/>';
    o += '<text x="' + p + '" y="' + (H - MB + 16) + '" font-size="10"'
       + ' fill="#666" text-anchor="end" transform="rotate(-35 ' + p + ','
       + (H - MB + 16) + ')">' + yr(t) + "</text>";
  }

  d.forEach(function (h, i) {
    var y = MT + i * (RH + RG) + RG / 2;
    var x1 = px(h.begin), w = Math.max(px(h.end) - x1, 2);
    o += '<rect x="' + x1 + '" y="' + y + '" width="' + w + '"'
       + ' height="' + RH + '" fill="' + h.colour + '">'
       + "<title>" + h.label + " \\u00b7 " + h.findspots
       + " findspots</title></rect>";
    var inside = w > 210;
    o += '<text x="' + (inside ? x1 + w / 2 : x1 + w + 8) + '"'
       + ' y="' + (y + RH / 2 + 4) + '" font-size="12.5" font-weight="600"'
       + ' text-anchor="' + (inside ? "middle" : "start") + '"'
       + ' fill="' + (inside ? h.ink : "#333") + '">' + h.label + "</text>";
    if (inside) {
      o += '<text x="' + (x1 + w + 8) + '" y="' + (y + RH / 2 + 4) + '"'
         + ' font-size="11.5" fill="#555">' + h.findspots + " findspots</text>";
    }
  });

  o += "</svg>";
  ce.innerHTML = o;
})();
"""

timeline = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{style}</style></head>
<body>
<div id="chart"></div>
<div id="foot">
  <span>Number of findspots</span>{bar}
  <span>Bars span the envelope of the horizon's findspots, from the earliest
  start to the latest end &mdash; so two horizons can overlap.</span>
</div>
<script id="payload" type="application/json">{payload}</script>
<script>{script}</script>
</body></html>"""

Frame(timeline, height=len(horizons) * 60 + 130)
