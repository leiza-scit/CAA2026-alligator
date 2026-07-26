# Figure: each findspot's assemblage split into its service types, drawn on the
# time axis over the findspot's interval and grouped by horizon. The browser
# counterpart of output/events_timeline_by_service_en.svg, and drawn to match
# it: latest horizon at the top, dashed rules between horizons, horizon labels
# down the right, a leader line from the name to the bar, and the same palette.
#
# The bar *positions* carry the chronology, the *segments* carry the
# composition. Because a long interval then gets a wide bar, a second mode
# gives every findspot the same width, which is the only way to compare
# compositions of findspots whose intervals differ by a factor of five. The
# printed figure has to choose; this one does not.
#
# Percentages are not stored in the graph. The query computes them from
# aeont:sherdCount, so this figure and the printed one cannot drift apart:
# there is one copy of the counts and no copy of the shares.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

sites, order = {}, []
for r in rows:
    name = r["findspot"]
    if name not in sites:
        sites[name] = {"label": name,
                       "begin": int(r["begin"]), "end": int(r["end"]),
                       "horizon": r.get("horizon") or "\u2013",
                       "total": int(r["total"]), "parts": []}
        order.append(name)
    sites[name]["parts"].append({
        "slug": r["slug"],
        "type": r["type"],
        "sherds": int(r["sherds"]),
        "share": 100 * int(r["sherds"]) / int(r["total"]),
    })

# Colours are keyed by the concept's slug, not by its label: the label is
# display text and changes with the language, the slug is the identifier. An
# unmatched key used to leave every bar grey and the legend empty, which reads
# as a design choice rather than a fault, so it is an error now.
missing = ({p["slug"] for s in sites.values() for p in s["parts"]}
           - set(SERVICE_COLOUR))
assert not missing, (
    f"no colour for {sorted(missing)} \u2014 SERVICE_COLOUR in "
    f"py/viz/_prelude.py is keyed by concept slug")

label_of = {p["slug"]: p["type"] for s in sites.values() for p in s["parts"]}
present = [slug for slug in SERVICE_COLOUR if slug in label_of]

legend = "".join(
    f'<span><b style="background:{SERVICE_COLOUR[slug]}"></b>'
    f'{escape(label_of[slug])}</span>' for slug in present)

payload = json.dumps({
    "sites": [sites[n] for n in order],
    "colour": {slug: SERVICE_COLOUR[slug] for slug in present},
    "min": min(s["begin"] for s in sites.values()) - 2,
    "max": max(s["end"] for s in sites.values()) + 1,
})

style = """
  body{margin:0;font-family:sans-serif;padding:6px 4px 4px}
  #ctrl{display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:.2rem 0}
  #ctrl label{font-size:12px;color:#666}
  #ctrl select{font-size:12px;padding:2px 6px;border-radius:4px;
    border:1px solid #ccc;background:#fff;color:#333}
  #leg{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;color:#555;
    padding:.2rem 0 .4rem}
  #leg span{display:flex;align-items:center;gap:4px}
  #leg b{width:10px;height:10px;border-radius:2px;display:inline-block}
  #chart{overflow-x:auto}
"""

script = """
(function () {
  var C = JSON.parse(document.getElementById("payload").textContent);
  var me = document.getElementById("mode");
  var ce = document.getElementById("chart");

  function yr(v) { return v < 0 ? (-v) + " BC" : "AD " + v; }

  function draw() {
    var onAxis = me.value === "axis";

    // Latest horizon at the top, and within a horizon the latest start first,
    // which is how the printed figure orders its rows.
    var d = C.sites.slice().sort(function (a, b) {
      if (a.horizon !== b.horizon) return a.horizon < b.horizon ? 1 : -1;
      return b.begin - a.begin || a.label.localeCompare(b.label);
    });

    var LW = 232, RH = 15, RG = 7, MT = 14, MB = 40, CW = 640, RW = 74;
    var W = LW + CW + RW, H = MT + d.length * (RH + RG) + MB;
    function px(y) { return LW + (y - C.min) / (C.max - C.min) * CW; }

    var o = '<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '"'
          + ' height="' + H + '" style="font-family:sans-serif">';

    if (onAxis) {
      var step = 5, t;
      for (t = Math.ceil(C.min / step) * step; t <= C.max; t += step) {
        var p = px(t);
        o += '<line x1="' + p + '" y1="' + MT + '" x2="' + p + '"'
           + ' y2="' + (H - MB) + '" stroke="#f0f0f0" stroke-width="1"/>';
        o += '<text x="' + p + '" y="' + (H - MB + 14) + '" font-size="9.5"'
           + ' fill="#777" text-anchor="end" transform="rotate(-35 ' + p + ','
           + (H - MB + 14) + ')">' + yr(t) + "</text>";
      }
    }

    var prev = null, blockTop = MT;
    d.forEach(function (x, i) {
      var y = MT + i * (RH + RG);

      if (prev !== null && x.horizon !== prev) {
        var ry = y - RG / 2;
        o += '<line x1="4" y1="' + ry + '" x2="' + (LW + CW) + '"'
           + ' y2="' + ry + '" stroke="#bbb" stroke-width="1"'
           + ' stroke-dasharray="5 4"/>';
        o += horizonLabel(prev, blockTop, ry);
        blockTop = ry;
      }
      prev = x.horizon;

      var x1 = onAxis ? px(x.begin) : LW;
      var w = onAxis ? Math.max(px(x.end) - x1, 2.5) : CW;

      o += '<text x="' + (LW - 10) + '" y="' + (y + RH * 0.8) + '"'
         + ' text-anchor="end" font-size="10.5" font-weight="600"'
         + ' fill="#333">' + x.label + "</text>";
      if (onAxis && x1 > LW + 2) {
        // Faint enough not to compete with the bars, dark enough to survive
        // being looked at on a screen - #d5d5d5 at 0.8 was invisible.
        o += '<line x1="' + (LW - 5) + '" y1="' + (y + RH / 2) + '"'
           + ' x2="' + (x1 - 2) + '" y2="' + (y + RH / 2) + '"'
           + ' stroke="#9aa0a6" stroke-width="1" stroke-dasharray="1 3"/>';
      }

      var cursor = x1;
      x.parts.forEach(function (p) {
        var seg = p.share / 100 * w;
        o += '<rect x="' + cursor + '" y="' + y + '"'
           + ' width="' + Math.max(seg, 0.35) + '" height="' + RH + '"'
           + ' fill="' + (C.colour[p.slug] || "#ccc") + '">'
           + "<title>" + x.label + " (" + yr(x.begin) + "\\u2013" + yr(x.end)
           + ") \\u2014 " + p.type + ": " + p.sherds + " sherds, "
           + p.share.toFixed(1) + "%</title></rect>";
        cursor += seg;
      });

      if (!onAxis) {
        o += '<text x="' + (LW + CW + 8) + '" y="' + (y + RH * 0.8) + '"'
           + ' font-size="9.5" fill="#999">' + x.total + "</text>";
      }
    });

    o += horizonLabel(prev, blockTop, MT + d.length * (RH + RG) - RG / 2);
    o += "</svg>";
    ce.innerHTML = o;

    function horizonLabel(h, top, bottom) {
      var mid = (top + bottom) / 2, x = LW + CW + (onAxis ? 22 : 40);
      var s = '<line x1="' + (x - 8) + '" y1="' + (top + 3) + '" x2="'
            + (x - 8) + '" y2="' + (bottom - 3) + '" stroke="#999"'
            + ' stroke-width="1"/>';
      s += '<text x="' + x + '" y="' + mid + '" font-size="11"'
         + ' font-weight="600" fill="#444" text-anchor="middle"'
         + ' transform="rotate(-90 ' + x + ',' + mid + ')">Horizon ' + h
         + "</text>";
      return s;
    }
  }

  me.addEventListener("change", draw);
  draw();
})();
"""

composition = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{style}</style></head>
<body>
<div id="ctrl">
  <label for="mode">Bars</label>
  <select id="mode">
    <option value="axis">On the time axis</option>
    <option value="equal">Equal width (composition only)</option>
  </select>
  <span style="font-size:11px;color:#888">
    {len(sites)} findspots &middot; segment = that type's share of the
    findspot's sherds &middot; hover for counts
  </span>
</div>
<div id="leg">{legend}</div>
<div id="chart"></div>
<script id="payload" type="application/json">{payload}</script>
<script>{script}</script>
</body></html>"""

Frame(composition, height=14 + len(sites) * 22 + 40 + 120)
