# Figure: what each findspot's Arretine assemblage is made of, as a proportional
# bar split into service types, findspots ordered by the year the seriation gives
# them. The browser counterpart of output/events_timeline_by_service_en.svg —
# with one difference: that figure places each bar on the time axis, spanning the
# findspot's interval, whereas this one gives every findspot a full-width bar so
# the composition can be compared across findspots of very different duration.
#
# The percentages are not stored in the graph. They are computed by the query
# from aeont:sherdCount, which is why the numbers here and in the printed figure
# cannot drift apart: there is one copy of the counts and no copy of the shares.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

# rows arrive one per findspot-and-type, ordered by year then typological rank.
sites, order = {}, []
for r in rows:
    name = r["findspot"]
    if name not in sites:
        sites[name] = {"label": name, "begin": int(r["begin"]),
                       "horizon": r.get("horizon"), "total": int(r["total"]),
                       "parts": []}
        order.append(name)
    sites[name]["parts"].append({
        "type": r["type"],
        "sherds": int(r["sherds"]),
        "share": 100 * int(r["sherds"]) / int(r["total"]),
    })

present = [t for t in SERVICE_COLOUR if any(
    p["type"] == t for s in sites.values() for p in s["parts"])]

legend = "".join(
    f'<span><b style="background:{SERVICE_COLOUR[t]}"></b>{escape(t)}</span>'
    for t in present)

payload = json.dumps({
    "sites": [sites[n] for n in order],
    "colour": {t: SERVICE_COLOUR[t] for t in present},
})

style = """
  body{margin:0;font-family:sans-serif;padding:6px 4px 4px}
  #ctrl{display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:.3rem 0}
  #ctrl label{font-size:12px;color:#666}
  #ctrl select{font-size:12px;padding:2px 6px;border-radius:4px;
    border:1px solid #ccc;background:#fff;color:#333}
  #leg{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;color:#555;
    padding:.2rem 0 .4rem}
  #leg span{display:flex;align-items:center;gap:4px}
  #leg b{width:10px;height:10px;border-radius:2px;display:inline-block}
"""

script = """
(function () {
  var C = JSON.parse(document.getElementById("payload").textContent);
  var se = document.getElementById("sort");
  var ce = document.getElementById("chart");

  function yr(v) { return v < 0 ? (-v) + " BC" : "AD " + v; }

  function draw() {
    var s = se.value;
    var d = C.sites.slice();
    d.sort(function (a, b) {
      if (s === "year")    return a.begin - b.begin || a.label.localeCompare(b.label);
      if (s === "sherds")  return b.total - a.total;
      if (s === "horizon") return (a.horizon || "") > (b.horizon || "") ? 1
                                : (a.horizon || "") < (b.horizon || "") ? -1
                                : a.begin - b.begin;
      return a.label.localeCompare(b.label);
    });

    var LW = 210, RH = 18, RG = 3, MT = 20, MB = 8, CW = 400, NW = 62;
    var W = LW + CW + NW + 12, H = MT + d.length * (RH + RG) + MB;

    var o = '<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '"'
          + ' height="' + H + '" style="font-family:sans-serif;overflow:visible">';

    [0, 25, 50, 75, 100].forEach(function (pc) {
      var x = LW + pc / 100 * CW;
      o += '<line x1="' + x + '" y1="' + (MT - 3) + '" x2="' + x + '"'
         + ' y2="' + (H - MB) + '" stroke="#e8e8e8" stroke-width=".5"/>';
      o += '<text x="' + x + '" y="' + (MT - 7) + '" text-anchor="middle"'
         + ' font-size="9" fill="#aaa">' + pc + "%</text>";
    });

    d.forEach(function (x, i) {
      var y = MT + i * (RH + RG), cursor = LW;
      o += '<text x="' + (LW - 6) + '" y="' + (y + RH * 0.72) + '"'
         + ' text-anchor="end" font-size="10.5" fill="#444">' + x.label
         + '<title>' + x.label + " \\u00b7 " + yr(x.begin)
         + (x.horizon ? " \\u00b7 horizon " + x.horizon : "")
         + " \\u00b7 " + x.total + " sherds</title></text>";

      x.parts.forEach(function (p) {
        var w = p.share / 100 * CW;
        o += '<rect x="' + cursor + '" y="' + (y + 2) + '"'
           + ' width="' + Math.max(w, 0.4) + '" height="' + (RH - 4) + '"'
           + ' fill="' + (C.colour[p.type] || "#ccc") + '">'
           + "<title>" + x.label + " \\u2014 " + p.type + ": " + p.sherds
           + " sherds, " + p.share.toFixed(1) + "%</title></rect>";
        cursor += w;
      });

      o += '<text x="' + (LW + CW + 6) + '" y="' + (y + RH * 0.72) + '"'
         + ' font-size="9.5" fill="#999">' + x.total + "</text>";
    });

    o += "</svg>";
    ce.innerHTML = o;
  }

  se.addEventListener("change", draw);
  draw();
})();
"""

composition = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{style}</style></head>
<body>
<div id="ctrl">
  <label for="sort">Sort by</label>
  <select id="sort">
    <option value="year">Start year</option>
    <option value="horizon">Horizon</option>
    <option value="sherds">Sherds (most first)</option>
    <option value="label">Findspot</option>
  </select>
  <span style="font-size:11px;color:#888">
    {len(sites)} findspots &middot; bar = 100% of that findspot's sherds
    &middot; figure at the right = sherds counted
  </span>
</div>
<div id="leg">{legend}</div>
<div id="chart"></div>
<script id="payload" type="application/json">{payload}</script>
<script>{script}</script>
</body></html>"""

Frame(composition, height=20 + len(sites) * 21 + 8 + 110)
