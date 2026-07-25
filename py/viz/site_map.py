# Figure: the findspots on a slippy map, coloured and grouped by chronological
# horizon. There is no counterpart in notebook/clades_variana_temporal.ipynb —
# this is the spatial reading of the same graph, added because every findspot
# carries a geosparql:asWKT point and the horizons are as much a distribution
# as a sequence.
#
# The tiles come from openstreetmap.org, so this is the one figure in the
# notebook that makes a network request beyond the graph itself. Everything
# else, including which point sits where, is computed from the Turtle file.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

# One entry per findspot. The OPTIONAL blocks in the query would multiply rows
# if a findspot ever gained a second authority link, so collapse on the label.
sites = {}
for r in rows:
    sites[r["findspot"]] = {
        "label": r["findspot"],
        "horizon": r["horizon"],
        "start": int(r["begin"]),
        "end": int(r["end"]),
        "lat": float(r["latitude"]),
        "lon": float(r["longitude"]),
        "span": f'{year(r["begin"])}\u2013{year(r["end"])}',
        "wikidata": r.get("wikidata"),
        "pleiades": r.get("pleiades"),
    }

horizons = sorted({s["horizon"] for s in sites.values()})
by_horizon = {h: sum(1 for s in sites.values() if s["horizon"] == h)
              for h in horizons}

payload = json.dumps({
    "sites": sorted(sites.values(), key=lambda s: (s["horizon"], s["label"])),
    "colour": {h: HORIZON_COLOUR.get(h, "#666666") for h in horizons},
})

legend = "".join(
    f'<span><b style="background:{HORIZON_COLOUR.get(h, "#666")}"></b>'
    f'Horizon {escape(h)} ({by_horizon[h]})</span>' for h in horizons)

# Leaflet sizes the map from its container, and a container given a percentage
# height inside an unsized body collapses to nothing. So the frame height is
# fixed here and the map gets what is left after the legend.
FRAME_HEIGHT = 580
MAP_HEIGHT = FRAME_HEIGHT - 46

style = f"""
  body{{margin:0;font-family:sans-serif;padding:6px 4px 4px}}
  #leg{{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#555;
    padding:.2rem 0 .4rem}}
  #leg span{{display:flex;align-items:center;gap:4px}}
  #leg b{{width:10px;height:10px;border-radius:50%;display:inline-block}}
  #map{{width:100%;height:{MAP_HEIGHT}px;border:1px solid #ddd;
    border-radius:4px}}
  .leaflet-popup-content{{font-size:12px;line-height:1.45;margin:8px 12px}}
  .leaflet-popup-content b{{font-size:13px}}
  .leaflet-popup-content .meta{{color:#666}}
  .leaflet-popup-content a{{color:#185fa5}}
"""

script = """
(function () {
  var C = JSON.parse(document.getElementById("payload").textContent);
  var map = L.map("map", {scrollWheelZoom: false});

  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 12, minZoom: 3,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright"'
               + ' target="_blank" rel="noopener">OpenStreetMap</a> contributors'
  }).addTo(map);

  function link(url, text) {
    if (!url) return "";
    return ' <a href="' + url + '" target="_blank" rel="noopener">'
         + text + "</a>";
  }

  var layers = {}, bounds = [];
  C.sites.forEach(function (s) {
    var colour = C.colour[s.horizon] || "#666666";
    var marker = L.circleMarker([s.lat, s.lon], {
      radius: 6, color: "#ffffff", weight: 1.5,
      fillColor: colour, fillOpacity: 0.9
    });
    var authorities = link(s.wikidata, "Wikidata") + link(s.pleiades, "Pleiades");
    marker.bindPopup(
      "<b>" + s.label + "</b><br>"
      + '<span class="meta">' + s.span + " \\u00b7 horizon " + s.horizon
      + "</span>" + (authorities ? "<br>" + authorities : ""));
    marker.bindTooltip(s.label, {direction: "top", offset: [0, -6]});

    var key = "Horizon " + s.horizon;
    if (!layers[key]) layers[key] = L.layerGroup().addTo(map);
    marker.addTo(layers[key]);
    bounds.push([s.lat, s.lon]);
  });

  L.control.layers(null, layers, {collapsed: false}).addTo(map);
  map.fitBounds(bounds, {padding: [24, 24]});
})();
"""

site_map = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js">
</script>
<style>{style}</style>
</head>
<body>
<div id="leg">{legend}</div>
<div id="map"></div>
<script id="payload" type="application/json">{payload}</script>
<script>{script}</script>
</body></html>"""

Frame(site_map, height=FRAME_HEIGHT)
