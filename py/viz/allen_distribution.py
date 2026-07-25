# Figure: how the 44 findspots distribute across Allen's thirteen relations to
# the Clades Variana. Ported from section 10 of notebook/clades_variana_temporal
# .ipynb, which built the same chart from a relation computed in pandas; here
# the relation arrives from the query above, computed in SPARQL.
#
# Runs in a Pyodide cell with `rows` (the query result) and the helpers from
# py/viz/_prelude.py in scope. It must end in a Frame.

counts = {rel: 0 for rel in ALLEN_ORDER}
for r in rows:
    counts[r["relation"]] = counts.get(r["relation"], 0) + 1

labels = ALLEN_ORDER
values = [counts[rel] for rel in labels]
colours = [ALLEN_COLOUR[rel] if rel in CONTEMPORARY_OR_LATER else "#b4b2a9"
           for rel in labels]

occupied = sum(1 for v in values if v)
qualifying = sum(v for rel, v in zip(labels, values)
                 if rel in CONTEMPORARY_OR_LATER)

chart = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js">
</script>
</head>
<body style="margin:0;font-family:sans-serif;padding:8px 4px 4px">
<p style="font-size:12px;color:#555;margin:0 0 6px">
  {len(rows)} findspots over {occupied} of the thirteen relations.
  <span style="display:inline-block;width:10px;height:10px;
    background:{ALLEN_COLOUR['finishedBy']};border-radius:2px;
    vertical-align:middle"></span>
  contemporary with or later than <em>{EVENT_LABEL}</em>
  ({qualifying}) &mdash;
  <span style="display:inline-block;width:10px;height:10px;background:#b4b2a9;
    border-radius:2px;vertical-align:middle"></span>
  wholly before it ({len(rows) - qualifying}).
</p>
<div style="position:relative;width:100%;height:{len(labels) * 30 + 60}px">
  <canvas id="allen"></canvas>
</div>
<script>
new Chart(document.getElementById("allen"), {{
  type: "bar",
  data: {{
    labels: {json.dumps(labels)},
    datasets: [{{
      data: {json.dumps(values)},
      backgroundColor: {json.dumps(colours)},
      borderWidth: 0, borderRadius: 3
    }}]
  }},
  options: {{
    indexAxis: "y", responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{display: false}},
      tooltip: {{callbacks: {{label: function (c) {{
        return " " + c.parsed.x + " findspot" + (c.parsed.x === 1 ? "" : "s");
      }}}}}}
    }},
    scales: {{
      x: {{beginAtZero: true, ticks: {{stepSize: 1, font: {{size: 12}}}},
          title: {{display: true, text: "Number of findspots",
                   font: {{size: 12}}}}}},
      y: {{ticks: {{font: {{size: 12, family: "monospace"}}}}}}
    }}
  }}
}});
</script>
</body></html>"""

Frame(chart, height=len(labels) * 30 + 130)
