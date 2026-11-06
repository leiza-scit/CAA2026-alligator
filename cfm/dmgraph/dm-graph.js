/* dm-graph.js
 *
 * Draws the Mermaid source that dm_graph.cfm embeds and offers the Turtle and
 * the SVG for download. Needs vendor/mermaid.min.js loaded first; nothing else.
 *
 * The colour scheme is defined here and only here. The server sends structure
 * (nodes, edges, classes ent / term / best); this file decides what they look like.
 */
(function () {
  'use strict';

  var CLASS_DEFS = [
    // Subject / object nodes: pastel grey
    'classDef ent fill:#e2e8f0,stroke:#000000,color:#000000;',
    // Terms (SKOS concepts): dark purple
    'classDef term fill:#4c1d95,stroke:#000000,color:#ffffff;',
    // The most probable horizon: heavier outline, colours stay as they are
    'classDef best stroke-width:4px;'
  ];

  function byId(id) { return document.getElementById(id); }

  function save(name, type, text) {
    var url = URL.createObjectURL(new Blob([text], { type: type }));
    var link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function init() {
    var canvas = byId('dm-graph-canvas');
    var turtleNode = byId('dm-turtle');
    var mermaidNode = byId('dm-mermaid');
    if (!canvas || !turtleNode || !mermaidNode) { return; }

    var turtle = turtleNode.textContent.replace(/^\s+|\s+$/g, '') + '\n';
    var source = mermaidNode.textContent.replace(/^\s+|\s+$/g, '');

    byId('dm-turtle-view').textContent = turtle;

    byId('dm-download-ttl').addEventListener('click', function () {
      save('horizon-assessment.ttl', 'text/turtle;charset=utf-8', turtle);
    });
    byId('dm-download-svg').addEventListener('click', function () {
      var svg = canvas.querySelector('svg');
      if (!svg) { return; }
      var text = new XMLSerializer().serializeToString(svg);
      save('horizon-assessment.svg', 'image/svg+xml;charset=utf-8',
           '<?xml version="1.0" encoding="UTF-8"?>\n' + text);
    });

    function fail(reason) {
      canvas.textContent = '';
      var note = document.createElement('p');
      note.className = 'dm-graph__error';
      note.textContent = 'The graph could not be drawn (' + reason +
        '). The triples are still available below and as a download.';
      canvas.appendChild(note);
      byId('dm-download-svg').disabled = true;
      byId('dm-layout').disabled = true;
      var details = canvas.parentNode.querySelector('details');
      if (details) { details.open = true; }
    }

    if (!window.mermaid || typeof window.mermaid.render !== 'function') {
      fail('vendor/mermaid.min.js was not loaded');
      return;
    }

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      theme: 'base',
      themeVariables: { fontFamily: 'Arial, sans-serif', fontSize: '12px' },
      // Plain SVG text rather than foreignObject, so the downloaded SVG opens
      // correctly in Inkscape, Illustrator and the like. Mermaid 11 reads the
      // switch at the top level; the flowchart entry is kept for older versions.
      htmlLabels: false,
      flowchart: { htmlLabels: false, curve: 'basis', nodeSpacing: 18, rankSpacing: 36, wrappingWidth: 220 }
    });

    // The server sends one direction (the first line of the source); the choice
    // in the menu replaces it and the diagram is drawn again.
    var select = byId('dm-layout');
    var drawn = 0;

    function draw(direction) {
      var text = source.replace(/^flowchart\s+\w+;/, 'flowchart ' + direction + ';');
      drawn += 1;
      window.mermaid.render('dm-graph-svg-' + drawn, text + '\n' + CLASS_DEFS.join('\n'))
        .then(function (out) { canvas.innerHTML = out.svg; })
        .catch(function (err) { fail(err && err.message ? err.message : 'render error'); });
    }

    var start = 'TB';
    try {
      var saved = window.localStorage.getItem('dmgraph.layout');
      if (saved && /^(TB|LR|BT|RL)$/.test(saved)) { start = saved; }
    } catch (e) { /* storage blocked: the default is fine */ }
    select.value = start;

    select.addEventListener('change', function () {
      try { window.localStorage.setItem('dmgraph.layout', select.value); } catch (e) { /* ignore */ }
      draw(select.value);
    });

    draw(start);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}());
