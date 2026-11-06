/* dm-app.js — the dating-machine page: form, bars, graph, downloads.
 *
 * Needs dm-model.js and vendor/mermaid.min.js loaded first. The colour scheme of
 * the graph is defined here and only here; dm-model.js sends structure
 * (classes ent / term / best).
 */
(function () {
  'use strict';

  var M = window.DMModel;
  var CLASS_DEFS = [
    'classDef ent fill:#e2e8f0,stroke:#000000,color:#000000;',   // subject / object nodes
    'classDef term fill:#4c1d95,stroke:#000000,color:#ffffff;',  // concepts of the typology
    'classDef best stroke-width:4px;'                            // most probable horizon
  ];
  var EXAMPLE = [0, 0, 13, 27, 60];
  var NS = 'http://www.w3.org/2000/svg';

  function $(id) { return document.getElementById(id); }
  function svgEl(name, attrs, text) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) { if (Object.prototype.hasOwnProperty.call(attrs, k)) { e.setAttribute(k, attrs[k]); } }
    if (text !== undefined) { e.textContent = text; }
    return e;
  }
  function save(name, type, text) {
    var url = URL.createObjectURL(new Blob([text], { type: type }));
    var a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }
  function pct(p) { return M.decimal(p * 100, 2) + ' %'; }

  var state = { reference: null, turtle: '', source: '', renders: 0, timer: null };
  var inputs = [], layoutSel, pcSel;

  // ---- input -------------------------------------------------------------
  function readCounts() {
    var values = [], ok = true;
    inputs.forEach(function (input) {
      var raw = input.value.trim(), n = raw === '' ? 0 : Number(raw);
      var good = isFinite(n) && n >= 0 && Math.floor(n) === n && n <= 1e7;
      input.classList.toggle('is-bad', !good);
      if (!good) { ok = false; }
      values.push(good ? n : 0);
    });
    return ok ? values : null;
  }
  function setCounts(values) {
    inputs.forEach(function (input, i) { input.value = values[i]; });
    update();
  }
  function fromHash() {
    var parts = decodeURIComponent(location.hash.replace(/^#/, '')).split(',');
    if (parts.length !== 5) { return null; }
    var v = parts.map(Number);
    return v.every(function (n) { return isFinite(n) && n >= 0 && Math.floor(n) === n; }) ? v : null;
  }

  // ---- result ------------------------------------------------------------
  function drawBars(p) {
    var W = 760, left = 130, right = 70, rowH = 34, top = 6, H = top + rowH * 5 + 24;
    var best = p.indexOf(Math.max.apply(null, p));
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img',
      'aria-label': 'Probability of each horizon' });
    for (var t = 0; t <= 100; t += 25) {
      var x = left + (W - left - right) * t / 100;
      svg.appendChild(svgEl('line', { x1: x, x2: x, y1: top, y2: top + rowH * 5, 'class': 'axis' }));
      svg.appendChild(svgEl('text', { x: x, y: H - 4, 'text-anchor': 'middle', 'class': 'muted' }, t + ' %'));
    }
    p.forEach(function (v, i) {
      var y = top + i * rowH, w = (W - left - right) * v;
      svg.appendChild(svgEl('text', { x: left - 10, y: y + 21, 'text-anchor': 'end' }, 'Horizon ' + (i + 1)));
      svg.appendChild(svgEl('rect', { x: left, y: y + 5, width: Math.max(w, 1), height: rowH - 10,
        rx: 3, 'class': 'bar' + (i === best ? ' best' : '') }));
      svg.appendChild(svgEl('text', { x: left + w + 8, y: y + 21 }, pct(v)));
    });
    var box = $('dm-bars');
    box.textContent = '';
    box.appendChild(svg);
    $('dm-verdict').innerHTML = 'Most probable: <strong>horizon ' + (best + 1) + '</strong> (' + pct(p[best]) + ').';
  }

  function drawGraph() {
    var canvas = $('dm-graph-canvas');
    var text = state.source.replace(/^flowchart\s+\w+;/, 'flowchart ' + layoutSel.value + ';');
    var id = ++state.renders;
    window.mermaid.render('dm-graph-svg-' + id, text + '\n' + CLASS_DEFS.join('\n'))
      .then(function (out) {
        if (id !== state.renders) { return; }          // a newer render is on its way
        canvas.innerHTML = out.svg;
        $('dm-download-svg').disabled = false;
      })
      .catch(function (err) {
        if (id !== state.renders) { return; }
        canvas.textContent = '';
        var p = document.createElement('p');
        p.className = 'dm-graph__error';
        p.textContent = 'The graph could not be drawn (' + (err && err.message ? err.message : 'render error') +
          '). The triples are still available below.';
        canvas.appendChild(p);
        $('dm-download-svg').disabled = true;
      });
  }

  function update() {
    var counts = readCounts(), msg = $('dm-msg');
    var downloads = [$('dm-download-ttl'), $('dm-download-svg')];
    if (!counts) { msg.textContent = 'Please enter whole numbers, 0 or more.'; return; }
    if (counts.every(function (n) { return n === 0; })) {
      msg.textContent = 'Enter at least one sherd.';
      downloads.forEach(function (b) { b.disabled = true; });
      return;
    }
    msg.textContent = '';
    var pc = Number(pcSel.value);
    var p = M.posterior(counts, state.reference, pc);
    state.turtle = M.buildTurtle(counts, p, pc) + '\n';
    state.source = M.buildMermaid(counts, p, pc);
    $('dm-turtle-view').textContent = state.turtle;
    $('dm-download-ttl').disabled = false;
    drawBars(p);
    drawGraph();
    try { history.replaceState(null, '', '#' + counts.join(',')); } catch (e) { /* file: or blocked */ }
  }

  function later() { clearTimeout(state.timer); state.timer = setTimeout(update, 250); }

  // ---- reference table ---------------------------------------------------
  function drawReference(ref) {
    var table = $('dm-ref'), html = '<thead><tr><th>Horizon</th><th>n</th>';
    ref.labels.forEach(function (l) { html += '<th>' + l + '</th>'; });
    html += '</tr></thead><tbody>';
    ref.horizons.forEach(function (h) {
      html += '<tr><td>' + h.horizon + '</td><td>' + h.n_total + '</td>';
      h.counts.forEach(function (c) { html += '<td>' + c + '</td>'; });
      html += '</tr>';
    });
    table.innerHTML = html + '</tbody>';
  }

  // ---- start -------------------------------------------------------------
  function init() {
    inputs = Array.prototype.slice.call(document.querySelectorAll('.dm-form input'));
    layoutSel = $('dm-layout'); pcSel = $('dm-pseudocount');

    if (!window.mermaid || typeof window.mermaid.render !== 'function') {
      $('dm-msg').textContent = 'vendor/mermaid.min.js was not loaded.'; return;
    }
    window.mermaid.initialize({
      startOnLoad: false, securityLevel: 'strict', theme: 'base',
      themeVariables: { fontFamily: 'Arial, sans-serif', fontSize: '12px' },
      htmlLabels: false,       // plain SVG text: the downloaded SVG opens in Inkscape and the like
      flowchart: { htmlLabels: false, curve: 'basis', nodeSpacing: 18, rankSpacing: 36, wrappingWidth: 220 }
    });

    try {
      var saved = window.localStorage.getItem('dmgraph.layout');
      if (saved && /^(TB|LR|BT|RL)$/.test(saved)) { layoutSel.value = saved; }
    } catch (e) { /* storage blocked: default */ }

    inputs.forEach(function (input) { input.addEventListener('input', later); });
    pcSel.addEventListener('change', update);
    layoutSel.addEventListener('change', function () {
      try { window.localStorage.setItem('dmgraph.layout', layoutSel.value); } catch (e) { /* ignore */ }
      if (state.source) { drawGraph(); }
    });
    $('dm-example').addEventListener('click', function () { setCounts(EXAMPLE); });
    $('dm-clear').addEventListener('click', function () { setCounts([0, 0, 0, 0, 0]); });
    $('dm-download-ttl').addEventListener('click', function () {
      save('horizon-assessment.ttl', 'text/turtle;charset=utf-8', state.turtle);
    });
    $('dm-download-svg').addEventListener('click', function () {
      var svg = $('dm-graph-canvas').querySelector('svg');
      if (!svg) { return; }
      save('horizon-assessment.svg', 'image/svg+xml;charset=utf-8',
        '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(svg));
    });

    var ref = window.DMReference;
    if (!ref) { $('dm-msg').textContent = 'dmgraph/reference.js was not loaded.'; return; }
    state.reference = ref.horizons.map(function (h) { return h.counts; });
    drawReference(ref);
    setCounts(fromHash() || EXAMPLE);
  }

  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', init); }
  else { init(); }
}());
