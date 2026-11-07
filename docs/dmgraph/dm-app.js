/* dm-app.js — the dating-machine page: form, charts, graph, downloads.
 *
 * Needs reference.js, dm-model.js, dm-charts.js and vendor/mermaid.min.js loaded
 * first. The colours of the RDF graph (grey, purple, heavy outline) are defined
 * here and only here; the colours of the charts are in dm-charts.js.
 */
(function () {
  'use strict';

  var M = window.DMModel, C = window.DMCharts, P = window.DMPoster;
  var CLASS_DEFS = [
    'classDef ent fill:#e2e8f0,stroke:#000000,color:#000000;',   // subject / object nodes
    'classDef term fill:#4c1d95,stroke:#000000,color:#ffffff;',  // concepts of the typology
    'classDef best stroke-width:4px;'                            // most probable horizon
  ];
  var EXAMPLE = [0, 0, 13, 27, 60];
  var FEW = 10;       // below this many sherds the result is called indicative
  var CLEAR = 0.6;    // below this probability no horizon "stands out"

  function $(id) { return document.getElementById(id); }
  function save(name, type, text) {
    var url = URL.createObjectURL(new Blob([text], { type: type }));
    var a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }
  function pct(p) { return M.decimal(p * 100, 2) + ' %'; }
  function remember(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* blocked */ } }
  function recall(key, pattern) {
    try { var v = window.localStorage.getItem(key); return v && pattern.test(v) ? v : null; }
    catch (e) { return null; }
  }

  var state = { ref: null, reference: null, counts: null, p: null, meta: null,
                turtle: '', source: '', renders: 0, timer: null };
  var inputs = [], layoutSel, viewSel, pcSel;

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
  function setCounts(values) { inputs.forEach(function (input, i) { input.value = values[i]; }); update(); }
  function fromHash() {
    var parts = decodeURIComponent(location.hash.replace(/^#/, '')).split(',');
    if (parts.length !== 5) { return null; }
    var v = parts.map(Number);
    return v.every(function (n) { return isFinite(n) && n >= 0 && Math.floor(n) === n; }) ? v : null;
  }

  // ---- result: bars, notes, profile, timeline ----------------------------
  function put(id, svg) { var box = $(id); box.textContent = ''; box.appendChild(svg); }

  function note(n, p) {
    var max = Math.max.apply(null, p);
    if (n < FEW) {
      return 'Only ' + n + (n === 1 ? ' sherd' : ' sherds') + ': the probabilities are indicative at best, ' +
        'a few more or fewer sherds could move them a long way.';
    }
    if (max < CLEAR) {
      return 'No horizon clearly stands out: the highest probability is ' + pct(max) +
        '. The assemblage fits several horizons about equally.';
    }
    return '';
  }

  function profileTable(counts) {
    var n = counts.reduce(function (a, b) { return a + b; }, 0);
    var rows = [['Assemblage'].concat(counts.map(function (c) { return (100 * c / n).toFixed(1); }))];
    state.ref.horizons.forEach(function (h) {
      var s = h.counts.reduce(function (a, b) { return a + b; }, 0);
      rows.push(['Horizon ' + h.horizon].concat(h.counts.map(function (c) { return (100 * c / s).toFixed(1); })));
    });
    var html = '<thead><tr><th>Share of sherds, %</th>' +
      state.ref.labels.map(function (l) { return '<th>' + l + '</th>'; }).join('') + '</tr></thead><tbody>';
    rows.forEach(function (r) {
      html += '<tr>' + r.map(function (c, k) { return k ? '<td>' + c + '</td>' : '<td>' + c + '</td>'; }).join('') + '</tr>';
    });
    $('dm-profile-table').innerHTML = html + '</tbody>';
  }

  function drawResult() {
    var p = state.p, counts = state.counts, best = p.indexOf(Math.max.apply(null, p));
    var n = counts.reduce(function (a, b) { return a + b; }, 0);
    put('dm-bars', C.bars(p));
    $('dm-verdict').innerHTML = 'Most probable: <strong>horizon ' + (best + 1) + '</strong> (' + pct(p[best]) +
      ') from ' + n + (n === 1 ? ' sherd' : ' sherds') + '.';
    var text = note(n, p), box = $('dm-note');
    box.hidden = !text; box.textContent = text;
    put('dm-profile', C.profile(counts, state.ref, p));
    put('dm-timeline', C.timeline(p, state.ref));
    profileTable(counts);
  }

  // ---- graph -------------------------------------------------------------
  function legend(view) {
    var box = $('dm-legend');
    box.innerHTML = view === 'poster'
      ? '<span class="sw sw--ent"></span> resource or blank node &nbsp; ' +
        '<span class="sw sw--real"></span> real object (the sherds) &nbsp; ' +
        '<span class="sw sw--class"></span> class (<code>rdf:type</code>) &nbsp; ' +
        '<span class="sw sw--term"></span> SKOS concept &nbsp; ' +
        '<span class="sw sw--lit"></span> literal or property value &nbsp; ' +
        '<span class="sw sw--owl"></span> OWL-Time / SKOS term. ' +
        'Dashed frames group five alike (×5); heavy outline: most probable horizon. Point at a node to follow its edges.'
      : view === 'simple'
      ? 'Arrow width and label: probability of the horizon. Each arrow stands for one result ' +
        '(<code>dm:HorizonResult</code>, a CIDOC-CRM <code>E13</code> attribute assignment). ' +
        'The heavy outline marks the most probable horizon.'
      : '<span class="sw sw--ent"></span> resource or blank node &nbsp; ' +
        '<span class="sw sw--term"></span> concept of the service typology (<code>aevoc:</code>) &nbsp; ' +
        '<span class="sw sw--best"></span> most probable horizon. ' +
        'Edge labels are the CIDOC-CRM and <code>dm:</code> properties of the Turtle; ' +
        'point at a node to see what it is and what it touches.';
  }

  function fail(message) {
    var canvas = $('dm-graph-canvas');
    canvas.textContent = '';
    var p = document.createElement('p');
    p.className = 'dm-graph__error';
    p.textContent = 'The graph could not be drawn (' + message + '). The triples are still available below.';
    canvas.appendChild(p);
    $('dm-download-svg').disabled = true; $('dm-download-png').disabled = true;
  }

  // Points at a node: it and what it touches stay, everything else fades.
  function decorate(svg) {
    var info = M.nodeInfo(state.meta), nodes = {}, edges = [], labels = {};
    svg.querySelectorAll('g.node').forEach(function (g) {
      var m = /-flowchart-([A-Za-z0-9]+)-\d+$/.exec(g.id);
      if (!m) { return; }
      nodes[m[1]] = g;
      g.setAttribute('tabindex', '0');
      if (info[m[1]]) {
        var t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        t.textContent = info[m[1]];
        g.appendChild(t);
      }
    });
    svg.querySelectorAll('path.flowchart-link').forEach(function (path) {
      var m = /-(L_([A-Za-z0-9]+)_([A-Za-z0-9]+)_\d+)$/.exec(path.id);
      if (m) { edges.push({ id: m[1], from: m[2], to: m[3], path: path }); }
    });
    svg.querySelectorAll('g.edgeLabel').forEach(function (g) {
      var inner = g.querySelector('[data-id]');
      if (inner) { labels[inner.getAttribute('data-id')] = g; }
    });

    function clear() { svg.querySelectorAll('.dm-dim').forEach(function (e) { e.classList.remove('dm-dim'); }); }
    function focus(key) {
      var near = {}, keep = {};
      near[key] = true;
      edges.forEach(function (e) {
        if (e.from === key || e.to === key) { near[e.from] = true; near[e.to] = true; keep[e.id] = true; }
      });
      Object.keys(nodes).forEach(function (k) { nodes[k].classList.toggle('dm-dim', !near[k]); });
      edges.forEach(function (e) {
        e.path.classList.toggle('dm-dim', !keep[e.id]);
        if (labels[e.id]) { labels[e.id].classList.toggle('dm-dim', !keep[e.id]); }
      });
    }
    Object.keys(nodes).forEach(function (k) {
      var g = nodes[k];
      g.addEventListener('mouseenter', function () { focus(k); });
      g.addEventListener('focus', function () { focus(k); });
      g.addEventListener('mouseleave', clear);
      g.addEventListener('blur', clear);
    });
  }

  function drawGraph() {
    var canvas = $('dm-graph-canvas'), view = viewSel.value;
    $('dm-layout-wrap').hidden = view !== 'full';
    $('dm-download-png').hidden = view === 'full';
    legend(view);
    var id = ++state.renders;                 // a newer draw makes an older one stale
    if (view === 'poster') {
      canvas.textContent = '';
      canvas.appendChild(P.build({ counts: state.counts, p: state.p, ref: state.ref, meta: state.meta,
        pseudocountText: M.decimal(Number(pcSel.value), 6), labels: M.LABELS, slugs: M.SLUGS, yearLabel: C.yearLabel }));
      $('dm-download-svg').disabled = false; $('dm-download-png').disabled = false;
      return;
    }
    if (view === 'simple') {
      canvas.textContent = '';
      canvas.appendChild(C.simple(state.counts, state.p, state.ref, state.meta));
      $('dm-download-svg').disabled = false;
      return;
    }
    if (!window.mermaid || typeof window.mermaid.render !== 'function') {
      fail('vendor/mermaid.min.js was not loaded'); return;
    }
    var text = state.source.replace(/^flowchart\s+\w+;/, 'flowchart ' + layoutSel.value + ';');
    window.mermaid.render('dm-graph-svg-' + id, text + '\n' + CLASS_DEFS.join('\n'))
      .then(function (out) {
        if (id !== state.renders) { return; }
        canvas.innerHTML = out.svg;
        decorate(canvas.querySelector('svg'));
        $('dm-download-svg').disabled = false;
      })
      .catch(function (err) {
        if (id !== state.renders) { return; }
        fail(err && err.message ? err.message : 'render error');
      });
  }

  // ---- update ------------------------------------------------------------
  function update() {
    var counts = readCounts(), msg = $('dm-msg');
    if (!counts) { msg.textContent = 'Please enter whole numbers, 0 or more.'; return; }
    if (counts.every(function (n) { return n === 0; })) {
      msg.textContent = 'Enter at least one sherd.';
      $('dm-download-ttl').disabled = true; $('dm-download-svg').disabled = true; $('dm-download-png').disabled = true;
      return;
    }
    msg.textContent = '';
    var pc = Number(pcSel.value);
    var p = M.posterior(counts, state.reference, pc);
    var ll = M.logLikelihood(counts, state.reference, pc);
    state.counts = counts; state.p = p;
    state.meta = M.makeMeta(M.nowUtc(), state.ref.countsSha256);
    state.turtle = M.buildTurtle(counts, p, ll, pc, state.meta) + '\n';
    state.source = M.buildMermaid(counts, p, pc, state.meta);
    $('dm-turtle-view').textContent = state.turtle;
    $('dm-download-ttl').disabled = false;
    drawResult();
    drawGraph();
    try { history.replaceState(null, '', '#' + counts.join(',')); } catch (e) { /* file: or blocked */ }
  }
  function later() { clearTimeout(state.timer); state.timer = setTimeout(update, 250); }

  // ---- reference table ---------------------------------------------------
  function drawReference(ref) {
    var html = '<thead><tr><th>Horizon</th><th>Years</th><th>n</th>' +
      ref.labels.map(function (l) { return '<th>' + l + '</th>'; }).join('') + '</tr></thead><tbody>';
    ref.horizons.forEach(function (h) {
      html += '<tr><td>' + h.horizon + '</td><td>' + C.yearLabel(h.begin) + ' – ' + C.yearLabel(h.end) +
        '</td><td>' + h.n_total + '</td>' + h.counts.map(function (c) { return '<td>' + c + '</td>'; }).join('') + '</tr>';
    });
    $('dm-ref').innerHTML = html + '</tbody>';
  }

  // ---- start -------------------------------------------------------------
  function init() {
    inputs = Array.prototype.slice.call(document.querySelectorAll('.dm-form input'));
    layoutSel = $('dm-layout'); viewSel = $('dm-view'); pcSel = $('dm-pseudocount');

    var ref = window.DMReference;
    if (!ref) { $('dm-msg').textContent = 'dmgraph/reference.js was not loaded.'; return; }
    state.ref = ref;
    state.reference = ref.horizons.map(function (h) { return h.counts; });
    drawReference(ref);

    if (window.mermaid && typeof window.mermaid.initialize === 'function') {
      window.mermaid.initialize({
        startOnLoad: false, securityLevel: 'strict', theme: 'base',
        themeVariables: { fontFamily: 'Arial, sans-serif', fontSize: '12px' },
        htmlLabels: false,     // plain SVG text: the downloaded SVG opens in Inkscape and the like
        flowchart: { htmlLabels: false, curve: 'basis', nodeSpacing: 18, rankSpacing: 36, wrappingWidth: 220 }
      });
    }

    layoutSel.value = recall('dmgraph.layout', /^(TB|LR|BT|RL)$/) || 'LR';
    viewSel.value = recall('dmgraph.view', /^(simple|full|poster)$/) || 'simple';

    inputs.forEach(function (input) { input.addEventListener('input', later); });
    pcSel.addEventListener('change', update);
    layoutSel.addEventListener('change', function () {
      remember('dmgraph.layout', layoutSel.value); if (state.source) { drawGraph(); }
    });
    viewSel.addEventListener('change', function () {
      remember('dmgraph.view', viewSel.value); if (state.source) { drawGraph(); }
    });
    $('dm-example').addEventListener('click', function () { setCounts(EXAMPLE); });
    $('dm-clear').addEventListener('click', function () { setCounts([0, 0, 0, 0, 0]); });
    $('dm-download-ttl').addEventListener('click', function () {
      save('horizon-assessment.ttl', 'text/turtle;charset=utf-8', state.turtle);
    });
    function cleanSvg() {
      var svg = $('dm-graph-canvas').querySelector('svg');
      if (!svg) { return null; }
      var copy = svg.cloneNode(true);
      copy.querySelectorAll('.dm-dim').forEach(function (e) { e.classList.remove('dm-dim'); });
      copy.querySelectorAll('[tabindex]').forEach(function (e) { e.removeAttribute('tabindex'); });
      if (!copy.getAttribute('xmlns')) { copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg'); }
      return copy;
    }
    $('dm-download-svg').addEventListener('click', function () {
      var copy = cleanSvg();
      if (!copy) { return; }
      save('horizon-assessment.svg', 'image/svg+xml;charset=utf-8',
        '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(copy));
    });
    // PNG at three times the drawn size (the SVG carries its own fonts and colours).
    $('dm-download-png').addEventListener('click', function () {
      var copy = cleanSvg();
      if (!copy) { return; }
      var vb = copy.getAttribute('viewBox').split(/\s+/).map(Number), k = 3;
      copy.setAttribute('width', vb[2] * k); copy.setAttribute('height', vb[3] * k);
      var url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)], { type: 'image/svg+xml;charset=utf-8' }));
      var img = new Image();
      img.onload = function () {
        var cv = document.createElement('canvas'); cv.width = vb[2] * k; cv.height = vb[3] * k;
        var ctx = cv.getContext('2d'); ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, cv.width, cv.height);
        ctx.drawImage(img, 0, 0, cv.width, cv.height); URL.revokeObjectURL(url);
        cv.toBlob(function (b) {
          var u = URL.createObjectURL(b), a = document.createElement('a');
          a.href = u; a.download = 'horizon-assessment.png'; document.body.appendChild(a); a.click(); a.remove();
          setTimeout(function () { URL.revokeObjectURL(u); }, 1000);
        }, 'image/png');
      };
      img.src = url;
    });

    setCounts(fromHash() || EXAMPLE);
  }

  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', init); }
  else { init(); }
}());
