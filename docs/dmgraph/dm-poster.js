/* dm-poster.js — the "full semantics" picture of one assessment, drawn for
 * posters, slides and posts.
 *
 * A fixed, hand-placed layout (the shape of the graph never changes, only its
 * numbers do), drawn as plain SVG: ovals in the project's colour scheme, groups
 * of five framed once, class nodes (rdf:type, rdfs:subClassOf) above and below
 * the instances they describe, the horizons with their OWL-Time intervals and
 * Allen relations, and where the numbers came from.
 *
 * Colour scheme (the same as the Mermaid diagrams of the project):
 *   grey   resource / blank node          dark green  real object (the sherds)
 *   orange class                          purple      concept (SKOS)
 *   white  OWL-Time / SKOS term           amber       literal / property value
 */
(function (root) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var W = 1420, H = 800;
  var STYLE = {
    res:  { fill: '#e2e8f0', stroke: '#000000', text: '#000000' },
    real: { fill: '#166534', stroke: '#000000', text: '#ffffff' },
    cls:  { fill: '#9a3412', stroke: '#000000', text: '#ffffff' },
    term: { fill: '#4c1d95', stroke: '#000000', text: '#ffffff' },
    owl:  { fill: '#ffffff', stroke: '#000000', text: '#000000' },
    lit:  { fill: '#fbbf24', stroke: '#000000', text: '#000000' }
  };
  var EDGE = { inst: '#334155', type: '#9a3412', sub: '#64748b' };

  function el(name, attrs, text) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) { if (Object.prototype.hasOwnProperty.call(attrs, k)) { e.setAttribute(k, attrs[k]); } }
    if (text !== undefined) { e.textContent = text; }
    return e;
  }
  function pct(p) { return (Math.round(p * 10000) / 100).toFixed(2) + ' %'; }

  /** d = {counts, p, ref, meta, pseudocount, labels, slugs, yearLabel} */
  function build(d) {
    var best = 0, i, n = 0;
    for (i = 1; i < 5; i++) { if (d.p[i] > d.p[best]) { best = i; } }
    for (i = 0; i < 5; i++) { n += d.counts[i]; }

    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img',
      'aria-label': 'The result as a knowledge graph: assemblage, assessment, results, horizons, classes',
      'class': 'dm-chart dm-chart--poster', 'font-family': 'Arial, Helvetica, sans-serif', 'font-size': '12' });
    var defs = el('defs', {});
    Object.keys(EDGE).forEach(function (k) {
      var m = el('marker', { id: 'dmp-arrow-' + k, viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '8',
        markerHeight: '8', orient: 'auto-start-reverse', markerUnits: 'userSpaceOnUse' });
      m.appendChild(el('path', { d: 'M0 0L10 5L0 10z', fill: EDGE[k] }));
      defs.appendChild(m);
    });
    svg.appendChild(defs);
    svg.appendChild(el('rect', { x: 0, y: 0, width: W, height: H, fill: '#ffffff' }));

    var layers = { band: el('g', {}), group: el('g', {}), edge: el('g', {}), label: el('g', {}), node: el('g', {}) };
    ['band', 'group', 'edge', 'node', 'label'].forEach(function (k) { svg.appendChild(layers[k]); });
    var nodes = {}, edges = [];

    // ---- nodes -----------------------------------------------------------
    function node(id, kind, cx, cy, w, h, lines, o) {
      o = o || {};
      var s = STYLE[kind], g = el('g', { 'class': 'dmp-node', 'data-id': id });
      g.appendChild(el('ellipse', { cx: cx, cy: cy, rx: w / 2, ry: h / 2, fill: s.fill, stroke: s.stroke,
        'stroke-width': o.heavy ? 4 : 1.2 }));
      var fs = o.size || 12, lh = fs + 2.5, y0 = cy - (lines.length - 1) * lh / 2 + fs * 0.35;
      lines.forEach(function (t, k) {
        g.appendChild(el('text', { x: cx, y: y0 + k * lh, 'text-anchor': 'middle', fill: s.text, 'font-size': fs,
          'font-weight': (k === 0 && !o.plain) ? '700' : '400' }, t));
      });
      if (o.title) { g.appendChild(el('title', {}, o.title)); }
      layers.node.appendChild(g);
      nodes[id] = { id: id, shape: 'oval', cx: cx, cy: cy, w: w, h: h, el: g, group: o.group || null, members: [] };
      return nodes[id];
    }
    function frame(id, x, y, w, h, o) {
      o = o || {};
      var g = el('g', { 'class': 'dmp-node dmp-frame', 'data-id': id });
      g.appendChild(el('rect', { x: x, y: y, width: w, height: h, rx: 14, fill: o.fill || '#f8fafc',
        stroke: '#94a3b8', 'stroke-width': 1.2, 'stroke-dasharray': '6 4' }));
      if (o.badge) {
        g.appendChild(el('rect', { x: x + w - 26, y: y - 10, width: 40, height: 20, rx: 10, fill: '#ffffff', stroke: '#94a3b8' }));
        g.appendChild(el('text', { x: x + w - 6, y: y + 4, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': '700',
          fill: '#334155' }, o.badge));
      }
      if (o.title) { g.appendChild(el('title', {}, o.title)); }
      layers.group.appendChild(g);
      nodes[id] = { id: id, shape: 'rect', cx: x + w / 2, cy: y + h / 2, w: w, h: h, el: g, group: null, members: [] };
      return nodes[id];
    }
    function band(x, y, w, h, caption) {
      layers.band.appendChild(el('rect', { x: x, y: y, width: w, height: h, rx: 16, fill: '#fff7ed', stroke: '#fed7aa' }));
      if (caption) {
        layers.band.appendChild(el('text', { x: x + 12, y: y + 17, 'font-size': 11, fill: '#9a3412',
          'font-weight': '700' }, caption));
      }
    }

    // ---- ports and edges ---------------------------------------------------
    // A point on the outline of a node, with the direction leaving it.
    function port(nd, side, t) {
      t = t || 0;
      var hw = nd.w / 2, hh = nd.h / 2, x, y;
      if (side === 'top' || side === 'bottom') {
        var sgn = side === 'top' ? -1 : 1;
        x = nd.cx + t;
        y = nd.cy + sgn * (nd.shape === 'oval' ? hh * Math.sqrt(Math.max(0, 1 - (t / hw) * (t / hw))) : hh);
        return { x: x, y: y, nx: 0, ny: sgn };
      }
      var sg = side === 'left' ? -1 : 1;
      y = nd.cy + t;
      x = nd.cx + sg * (nd.shape === 'oval' ? hw * Math.sqrt(Math.max(0, 1 - (t / hh) * (t / hh))) : hw);
      return { x: x, y: y, nx: sg, ny: 0 };
    }
    function edge(from, to, o) {
      var a = port(nodes[from], o.fs, o.ft), b = port(nodes[to], o.ts, o.tt);
      var k = Math.max(26, Math.min(0.5 * Math.hypot(b.x - a.x, b.y - a.y), o.k || 110));
      var c1 = { x: a.x + a.nx * k, y: a.y + a.ny * k }, c2 = { x: b.x + b.nx * k, y: b.y + b.ny * k };
      var kind = o.kind || 'inst';
      var path = el('path', { d: 'M' + a.x.toFixed(1) + ' ' + a.y.toFixed(1) + ' C' + c1.x.toFixed(1) + ' ' + c1.y.toFixed(1) +
        ' ' + c2.x.toFixed(1) + ' ' + c2.y.toFixed(1) + ' ' + b.x.toFixed(1) + ' ' + b.y.toFixed(1),
        fill: 'none', stroke: EDGE[kind], 'stroke-width': o.w || 1.5,
        'stroke-dasharray': kind === 'inst' ? 'none' : '5 4', 'marker-end': 'url(#dmp-arrow-' + kind + ')' });
      layers.edge.appendChild(path);
      var rec = { from: from, to: to, el: path, labelEl: null };
      if (o.label) {
        var mx = (a.x + 3 * c1.x + 3 * c2.x + b.x) / 8 + (o.lx || 0), my = (a.y + 3 * c1.y + 3 * c2.y + b.y) / 8 + (o.ly || 0);
        var rows = o.label.split('\n'), tw = 0;
        rows.forEach(function (r) { tw = Math.max(tw, r.length * 5.7 + 10); });
        var th = rows.length * 13 + 5, g = el('g', { 'class': 'dmp-label' });
        g.appendChild(el('rect', { x: mx - tw / 2, y: my - th / 2, width: tw, height: th, rx: 7, fill: '#ffffff',
          'fill-opacity': 0.92, stroke: kind === 'type' ? '#fdba74' : '#e2e8f0' }));
        rows.forEach(function (r, j) {
          g.appendChild(el('text', { x: mx, y: my - th / 2 + 13 + j * 13 - 1.5, 'text-anchor': 'middle', 'font-size': 10.5,
            fill: kind === 'type' ? '#9a3412' : '#334155', 'font-style': kind === 'inst' ? 'normal' : 'italic' }, r));
        });
        layers.label.appendChild(g);
        rec.labelEl = g;
      }
      edges.push(rec);
    }

    var LAB = d.labels, SLUG = d.slugs, REF = d.ref;
    var RX = [315, 445, 575, 705, 835], RY = 332, HY = 500;

    // ---- bands (classes) -------------------------------------------------
    band(8, 8, W - 16, 124, 'classes');
    band(228, 716, 692, 68, 'classes');
    band(962, 520, 428, 70, null);

    // ---- classes ---------------------------------------------------------
    node('k_e7', 'cls', 320, 46, 150, 30, ['crm:E7_Activity'], { plain: true });
    node('k_dha', 'cls', 440, 104, 210, 30, ['dm:HorizonAssessment'], { plain: true });
    node('k_e13', 'cls', 540, 46, 232, 30, ['crm:E13_Attribute_Assignment'], { plain: true });
    node('k_dhr', 'cls', 685, 104, 170, 30, ['dm:HorizonResult'], { plain: true });
    node('k_e5', 'cls', 1000, 104, 130, 30, ['crm:E5_Event'], { plain: true });
    node('k_e18', 'cls', 1225, 46, 206, 30, ['crm:E18_Physical_Thing'], { plain: true });
    node('k_da', 'cls', 1335, 104, 140, 30, ['dm:Assemblage'], { plain: true });
    node('k_lado', 'cls', 395, 752, 210, 30, ['lado:ChronologicalHorizon'], { plain: true });
    node('k_e4', 'cls', 575, 752, 140, 30, ['crm:E4_Period'], { plain: true });
    node('k_ti', 'owl', 715, 752, 140, 30, ['time:Interval'], { plain: true });
    node('k_sc', 'cls', 1300, 556, 176, 30, ['aeont:ServiceCount'], { plain: true });
    node('k_skos', 'owl', 1065, 556, 140, 30, ['skos:Concept'], { plain: true });

    // ---- the assessment and where it came from -----------------------------
    node('asx', 'res', 430, 200, 230, 72, ['_:assessment', 'pseudocount ' + d.pseudocountText, d.meta.when],
      { title: 'Blank node: this assessment (dm:HorizonAssessment, crm:E7_Activity). The time is crm:P4_has_time-span / crm:P82_at_some_time_within.' });
    node('ref', 'res', 115, 160, 206, 62, ['reference table', 'crm:E73_Information_Object',
      'dm:countsSha256 ' + (d.meta.refHash ? d.meta.refHash.slice(0, 8) + '…' : '–')],
      { title: d.meta.refIri + (d.meta.refHash ? '\nSHA-256 of the 25 counts: ' + d.meta.refHash : '') });
    if (d.meta.software) {
      node('sw', 'res', 115, 238, 206, 46, ['program (this page)', 'crm:E70_Thing'], { title: d.meta.software });
    }

    // ---- results, deposition, assemblage ---------------------------------
    frame('gres', 233, 276, 680, 112, { badge: '×5', title: 'Five results, one per horizon: dm:HorizonResult, crm:E13_Attribute_Assignment' });
    for (i = 0; i < 5; i++) {
      node('r' + (i + 1), 'res', RX[i], RY, 120, 42, ['_:result_' + (i + 1), 'P = ' + pct(d.p[i])],
        { heavy: i === best, group: 'gres', title: 'Blank node: result for horizon ' + (i + 1) +
          ' (crm:E13_Attribute_Assignment). dm:probability ' + pct(d.p[i]) });
    }
    node('lit_p177', 'lit', 112, 336, 214, 46, ['crm:P177_assigned_property_of_type', 'crm:P10_falls_within'], { size: 11, plain: true,
      title: 'Every result states that the deposition falls within the horizon (P177 = P10).' });
    node('lit_prior', 'lit', 112, 398, 130, 36, ['dm:prior', '0.200000'], { size: 11, plain: true });
    node('dep', 'res', 1000, 200, 140, 44, ['_:deposition'], { title: 'Blank node: the deposition event of the assemblage (crm:E5_Event). The horizons are assigned to it.' });
    node('asm', 'real', 1310, 200, 184, 72, ['_:assemblage', 'n = ' + n + ' sherds'],
      { title: 'Blank node: the assemblage (dm:Assemblage, crm:E18_Physical_Thing)' });

    // ---- counts and concepts -----------------------------------------------
    frame('gcnt', 1220, 286, 160, 190, { badge: '×5', title: 'Five counts: aeont:ServiceCount' });
    frame('gcon', 968, 286, 194, 190, { badge: '×5', title: 'Five concepts of the service typology (aevoc:)' });
    for (i = 0; i < 5; i++) {
      var y = 318 + i * 36;
      node('c' + (i + 1), 'res', 1300, y, 130, 29, [LAB[i] + ': ' + d.counts[i]], { size: 11.5, plain: true, group: 'gcnt',
        title: 'Blank node: aeont:ServiceCount, ' + LAB[i] + ', ' + d.counts[i] + ' sherds' });
      node('s' + (i + 1), 'term', 1065, y, 166, 29, ['aevoc:' + SLUG[i]], { size: 11, plain: true, group: 'gcon',
        title: 'http://leiza-scit.github.io/CAA2026-alligator/vocabulary/' + SLUG[i] });
    }

    // ---- horizons, intervals, Allen relations -------------------------------
    frame('ghor', 233, 456, 680, 232, { badge: '×5', title: 'The five chronological horizons of the site graph (aecol:)' });
    for (i = 0; i < 5; i++) {
      var h = REF.horizons[i];
      node('h' + (i + 1), 'res', RX[i], HY, 124, 34, ['aecol:horizon_' + (i + 1)], { heavy: i === best, plain: true, group: 'ghor',
        title: h.iri + '\n' + h.label });
      node('b' + (i + 1), 'lit', RX[i] - 32, 566, 46, 24, [d.yearLabel(h.begin)], { size: 10.5, plain: true, group: 'ghor',
        title: 'time:hasBeginning, year ' + h.begin });
      node('e' + (i + 1), 'lit', RX[i] + 32, 566, 46, 24, [d.yearLabel(h.end)], { size: 10.5, plain: true, group: 'ghor',
        title: 'time:hasEnd, year ' + h.end });
    }

    // ---- register group members ---------------------------------------------
    Object.keys(nodes).forEach(function (id) {
      var g = nodes[id].group;
      if (g) { nodes[g].members.push(id); }
    });

    // ---- edges: classes -----------------------------------------------------
    var T = function (f, t, o) { o = o || {}; o.kind = 'type'; edge(f, t, o); };
    T('asx', 'k_dha', { fs: 'top', ft: 10, ts: 'bottom', label: 'rdf:type', ly: 4 });
    T('asx', 'k_e7', { fs: 'top', ft: -60, ts: 'bottom', tt: 10 });
    T('gres', 'k_dhr', { fs: 'top', ft: 30, ts: 'bottom', tt: 10 });
    T('gres', 'k_e13', { fs: 'top', ft: -20, ts: 'bottom', tt: 30 });
    T('dep', 'k_e5', { fs: 'top', ts: 'bottom' });
    T('asm', 'k_da', { fs: 'top', ft: 30, ts: 'bottom', tt: -5 });
    T('asm', 'k_e18', { fs: 'top', ft: -30, ts: 'bottom', tt: 20 });
    T('gcnt', 'k_sc', { fs: 'bottom', ts: 'top' });
    T('gcon', 'k_skos', { fs: 'bottom', ts: 'top' });
    T('ghor', 'k_lado', { fs: 'bottom', ft: -60, ts: 'top', tt: 20 });
    T('ghor', 'k_e4', { fs: 'bottom', ft: -10, ts: 'top' });
    T('ghor', 'k_ti', { fs: 'bottom', ft: 40, ts: 'top', tt: -10 });
    var S = function (f, t, o) { o = o || {}; o.kind = 'sub'; edge(f, t, o); };
    S('k_dha', 'k_e7', { fs: 'left', ts: 'bottom', tt: 28, label: 'rdfs:subClassOf', ly: -2, lx: -8 });
    S('k_dhr', 'k_e13', { fs: 'left', ts: 'bottom', tt: 70 });
    S('k_da', 'k_e18', { fs: 'left', ts: 'bottom', tt: 70 });

    // ---- edges: instances ---------------------------------------------------
    for (i = 0; i < 5; i++) {
      edge('asx', 'r' + (i + 1), { fs: 'bottom', ft: -70 + i * 35, ts: 'top', tt: -26, label: i === 0 ? 'crm:P9_consists_of' : '', lx: i === 0 ? -50 : 0, ly: 4 });
      edge('r' + (i + 1), 'dep', { fs: 'top', ft: 26, ts: 'bottom', tt: -50 + i * 25, label: i === 4 ? 'crm:P140_assigned_\nattribute_to' : '', lx: i === 4 ? 14 : 0, ly: i === 4 ? 6 : 0 });
      edge('r' + (i + 1), 'h' + (i + 1), { fs: 'bottom', ts: 'top', label: i === 0 ? 'crm:P141_assigned' : '', lx: i === 0 ? 0 : 0 });
      edge('c' + (i + 1), 's' + (i + 1), { fs: 'left', ts: 'right', label: i === 0 ? 'aeont:\nserviceType' : '' });
      edge('h' + (i + 1), 'b' + (i + 1), { fs: 'bottom', ft: -36, ts: 'top', tt: 0, label: i === 0 ? 'time:hasBeginning' : '', lx: i === 0 ? -34 : 0, ly: -4 });
      edge('h' + (i + 1), 'e' + (i + 1), { fs: 'bottom', ft: 36, ts: 'top', tt: 0, label: i === 0 ? 'time:hasEnd' : '', lx: i === 0 ? 40 : 0, ly: -4 });
    }
    edge('asm', 'dep', { fs: 'left', ts: 'right', label: 'crm:P12i_was_\npresent_at' });
    edge('asm', 'gcnt', { fs: 'bottom', ft: 6, ts: 'top', tt: 20, label: 'dm:hasCount', lx: -52 });
    edge('asx', 'ref', { fs: 'left', ft: -14, ts: 'right', label: 'crm:P16_used_\nspecific_object', lx: 4, ly: -2 });
    if (d.meta.software) { edge('asx', 'sw', { fs: 'left', ft: 14, ts: 'right' }); }
    edge('gres', 'lit_p177', { fs: 'left', ft: -22, ts: 'right', tt: -4 });
    edge('gres', 'lit_prior', { fs: 'left', ft: 28, ts: 'right', tt: 4 });

    // Allen relations between the horizons: down through the gap between the two
    // year pills, along under them, and up into the other horizon.
    var used = {}, bottom = nodes.h1.cy + nodes.h1.h / 2, y1 = 596;
    function slot(id) { used[id] = (used[id] || 0) + 1; return (used[id] - 2) * 5; }
    REF.allen.forEach(function (a) {
      var from = nodes['h' + a.from], to = nodes['h' + a.to], span = Math.abs(a.to - a.from);
      var x1 = from.cx + slot(from.id), x2 = to.cx + slot(to.id), dd = 26 + span * 26;
      var path = el('path', { d: 'M' + x1 + ' ' + bottom + ' L' + x1 + ' ' + y1 + ' C' + x1 + ' ' + (y1 + dd) + ' ' + x2 + ' ' + (y1 + dd) +
        ' ' + x2 + ' ' + y1 + ' L' + x2 + ' ' + (bottom + 2), fill: 'none', stroke: EDGE.inst, 'stroke-width': 1.4,
        'marker-end': 'url(#dmp-arrow-inst)' });
      layers.edge.appendChild(path);
      var mx = (x1 + x2) / 2, my = y1 + 0.75 * dd + 1, rel = a.rel;
      var g = el('g', { 'class': 'dmp-label' }), tw = rel.length * 5.7 + 10;
      g.appendChild(el('rect', { x: mx - tw / 2, y: my - 9, width: tw, height: 17, rx: 7, fill: '#ffffff', 'fill-opacity': 0.95, stroke: '#e2e8f0' }));
      g.appendChild(el('text', { x: mx, y: my + 3, 'text-anchor': 'middle', 'font-size': 10.5, fill: '#334155' }, rel));
      layers.label.appendChild(g);
      edges.push({ from: 'h' + a.from, to: 'h' + a.to, el: path, labelEl: g });
    });
    layers.label.appendChild(el('text', { x: 244, y: 682, 'font-size': 10.5, fill: '#64748b', 'font-style': 'italic' },
      'time:intervalBefore and the inverse relations are in the site graph; not drawn'));

    // ---- hover -------------------------------------------------------------
    function clear() { svg.querySelectorAll('.dm-dim').forEach(function (e) { e.classList.remove('dm-dim'); }); }
    function focus(id) {
      var near = {}, keep = {}, self = nodes[id], ids = [id];
      if (self.group) { ids.push(self.group); }
      if (self.members.length) { ids = ids.concat(self.members); }
      ids.forEach(function (k) { near[k] = true; });
      edges.forEach(function (e, idx) {
        if (ids.indexOf(e.from) >= 0 || ids.indexOf(e.to) >= 0) {
          near[e.from] = true; near[e.to] = true; keep[idx] = true;
          [e.from, e.to].forEach(function (k) { var g = nodes[k].group; if (g) { near[g] = true; } });
        }
      });
      Object.keys(nodes).forEach(function (k) { nodes[k].el.classList.toggle('dm-dim', !near[k]); });
      edges.forEach(function (e, idx) {
        e.el.classList.toggle('dm-dim', !keep[idx]);
        if (e.labelEl) { e.labelEl.classList.toggle('dm-dim', !keep[idx]); }
      });
    }
    Object.keys(nodes).forEach(function (id) {
      var g = nodes[id].el;
      g.setAttribute('tabindex', '0');
      g.addEventListener('mouseenter', function () { focus(id); });
      g.addEventListener('focus', function () { focus(id); });
      g.addEventListener('mouseleave', clear);
      g.addEventListener('blur', clear);
    });
    return svg;
  }

  var api = { build: build, W: W, H: H };
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; }
  else { root.DMPoster = api; }
}(this));
