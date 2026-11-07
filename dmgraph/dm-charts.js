/* dm-charts.js — the small charts of the dating-machine page, as plain SVG.
 *
 *   bars      probability per horizon
 *   profile   the assemblage's service profile over the five horizon profiles
 *   timeline  the five horizons on the year axis, shaded by probability
 *   simple    assemblage -> five horizons, edge width = probability
 *
 * Each function returns an <svg> element; the page decides where it goes.
 * One colour per horizon, used by every chart (the first five slots of the
 * categorical palette, checked for colour-vision deficiency). Colours of the
 * RDF graph (grey, purple) are a different thing and live in dm-app.js.
 */
(function (root) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var HORIZON = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'];
  var INK = '#0b0b0b';

  function el(name, attrs, text) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) { if (Object.prototype.hasOwnProperty.call(attrs, k)) { e.setAttribute(k, attrs[k]); } }
    if (text !== undefined) { e.textContent = text; }
    return e;
  }
  function root_(w, h, label) {
    return el('svg', { viewBox: '0 0 ' + w + ' ' + h, role: 'img', 'aria-label': label, 'class': 'dm-chart',
      'font-family': 'Arial, Helvetica, sans-serif', 'font-size': '13', fill: INK });
  }
  // The look is written as attributes, not left to a stylesheet, so that a chart
  // saved on its own (the SVG download) looks the same as on the page.
  function finish(svg) {
    svg.querySelectorAll('.muted').forEach(function (e) { e.setAttribute('fill', '#5b6470'); });
    svg.querySelectorAll('.small').forEach(function (e) { e.setAttribute('font-size', '11'); });
    svg.querySelectorAll('line.axis').forEach(function (e) {
      var zero = e.getAttribute('class').indexOf('axis--zero') >= 0;
      e.setAttribute('stroke', zero ? '#5b6470' : '#d7dce2');
      e.setAttribute('stroke-width', zero ? '1.5' : '1');
    });
    return svg;
  }
  function tip(node, text) { node.appendChild(el('title', {}, text)); return node; }
  function pct(p) { return (Math.round(p * 10000) / 100).toFixed(2) + ' %'; }
  function pct1(p) { return p.toFixed(1) + ' %'; }
  function bestOf(p) { var b = 0; for (var i = 1; i < p.length; i++) { if (p[i] > p[b]) { b = i; } } return b; }
  function yearLabel(y) { return y < 0 ? (-y) + ' BCE' : (y > 0 ? y + ' CE' : 'BCE | CE'); }

  // Labels at the end of lines must not sit on top of each other.
  function spread(ys, gap, lo, hi) {
    var order = ys.map(function (y, i) { return { y: y, i: i }; }).sort(function (a, b) { return a.y - b.y; });
    var k;
    for (k = 1; k < order.length; k++) { if (order[k].y < order[k - 1].y + gap) { order[k].y = order[k - 1].y + gap; } }
    var over = order[order.length - 1].y - hi;
    if (over > 0) { for (k = order.length - 1; k >= 0; k--) { order[k].y -= over; } }
    var under = lo - order[0].y;
    if (under > 0) { for (k = 0; k < order.length; k++) { order[k].y += under; } }
    var out = [];
    order.forEach(function (o) { out[o.i] = o.y; });
    return out;
  }

  // ---- bars --------------------------------------------------------------
  function bars(p) {
    var W = 760, left = 100, right = 70, rowH = 34, top = 6, H = top + rowH * 5 + 24;
    var best = bestOf(p), span = W - left - right, t, x;
    var svg = root_(W, H, 'Probability of each horizon');
    for (t = 0; t <= 100; t += 25) {
      x = left + span * t / 100;
      svg.appendChild(el('line', { x1: x, x2: x, y1: top, y2: top + rowH * 5, 'class': 'axis' }));
      svg.appendChild(el('text', { x: x, y: H - 4, 'text-anchor': 'middle', 'class': 'muted' }, t + ' %'));
    }
    p.forEach(function (v, i) {
      var y = top + i * rowH, w = span * v;
      svg.appendChild(el('text', { x: left - 10, y: y + 21, 'text-anchor': 'end',
        'font-weight': i === best ? '650' : '400' }, 'Horizon ' + (i + 1)));
      svg.appendChild(tip(el('rect', { x: left, y: y + 5, width: Math.max(w, 1.5), height: rowH - 10, rx: 3,
        fill: HORIZON[i], stroke: i === best ? INK : 'none', 'stroke-width': i === best ? 2 : 0 }),
        'Horizon ' + (i + 1) + ': ' + pct(v)));
      svg.appendChild(el('text', { x: left + w + 8, y: y + 21, 'font-weight': i === best ? '650' : '400' }, pct(v)));
    });
    return finish(svg);
  }

  // ---- profile -----------------------------------------------------------
  /** ref: window.DMReference; observed: five counts; p: posterior. */
  function profile(observed, ref, p) {
    var W = 760, H = 330, left = 48, right = 96, top = 14, bottom = 46;
    var best = bestOf(p), n = 0, i, j;
    for (i = 0; i < 5; i++) { n += observed[i]; }
    var obs = observed.map(function (c) { return 100 * c / n; });
    var prof = ref.horizons.map(function (h) {
      var s = 0; h.counts.forEach(function (c) { s += c; });
      return h.counts.map(function (c) { return 100 * c / s; });
    });
    var max = Math.max.apply(null, obs.concat.apply([], prof));
    var top100 = Math.max(10, Math.ceil(max / 10) * 10);
    var pw = W - left - right, ph = H - top - bottom;
    var X = function (k) { return left + pw * k / 4; };
    var Y = function (v) { return top + ph * (1 - v / top100); };
    var svg = root_(W, H, 'Service profile of the assemblage compared with the five horizon profiles');

    for (var g = 0; g <= top100; g += top100 > 50 ? 25 : 10) {
      svg.appendChild(el('line', { x1: left, x2: left + pw, y1: Y(g), y2: Y(g), 'class': 'axis' }));
      svg.appendChild(el('text', { x: left - 8, y: Y(g) + 4, 'text-anchor': 'end', 'class': 'muted' }, g + ' %'));
    }
    ref.labels.forEach(function (label, k) {
      svg.appendChild(el('text', { x: X(k), y: H - 24, 'text-anchor': 'middle' }, label));
    });
    svg.appendChild(el('text', { x: left + pw / 2, y: H - 6, 'text-anchor': 'middle', 'class': 'muted' },
      'share of sherds per service type'));

    function path(values) {
      return values.map(function (v, k) { return (k ? 'L' : 'M') + X(k).toFixed(1) + ' ' + Y(v).toFixed(1); }).join('');
    }
    // dim lines first, the most probable horizon on top of them, the assemblage last
    var order = [0, 1, 2, 3, 4].filter(function (k) { return k !== best; }).concat([best]);
    order.forEach(function (k) {
      var isBest = k === best;
      svg.appendChild(el('path', { d: path(prof[k]), fill: 'none', stroke: HORIZON[k],
        'stroke-width': isBest ? 3 : 1.75, 'stroke-linejoin': 'round', opacity: isBest ? 1 : 0.6 }));
      for (j = 0; j < 5; j++) {
        svg.appendChild(tip(el('circle', { cx: X(j), cy: Y(prof[k][j]), r: isBest ? 4.5 : 3.5, fill: HORIZON[k],
          stroke: '#fcfcfb', 'stroke-width': 2, opacity: isBest ? 1 : 0.75 }),
          'Horizon ' + (k + 1) + ', ' + ref.labels[j] + ': ' + pct1(prof[k][j])));
      }
    });
    svg.appendChild(el('path', { d: path(obs), fill: 'none', stroke: INK, 'stroke-width': 3.5,
      'stroke-linejoin': 'round', 'stroke-dasharray': '7 4' }));
    for (j = 0; j < 5; j++) {
      svg.appendChild(tip(el('rect', { x: X(j) - 5, y: Y(obs[j]) - 5, width: 10, height: 10, fill: '#fcfcfb',
        stroke: INK, 'stroke-width': 2.5, transform: 'rotate(45 ' + X(j) + ' ' + Y(obs[j]) + ')' }),
        'Assemblage, ' + ref.labels[j] + ': ' + pct1(obs[j]) + ' (' + observed[j] + ' of ' + n + ')'));
    }

    // direct labels at the right end, spread so that they do not collide
    var ends = prof.map(function (v) { return Y(v[4]) + 4; }).concat([Y(obs[4]) + 4]);
    var placed = spread(ends, 18, top + 8, top + ph);
    var names = ['H1', 'H2', 'H3', 'H4', 'H5', 'Assemblage'];
    names.forEach(function (name, k) {
      var isObs = k === 5;
      svg.appendChild(el('text', { x: left + pw + 12, y: placed[k], 'font-weight': (isObs || k === best) ? '650' : '400',
        'text-decoration': isObs ? 'underline' : 'none' }, name));
      if (!isObs) {
        svg.appendChild(el('rect', { x: left + pw + 56, y: placed[k] - 9, width: 12, height: 12, rx: 2, fill: HORIZON[k] }));
      }
    });
    return finish(svg);
  }

  // ---- timeline ----------------------------------------------------------
  function timeline(p, ref) {
    var W = 760, left = 150, right = 80, top = 8, rowH = 40, axisH = 30, H = top + rowH * 5 + axisH;
    var lo = Math.min.apply(null, ref.horizons.map(function (h) { return h.begin; }));
    var hi = Math.max.apply(null, ref.horizons.map(function (h) { return h.end; }));
    lo = Math.floor((lo - 1) / 10) * 10; hi = Math.ceil((hi + 1) / 10) * 10;
    var pw = W - left - right, best = bestOf(p);
    var X = function (y) { return left + pw * (y - lo) / (hi - lo); };
    var svg = root_(W, H, 'The five horizons on the year axis, shaded by probability');
    var y;
    for (y = lo; y <= hi; y += 10) {
      svg.appendChild(el('line', { x1: X(y), x2: X(y), y1: top, y2: top + rowH * 5,
        'class': y === 0 ? 'axis axis--zero' : 'axis' }));
      svg.appendChild(el('text', { x: X(y), y: H - 8, 'text-anchor': 'middle', 'class': 'muted' }, yearLabel(y)));
    }
    ref.horizons.forEach(function (h, i) {
      var ry = top + i * rowH, x1 = X(h.begin), x2 = X(h.end);
      var years = yearLabel(h.begin) + ' – ' + yearLabel(h.end);
      svg.appendChild(el('text', { x: left - 12, y: ry + 18, 'text-anchor': 'end',
        'font-weight': i === best ? '650' : '400' }, 'Horizon ' + h.horizon));
      svg.appendChild(el('text', { x: left - 12, y: ry + 33, 'text-anchor': 'end', 'class': 'muted small' }, years));
      // outline in the horizon colour always, fill in proportion to the probability
      svg.appendChild(tip(el('rect', { x: x1, y: ry + 7, width: x2 - x1, height: rowH - 14, rx: 3,
        fill: HORIZON[i], 'fill-opacity': (0.1 + 0.9 * p[i]).toFixed(3),
        stroke: HORIZON[i], 'stroke-width': i === best ? 3 : 1.5 }),
        'Horizon ' + h.horizon + ', ' + years + ': ' + pct(p[i])));
      svg.appendChild(el('text', { x: W - right + 10, y: ry + 25, 'font-weight': i === best ? '650' : '400' }, pct(p[i])));
    });
    return finish(svg);
  }

  // ---- simple graph ------------------------------------------------------
  function simple(observed, p, ref, meta) {
    var W = 760, rowH = 68, top = 10, H = top + rowH * 5 + 4;
    var nodeW = 200, nodeH = 58, ax = 8, hx = W - nodeW - 8, cy = top + rowH * 2.5;
    var best = bestOf(p), n = 0, i;
    for (i = 0; i < 5; i++) { n += observed[i]; }
    var svg = root_(W, H, 'The assemblage and the five horizons; the width of each arrow is the probability');
    svg.setAttribute('class', 'dm-chart dm-chart--graph');

    function node(x, y, lines, strong, title) {
      var g = el('g', {});
      g.appendChild(el('rect', { x: x, y: y - nodeH / 2, width: nodeW, height: nodeH, rx: 8,
        fill: '#e2e8f0', stroke: '#000000', 'stroke-width': strong ? 4 : 1.2 }));
      lines.forEach(function (t, k) {
        g.appendChild(el('text', { x: x + nodeW / 2, y: y - nodeH / 2 + 20 + k * 15, 'text-anchor': 'middle',
          'font-weight': k === 0 ? '650' : '400', 'class': k === 0 ? 'node-title' : 'node-sub' }, t));
      });
      return tip(g, title);
    }

    // edges first, so the nodes sit on top
    p.forEach(function (v, k) {
      var y = top + rowH * k + rowH / 2, w = Math.max(1.4, 1 + 13 * v);
      var x1 = ax + nodeW, x2 = hx;
      var edge = el('path', { d: 'M' + x1 + ' ' + cy + 'C' + (x1 + 150) + ' ' + cy + ' ' + (x2 - 150) + ' ' + y + ' ' + x2 + ' ' + y,
        fill: 'none', stroke: '#475569', 'stroke-width': w.toFixed(2), opacity: k === best ? 0.95 : 0.55 });
      svg.appendChild(tip(edge, 'Result for horizon ' + (k + 1) + ' (dm:HorizonResult, crm:E13_Attribute_Assignment): ' + pct(v)));
      var lx = x2 - 70;
      svg.appendChild(el('rect', { x: lx - 36, y: y - 10, width: 72, height: 20, rx: 10, fill: '#ffffff', stroke: '#cbd5e1' }));
      svg.appendChild(el('text', { x: lx, y: y + 5, 'text-anchor': 'middle', 'font-weight': k === best ? '650' : '400' }, pct(v)));
    });
    svg.appendChild(node(ax, cy, ['Assemblage', 'n = ' + n + ' sherds', 'blank node, crm:E18'], false,
      'Blank node: the assemblage. Its deposition event is what the horizons are assigned to.'));
    p.forEach(function (v, k) {
      var y = top + rowH * k + rowH / 2, h = ref.horizons[k];
      svg.appendChild(node(hx, y, ['Horizon ' + h.horizon, 'aecol:horizon_' + h.horizon,
        h.label.replace(/^Horizon \d+ \(|\)$/g, '')], k === best, h.iri));
    });
    return finish(svg);
  }

  var api = { HORIZON: HORIZON, bars: bars, profile: profile, timeline: timeline, simple: simple, yearLabel: yearLabel };
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; }
  else { root.DMCharts = api; }
}(this));
