/* dm-model.js — the dating machine's model and the two texts it produces.
 *
 * Dirichlet-multinomial predictive likelihood, the reference counts of the five
 * horizons as Dirichlet parameters plus a pseudocount, equal priors; then the
 * result as Turtle and as Mermaid source. No DOM, no dependencies: the browser
 * page and the check in py/build_dating.py (node) load the same file.
 *
 * It mirrors py/dm_graph_example.py and cfm/dmgraph/dm_graph.cfm line for line.
 * Change one, change the others; py/build_dating.py compares them byte for byte.
 */
(function (root) {
  'use strict';

  var BASE = 'http://leiza-scit.github.io/CAA2026-alligator/';
  var SLUGS = ['Schraegrandteller', 'ServiceIa', 'ServiceIb', 'ServiceIc', 'ServiceII'];
  var LABELS = ['BordOblique', 'Ia', 'Ib', 'Ic', 'II'];

  // Lanczos approximation (g = 7, n = 9); agrees with math.lgamma to about 1e-14.
  var LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

  function logGamma(x) {
    if (x < 0.5) { return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x); }
    x -= 1;
    var a = LANCZOS[0], t = x + 7.5;
    for (var i = 1; i < 9; i++) { a += LANCZOS[i] / (x + i); }
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  /** Normalised posterior over the five horizons. reference: five rows of five counts. */
  function posterior(observed, reference, pseudocount) {
    var n = 0, i, j;
    for (i = 0; i < 5; i++) { n += observed[i]; }
    var logs = [];
    for (i = 0; i < 5; i++) {
      var total = 0, alpha = [];
      for (j = 0; j < 5; j++) { alpha.push(reference[i][j] + pseudocount); total += alpha[j]; }
      var v = logGamma(total) - logGamma(total + n);
      for (j = 0; j < 5; j++) { v += logGamma(alpha[j] + observed[j]) - logGamma(alpha[j]); }
      logs.push(v - Math.log(5));
    }
    var top = Math.max.apply(null, logs), weights = [], sum = 0;
    for (i = 0; i < 5; i++) { weights.push(Math.exp(logs[i] - top)); sum += weights[i]; }
    return weights.map(function (w) { return w / sum; });
  }

  // Integer arithmetic, no locale, no exponent: what dmDecimal() does in the cfm.
  function decimal(value, digits) {
    var scale = Math.pow(10, digits);
    var scaled = Math.floor(value * scale + 0.5);
    var whole = Math.floor(scaled / scale);
    var frac = String(scaled - whole * scale);
    while (frac.length < digits) { frac = '0' + frac; }
    return whole + '.' + frac;
  }

  function buildTurtle(observed, probabilities, pseudocount) {
    var prior = decimal(1 / 5, 6), i;
    var t = [
      '@prefix dm: <' + BASE + 'dm/> .',
      '@prefix aeont: <' + BASE + 'ontology#> .',
      '@prefix aevoc: <' + BASE + 'vocabulary/> .',
      '@prefix aecol: <' + BASE + 'collections/> .',
      '@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .',
      '@prefix lado: <http://archaeology.link/ontology#> .',
      '',
      '_:assemblage a dm:Assemblage , crm:E18_Physical_Thing ;',
      '    dm:hasCount _:count_1 , _:count_2 , _:count_3 , _:count_4 , _:count_5 .'
    ];
    for (i = 0; i < 5; i++) {
      t.push('', '_:count_' + (i + 1) + ' a aeont:ServiceCount ;',
        '    aeont:serviceType aevoc:' + SLUGS[i] + ' ;',
        '    aeont:sherdCount ' + observed[i] + ' .');
    }
    t.push('', '[] a dm:HorizonAssessment , crm:E7_Activity ;',
      '    dm:method "Dirichlet-multinomial predictive likelihood" ;',
      '    dm:pseudocount ' + decimal(pseudocount, 6) + ' ;',
      '    crm:P9_consists_of');
    for (i = 0; i < 5; i++) {
      t.push('        [ a dm:HorizonResult , crm:E13_Attribute_Assignment' +
        ' ; crm:P140_assigned_attribute_to _:assemblage' +
        ' ; crm:P141_assigned aecol:horizon_' + (i + 1) +
        ' ; dm:prior ' + prior +
        ' ; dm:probability ' + decimal(probabilities[i], 6) + ' ]' + (i < 4 ? ' ,' : ' .'));
    }
    for (i = 0; i < 5; i++) {
      t.push('', 'aecol:horizon_' + (i + 1) + ' a lado:ChronologicalHorizon , crm:E4_Period .');
    }
    return t.join('\n');
  }

  function buildMermaid(observed, probabilities, pseudocount) {
    var best = 0, i, q = '"';
    for (i = 1; i < 5; i++) { if (probabilities[i] > probabilities[best]) { best = i; } }
    best += 1;
    var m = ['flowchart TB;',
      'asm(' + q + '_:assemblage<br/>dm:Assemblage<br/>crm:E18_Physical_Thing' + q + ');'];
    for (i = 0; i < 5; i++) {
      m.push('c' + (i + 1) + '(' + q + '_:count_' + (i + 1) + '<br/>aeont:ServiceCount<br/>' +
        LABELS[i] + ': ' + observed[i] + ' sherds' + q + ');');
    }
    for (i = 0; i < 5; i++) { m.push('s' + (i + 1) + '(' + q + 'aevoc:' + SLUGS[i] + q + ');'); }
    m.push('asx(' + q + '_:assessment<br/>dm:HorizonAssessment<br/>crm:E7_Activity<br/>' +
      'pseudocount ' + decimal(pseudocount, 6) + q + ');');
    for (i = 0; i < 5; i++) {
      m.push('r' + (i + 1) + '(' + q + 'dm:HorizonResult<br/>crm:E13_Attribute_Assignment<br/>P = ' +
        decimal(probabilities[i] * 100, 2) + ' %' + q + ');');
    }
    for (i = 0; i < 5; i++) {
      m.push('h' + (i + 1) + '(' + q + 'aecol:horizon_' + (i + 1) +
        '<br/>lado:ChronologicalHorizon<br/>crm:E4_Period' + q + ');');
    }
    m.push('asx -- ' + q + 'crm:P9_consists_of' + q + ' --> r1;');
    for (i = 2; i <= 5; i++) { m.push('asx --> r' + i + ';'); }
    m.push('r1 -- ' + q + 'crm:P141_assigned' + q + ' --> h1;');
    for (i = 2; i <= 5; i++) { m.push('r' + i + ' --> h' + i + ';'); }
    for (i = 1; i <= 4; i++) { m.push('r' + i + ' --> asm;'); }
    m.push('r5 -- ' + q + 'crm:P140_assigned_attribute_to' + q + ' --> asm;');
    m.push('asm -- ' + q + 'dm:hasCount' + q + ' --> c1;');
    for (i = 2; i <= 5; i++) { m.push('asm --> c' + i + ';'); }
    m.push('c1 -- ' + q + 'aeont:serviceType' + q + ' --> s1;');
    for (i = 2; i <= 5; i++) { m.push('c' + i + ' --> s' + i + ';'); }
    m.push('class asm,asx,c1,c2,c3,c4,c5,r1,r2,r3,r4,r5,h1,h2,h3,h4,h5 ent;');
    m.push('class s1,s2,s3,s4,s5 term;');
    m.push('class r' + best + ',h' + best + ' best;');
    return m.join('\n');
  }

  var api = {
    SLUGS: SLUGS, LABELS: LABELS,
    logGamma: logGamma, posterior: posterior, decimal: decimal,
    buildTurtle: buildTurtle, buildMermaid: buildMermaid
  };
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; }
  else { root.DMModel = api; }
}(this));
