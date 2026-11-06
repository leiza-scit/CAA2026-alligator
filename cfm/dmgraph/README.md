# dmgraph — the dating machine's result as a graph

`services_output_quantities.cfm` attributes a group of Arretine sherds to one of
five horizons. This folder adds one thing to that page: the result as RDF
triples, drawn as a graph, with the Turtle and the SVG available for download.

Everything the browser needs is in this folder. Nothing is loaded from a CDN, so
the page works on an intranet and does not change when a library is updated.

## Install

Copy `dmgraph/` next to `services_output_quantities.cfm` and replace that file
with the one delivered with it. Its only change is these two lines, directly
before `</cfif>`:

    <!--- The result as RDF triples, drawn as a graph. Remove this line to switch it off. --->
    <cfinclude template="dmgraph/dm_graph.cfm">

If the folder lives somewhere else, set `dmgraphPath` before the include (URL of
the folder, trailing slash): `<cfset dmgraphPath = "/tools/dmgraph/">`.

`example/index.html` opens straight from disk and shows what the page produces
for one assemblage, without a ColdFusion server.

## What is in here

| File | Role |
|---|---|
| `dm_graph.cfm` | The include. Builds the Turtle and the Mermaid source from `observed`, `results` and `pseudocount` of the including page, and writes the HTML block. Changes no variable of the page. |
| `dm-graph.js` | Draws the diagram, adds the colours, wires the layout menu and the two download buttons. The only place the palette is defined. |
| `dm-graph.css` | Styling, all scoped to `.dm-graph`. |
| `dm.ttl` | The `dm:` vocabulary: three classes, five properties, with the CRM and LADO terms it reuses listed in the header. |
| `vendor/mermaid.min.js` | Mermaid 11.17.2 (MIT, `vendor/LICENSE-mermaid.txt`). |
| `example/` | One worked assemblage: `example.ttl`, `example.mmd`, `index.html`. Written by `py/dm_graph_example.py` in the repository. |

## The graph

```turtle
_:assemblage a dm:Assemblage , crm:E18_Physical_Thing ;
    dm:hasCount _:count_1 , _:count_2 , _:count_3 , _:count_4 , _:count_5 .

_:count_3 a aeont:ServiceCount ;
    aeont:serviceType aevoc:ServiceIb ;
    aeont:sherdCount 13 .

[] a dm:HorizonAssessment , crm:E7_Activity ;
    dm:method "Dirichlet-multinomial predictive likelihood" ;
    dm:pseudocount 0.500000 ;
    crm:P9_consists_of
        [ a dm:HorizonResult , crm:E13_Attribute_Assignment ;
          crm:P140_assigned_attribute_to _:assemblage ;
          crm:P141_assigned aecol:horizon_3 ;
          dm:prior 0.200000 ; dm:probability 0.845502 ] ,
        ... .

aecol:horizon_3 a lado:ChronologicalHorizon , crm:E4_Period .
```

The assemblage is a blank node, as is the assessment and every result. The
counts, the service concepts and the horizons are the ones of the
CAA2026-alligator graphs (`output/arretine_services.ttl`,
`output/arretine_sites_minigraph.ttl`), so the output merges with them.

**CIDOC-CRM.** The assessment is an `E7_Activity` that consists of (`P9`) five
`E13_Attribute_Assignment`s, one per horizon. Each assigns (`P141`) a horizon,
an `E4_Period`, to the assemblage (`P140`), an `E18_Physical_Thing`. The `dm:`
classes are subclasses of these, and the page states both types, so a CRM
reader needs no `dm:` knowledge. **LADO.** The horizons are typed
`lado:ChronologicalHorizon`, as in the minigraph. The repository uses only this
class and `lado:PeriodCluster` (not needed here); the LADO ontology itself
could not be reached when this was written, so no further LADO class is used.

Choices that are easy to miss:

- **Open CRM choices.** The assemblage is `E18`, the general class, and the E13
  carries no `P177_assigned_property_of_type`. Both are marked in `dm.ttl` for
  someone who knows CRM or CRMarchaeo better to decide.
- **Stage level.** The form's five categories map to the stage concepts
  `aevoc:Schraegrandteller`, `ServiceIa`, `ServiceIb`, `ServiceIc`, `ServiceII`,
  not to the nine cup/plate leaf types of the findspot graph. Cup and plate are
  merged, as in `horizon_assignment.py`.
- **Horizon numbers.** `h1`..`h5` of the database table become
  `aecol:horizon_1`..`horizon_5`. Horizon 1 is the earliest in both. The
  reference counts of the table and of `output/horizon_reference_profiles.csv`
  are identical.
- **Direction of `dm:hasCount`.** It points from the assemblage to the count,
  the other way round from `aeont:atFindspot`, which keeps the diagram a tree.
- **Rounding.** Probabilities are written with six decimals, so the two outer
  horizons of a typical assemblage read `0.000000`. The sum of the five is 1 up
  to that rounding.
- **Pseudocount.** The page adds 0.5 (Jeffreys) to every reference count;
  `py/horizon_assignment.py` adds 1.0. Same model, slightly different numbers
  (for the example in `example/`, 84.55 % against 85.49 % for horizon 3). The
  value used is written into every graph as `dm:pseudocount`.

## Layout

A menu above the buttons draws the graph top to bottom (default), left to right,
bottom to top or right to left. The server sends one direction, in the first line
of the Mermaid source (`flowchart TB;`); the menu replaces it in the browser and
draws again, so nothing is requested from the server. The last choice is kept
in `localStorage` (`dmgraph.layout`); where storage is blocked the default is
used. The SVG download is the layout currently shown. The graph is wider than
the text column, so the CSS lets it use up to 1320 px and scroll sideways below
900 px. In the left-to-right layouts a long edge label can touch a node.

## Colours

Defined in `dm-graph.js`, `CLASS_DEFS`. The server only sends the class names.

| Class | Used for | Fill |
|---|---|---|
| `ent` | assemblage, counts, assessment, results, horizons | `#e2e8f0` |
| `term` | SKOS concepts of the service typology | `#4c1d95`, white text |
| `best` | the most probable horizon: result and horizon node | heavier outline |

Nodes are rounded rectangles rather than ovals. Mermaid draws ovals as a hand-built
path of about 30 kB each, which made the downloaded SVG 1 MB for 22 nodes; with
rounded rectangles it is 52 kB.

## Editing this folder

- **No literal `#` in `dm_graph.cfm`**, neither in a string nor inside
  `<cfoutput>`: ColdFusion reads it as the start of an expression. The file uses
  `chr(35)` where it needs one, and keeps the colours in the JavaScript.
- **Numbers go through `dmDecimal()`**, never `numberFormat()`, which follows the
  server locale and would write `0,845502` into the Turtle.
- **Mermaid statements end in `;`**, so the diagram survives a server setting
  that collapses line breaks.
- **`py/dm_graph_example.py` mirrors the include** line for line. After a change
  to either, regenerate `example/` and compare it with the page.

## Updating Mermaid

    npm pack mermaid@11 && tar -xzf mermaid-*.tgz package/dist/mermaid.min.js package/LICENSE

Copy `package/dist/mermaid.min.js` to `vendor/` and `package/LICENSE` to
`vendor/LICENSE-mermaid.txt`, then open `example/index.html`. The page needs
`mermaid.render()` and the top-level `htmlLabels` switch; both exist from
Mermaid 10.

## Limits

The Python mirror and the page it produces have been run, in Chrome; the
ColdFusion include itself has not been run on a ColdFusion engine, because none
was available where it was written. The first thing to check on the server is that `View source` shows two `<script>` blocks, `dm-turtle` and
`dm-mermaid`, whose contents equal `example/example.ttl` and `example.mmd` for the
same counts apart from the numbers.
