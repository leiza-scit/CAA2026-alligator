# dmgraph — the dating machine's result as a graph

`services_output_quantities.cfm` attributes a group of Arretine sherds to one of
five horizons. This folder adds one thing to that page: the result as RDF
triples, drawn as a graph, with the Turtle and the SVG available for download.

Everything the browser needs is in this folder. Nothing is loaded from a CDN, so
the page works on an intranet and does not change when a library is updated.

## Install

Copy `dmgraph/` next to `services_output_quantities.cfm` and replace that file
with the one delivered with it. It differs from the plain page by these two lines,
directly before `</cfif>`,

    <!--- The result as RDF triples, drawn as a graph. Remove this line to switch it off. --->
    <cfinclude template="dmgraph/dm_graph.cfm">

and by two lines that keep the reference counts as the page reads them (the
include hashes them for the graph):

    referenceMatrix = [];                              // before the query loop
    arrayAppend(referenceMatrix, referenceCounts);     // right after referenceCounts is built

Without them the graph still works; it only leaves out the hash of the counts.

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
| `dm.ttl` | The `dm:` vocabulary, version 0.2: three classes, eight properties, with the CRM and LADO terms it reuses listed in the header. The Pages site publishes a copy at `docs/dm/`. |
| `vendor/mermaid.min.js` | Mermaid 11.17.2 (MIT, `vendor/LICENSE-mermaid.txt`). |
| `example/` | One worked assemblage: `example.ttl`, `example.mmd`, `index.html`. Written by `py/dm_graph_example.py` in the repository. |

## The graph

```turtle
_:assemblage a dm:Assemblage , crm:E18_Physical_Thing ;
    dm:totalSherds 100 ;
    crm:P12i_was_present_at _:deposition ;
    dm:hasCount _:count_1 , ... , _:count_5 .

_:deposition a crm:E5_Event .

_:count_3 a aeont:ServiceCount ;
    aeont:serviceType aevoc:ServiceIb ;
    aeont:sherdCount 13 .

_:assessment a dm:HorizonAssessment , crm:E7_Activity ;
    dm:method "Dirichlet-multinomial predictive likelihood" ;
    dm:pseudocount 0.500000 ;
    crm:P4_has_time-span [ a crm:E52_Time-Span ;
        crm:P82_at_some_time_within "2026-01-01T00:00:00Z"^^xsd:dateTime ] ;
    crm:P16_used_specific_object <.../dating.html> , <.../horizon_reference_profiles.csv> ;
    crm:P9_consists_of
        [ a dm:HorizonResult , crm:E13_Attribute_Assignment ;
          crm:P140_assigned_attribute_to _:deposition ;
          crm:P177_assigned_property_of_type crm:P10_falls_within ;
          crm:P141_assigned aecol:horizon_3 ;
          dm:prior 0.200000 ; dm:probability 0.845502 ; dm:logLikelihood -93.799599 ] ,
        ... .

<.../horizon_reference_profiles.csv> a crm:E73_Information_Object ;
    dm:countsSha256 "342fd585..." .

aecol:horizon_3 a lado:ChronologicalHorizon , crm:E4_Period .
```

The assemblage, the deposition, the assessment and every result are blank nodes
(the assessment is named `_:assessment` only so that the diagram can point at
it). The counts, the service concepts and the horizons are the ones of the
CAA2026-alligator graphs (`output/arretine_services.ttl`,
`output/arretine_sites_minigraph.ttl`), so the output merges with them.

**How the dating is modelled.** The horizon is not a property of the sherds as
objects but of the time of their deposition. The assemblage (`E18`) was present
at (`P12i`) one `E5_Event`, and each result is an `E13_Attribute_Assignment`
that assigns (`P141`) a horizon to that event (`P140`) as *falls within*
(`P177 crm:P10_falls_within`). `E5` is a kind of `E4_Period`, so `P10` holds
between event and horizon. This is the author's reading of CIDOC-CRM and has not
been reviewed by a CRM specialist; `E18` for the assemblage is the general class
and a CRMarchaeo class can replace it. The horizons are `lado:ChronologicalHorizon`
as in the minigraph; the LADO ontology could not be reached when this was written,
so no further LADO class is used.

**Where the numbers came from.** The graph names the time of the assessment, the
program (the address of the page that ran it, from the `cgi` scope; set
`dmSoftware` before the include to change it, or to `""` to leave it out) and
the reference table (the IRI of the CSV it mirrors). `dm:countsSha256` is the
SHA-256 of the 25 counts actually used, rows h1..h5 joined by `;`, the five
values of a row by `,`, so the page, the Pages site and the Python script give
the same hash whenever the numbers are the same. `dm:logLikelihood` keeps the
order of horizons whose probability rounds to `0.000000`.

Choices that are easy to miss:

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
- **Rounding.** Probabilities and log likelihoods are written with six decimals;
  a number that rounds to zero from below is written `-0.000000`.
- **Pseudocount.** The page adds 0.5 (Jeffreys) to every reference count;
  `py/horizon_assignment.py` adds 1.0. Same model, slightly different numbers
  (for the example in `example/`, 84.55 % against 85.49 % for horizon 3). The
  value used is written into every graph as `dm:pseudocount`.
- **The example is dated 2026-01-01.** `py/dm_graph_example.py` writes a fixed
  time so that the example files are reproducible (`--when` changes it).

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

The Python mirror and the page it produces have been run, in Chrome. The
ColdFusion include has not been run on a ColdFusion engine, because none was
available where it was written; its `cfscript` was translated mechanically to
JavaScript and run for five assemblages, with and without software and hash, and
the Turtle and Mermaid text it builds equal those of the Python mirror. That finds
logic and typing slips, not engine behaviour: `hash(..., "SHA-256")`, `dateConvert`,
`timeFormat` and the `cgi` scope are the calls to watch on the server.

The first thing to check there: `View source` shows two `<script>` blocks,
`dm-turtle` and `dm-mermaid`, whose contents equal `example/example.ttl` and
`example.mmd` for the same counts, apart from the time, the address of the page
(`dmSoftware`) and, if the database table differs from the CSV, the hash and
the numbers. For the CSV counts `dm:countsSha256` must read `342fd585f207...`.
