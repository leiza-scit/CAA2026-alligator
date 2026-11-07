<!--- dm_graph.cfm
     Dating-machine graph: the result of one horizon attribution as RDF triples,
     drawn as a graph in the browser.

     Include it from the output page, after the results have been computed:

         <cfinclude template="dmgraph/dm_graph.cfm">

     Reads from the including page (nothing is written back):
       observed[1..5]   sherd counts: bordoblique, ia, ib, ic, ii
       results[1..5]    structs with .horizon ("h1".."h5") and .probability
       pseudocount      the Jeffreys pseudocount used for the Dirichlet parameters
     Optional:
       dmgraphPath      URL of this folder, with trailing slash (default "dmgraph/")
       referenceMatrix  the five rows of reference counts the page used, each an
                        array of five numbers, h1 first. Gives the graph a hash of
                        the numbers it was computed from; without it that line is
                        left out.
       dmSoftware       IRI of the program; default: the address of this page,
                        built from the cgi scope. Set it to "" to leave it out.
       dmWhen           time of the assessment as 2026-01-01T00:00:00Z; default now (UTC).
       results[i].logLikelihood is used when present (the page has it), else 0.

     The Turtle and the Mermaid source are built here, on the server. The browser
     only draws them (dm-graph.js) and offers them for download.

     Two rules for editing this file:
       1. No literal hash sign inside a string or inside cfoutput. ColdFusion reads
          it as the start of an expression. dmHash = chr(35) stands in for it.
       2. Every Mermaid statement ends in a semicolon, so the diagram survives a
          server setting that collapses line breaks.
--->
<cfscript>
    if (!isDefined("dmgraphPath")) {
        dmgraphPath = "dmgraph/";
    }

    // Fixed-point text without numberFormat(), which follows the server locale
    // and would write a decimal comma into the Turtle. Declared at the top level
    // of the script: some engines reject a function declaration inside a block.
    function dmDecimal(required numeric value, required numeric digits) {
        var sign   = "";
        var scale  = 10 ^ arguments.digits;
        var scaled = 0;
        var whole  = 0;
        var frac   = 0;
        if (arguments.value LT 0) {
            sign = "-";
        }
        scaled = int(abs(arguments.value) * scale + 0.5);
        whole  = int(scaled / scale);
        frac   = scaled - (whole * scale);
        return sign & whole & "." & right(repeatString("0", arguments.digits) & frac, arguments.digits);
    }

    dmReady = isDefined("observed") AND isDefined("results")
              AND isArray(observed) AND isArray(results)
              AND arrayLen(observed) EQ 5 AND arrayLen(results) EQ 5;

    if (dmReady) {
        dmHash = chr(35);
        dmQ    = chr(34);
        dmNL   = chr(10);
        dmBase = "http://leiza-scit.github.io/CAA2026-alligator/";

        // What the graph says it was computed with. The table is named by the IRI
        // of the CSV it mirrors, and by a hash of the 25 counts actually used.
        dmRefIri = "https://github.com/leiza-scit/CAA2026-alligator/blob/main/output/horizon_reference_profiles.csv";
        dmRefHash = "";
        if (isDefined("referenceMatrix") AND isArray(referenceMatrix) AND arrayLen(referenceMatrix) EQ 5) {
            dmRows = [];
            for (dmI = 1; dmI LTE 5; dmI++) {
                dmCells = [];
                for (dmJ = 1; dmJ LTE 5; dmJ++) {
                    arrayAppend(dmCells, int(referenceMatrix[dmI][dmJ]));
                }
                arrayAppend(dmRows, arrayToList(dmCells, ","));
            }
            dmRefHash = lcase(hash(arrayToList(dmRows, ";"), "SHA-256"));
        }

        if (!isDefined("dmSoftware")) {
            dmSoftware = "";
            if (len(cgi.server_name) AND len(cgi.script_name)) {
                if (cgi.https EQ "on") {
                    dmSoftware = "https://" & cgi.server_name & cgi.script_name;
                } else {
                    dmSoftware = "http://" & cgi.server_name & cgi.script_name;
                }
            }
        }
        if (!isDefined("dmWhen")) {
            dmUtc = dateConvert("local2utc", now());
            dmWhen = dateFormat(dmUtc, "yyyy-mm-dd") & "T" & timeFormat(dmUtc, "HH:mm:ss") & "Z";
        }
        dmSwName = listLast(dmSoftware, "/");
        dmRefName = listLast(dmRefIri, "/");
        dmTotal = 0;
        for (dmI = 1; dmI LTE 5; dmI++) {
            dmTotal += int(observed[dmI]);
        }
        dmLog = [];
        for (dmI = 1; dmI LTE 5; dmI++) {
            if (structKeyExists(results[dmI], "logLikelihood")) {
                arrayAppend(dmLog, results[dmI].logLikelihood);
            } else {
                arrayAppend(dmLog, 0);
            }
        }

        // The five categories of the form, as concepts of the service typology
        // published in arretine_services.ttl (stage level, not cup/plate).
        dmSlugs  = ["Schraegrandteller", "ServiceIa", "ServiceIb", "ServiceIc", "ServiceII"];
        dmLabels = ["BordOblique", "Ia", "Ib", "Ic", "II"];

        // Horizon number per result row: "h3" -> 3, falling back to the row index.
        dmNum = [];
        for (dmI = 1; dmI LTE 5; dmI++) {
            if (reFind("^h[1-5]$", results[dmI].horizon)) {
                arrayAppend(dmNum, mid(results[dmI].horizon, 2, 1));
            } else {
                arrayAppend(dmNum, dmI);
            }
        }

        // The most probable horizon is drawn with a heavier outline.
        dmBest = 1;
        for (dmI = 2; dmI LTE 5; dmI++) {
            if (results[dmI].probability GT results[dmBest].probability) {
                dmBest = dmI;
            }
        }

        dmPrior = dmDecimal(1 / 5, 6);
        dmPseudo = dmDecimal(pseudocount, 6);

        // ---- Turtle --------------------------------------------------------
        // Mirrors py/dm_graph_example.py and docs/dmgraph/dm-model.js line for line.
        dmUsed = "";
        if (len(dmSoftware)) {
            dmUsed = "<" & dmSoftware & "> , ";
        }
        dmUsed = dmUsed & "<" & dmRefIri & ">";

        dmT = [];
        arrayAppend(dmT, "@prefix dm: <" & dmBase & "dm" & dmHash & "> .");
        arrayAppend(dmT, "@prefix aeont: <" & dmBase & "ontology" & dmHash & "> .");
        arrayAppend(dmT, "@prefix aevoc: <" & dmBase & "vocabulary/> .");
        arrayAppend(dmT, "@prefix aecol: <" & dmBase & "collections/> .");
        arrayAppend(dmT, "@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .");
        arrayAppend(dmT, "@prefix lado: <http://archaeology.link/ontology" & dmHash & "> .");
        arrayAppend(dmT, "@prefix xsd: <http://www.w3.org/2001/XMLSchema" & dmHash & "> .");
        arrayAppend(dmT, "");
        arrayAppend(dmT, "_:assemblage a dm:Assemblage , crm:E18_Physical_Thing ;");
        arrayAppend(dmT, "    dm:totalSherds " & dmTotal & " ;");
        arrayAppend(dmT, "    crm:P12i_was_present_at _:deposition ;");
        arrayAppend(dmT, "    dm:hasCount _:count_1 , _:count_2 , _:count_3 , _:count_4 , _:count_5 .");
        arrayAppend(dmT, "");
        arrayAppend(dmT, "_:deposition a crm:E5_Event .");

        for (dmI = 1; dmI LTE 5; dmI++) {
            arrayAppend(dmT, "");
            arrayAppend(dmT, "_:count_" & dmI & " a aeont:ServiceCount ;");
            arrayAppend(dmT, "    aeont:serviceType aevoc:" & dmSlugs[dmI] & " ;");
            arrayAppend(dmT, "    aeont:sherdCount " & int(observed[dmI]) & " .");
        }

        // The assessment is a CIDOC-CRM activity; each result is an attribute
        // assignment (E13) that assigns one horizon (P141) to the deposition event
        // of the assemblage (P140) as "falls within" (P177).
        arrayAppend(dmT, "");
        arrayAppend(dmT, "_:assessment a dm:HorizonAssessment , crm:E7_Activity ;");
        arrayAppend(dmT, "    dm:method " & dmQ & "Dirichlet-multinomial predictive likelihood" & dmQ & " ;");
        arrayAppend(dmT, "    dm:pseudocount " & dmPseudo & " ;");
        arrayAppend(dmT, "    crm:P4_has_time-span [ a crm:E52_Time-Span ; crm:P82_at_some_time_within "
            & dmQ & dmWhen & dmQ & "^^xsd:dateTime ] ;");
        arrayAppend(dmT, "    crm:P16_used_specific_object " & dmUsed & " ;");
        arrayAppend(dmT, "    crm:P9_consists_of");
        for (dmI = 1; dmI LTE 5; dmI++) {
            if (dmI LT 5) {
                dmEnd = " ,";
            } else {
                dmEnd = " .";
            }
            arrayAppend(dmT, "        [ a dm:HorizonResult , crm:E13_Attribute_Assignment"
                & " ; crm:P140_assigned_attribute_to _:deposition"
                & " ; crm:P177_assigned_property_of_type crm:P10_falls_within"
                & " ; crm:P141_assigned aecol:horizon_" & dmNum[dmI]
                & " ; dm:prior " & dmPrior
                & " ; dm:probability " & dmDecimal(results[dmI].probability, 6)
                & " ; dm:logLikelihood " & dmDecimal(dmLog[dmI], 6) & " ]" & dmEnd);
        }

        arrayAppend(dmT, "");
        if (len(dmRefHash)) {
            arrayAppend(dmT, "<" & dmRefIri & "> a crm:E73_Information_Object ;");
            arrayAppend(dmT, "    dm:countsSha256 " & dmQ & dmRefHash & dmQ & " .");
        } else {
            arrayAppend(dmT, "<" & dmRefIri & "> a crm:E73_Information_Object .");
        }
        if (len(dmSoftware)) {
            arrayAppend(dmT, "");
            arrayAppend(dmT, "<" & dmSoftware & "> a crm:E70_Thing .");
        }

        // The horizons are typed as in arretine_sites_minigraph.ttl, so this graph
        // says what they are without needing that one alongside.
        for (dmI = 1; dmI LTE 5; dmI++) {
            arrayAppend(dmT, "");
            arrayAppend(dmT, "aecol:horizon_" & dmNum[dmI] & " a lado:ChronologicalHorizon , crm:E4_Period .");
        }
        dmTurtle = arrayToList(dmT, dmNL);

        // ---- Mermaid -------------------------------------------------------
        // Colours are not set here; dm-graph.js adds the classDef lines, so the
        // palette lives in one place. Classes used: ent, term, best.
        dmM = [];
        arrayAppend(dmM, "flowchart TB;");
        arrayAppend(dmM, "asm(" & dmQ & "_:assemblage<br/>dm:Assemblage<br/>crm:E18_Physical_Thing<br/>n = "
            & dmTotal & " sherds" & dmQ & ");");
        arrayAppend(dmM, "dep(" & dmQ & "_:deposition<br/>crm:E5_Event" & dmQ & ");");
        for (dmI = 1; dmI LTE 5; dmI++) {
            arrayAppend(dmM, "c" & dmI & "(" & dmQ & "_:count_" & dmI & "<br/>aeont:ServiceCount<br/>"
                & dmLabels[dmI] & ": " & int(observed[dmI]) & " sherds" & dmQ & ");");
        }
        for (dmI = 1; dmI LTE 5; dmI++) {
            arrayAppend(dmM, "s" & dmI & "(" & dmQ & "aevoc:" & dmSlugs[dmI] & dmQ & ");");
        }
        arrayAppend(dmM, "asx(" & dmQ & "_:assessment<br/>dm:HorizonAssessment<br/>crm:E7_Activity<br/>pseudocount "
            & dmPseudo & "<br/>" & dmWhen & dmQ & ");");
        arrayAppend(dmM, "rf(" & dmQ & dmRefName & "<br/>crm:E73_Information_Object" & dmQ & ");");
        if (len(dmSoftware)) {
            arrayAppend(dmM, "sw(" & dmQ & dmSwName & "<br/>crm:E70_Thing" & dmQ & ");");
        }
        for (dmI = 1; dmI LTE 5; dmI++) {
            arrayAppend(dmM, "r" & dmI & "(" & dmQ & "dm:HorizonResult<br/>crm:E13_Attribute_Assignment<br/>"
                & "P177: crm:P10_falls_within<br/>P = "
                & dmDecimal(results[dmI].probability * 100, 2) & " %" & dmQ & ");");
        }
        for (dmI = 1; dmI LTE 5; dmI++) {
            arrayAppend(dmM, "h" & dmI & "(" & dmQ & "aecol:horizon_" & dmNum[dmI]
                & "<br/>lado:ChronologicalHorizon<br/>crm:E4_Period" & dmQ & ");");
        }

        // Edges follow the triples. A label is printed on one edge of each group
        // of parallel edges only (the last one for P140, where it has the room).
        arrayAppend(dmM, "asx -- " & dmQ & "crm:P9_consists_of" & dmQ & " --> r1;");
        for (dmI = 2; dmI LTE 5; dmI++) {
            arrayAppend(dmM, "asx --> r" & dmI & ";");
        }
        arrayAppend(dmM, "asx -- " & dmQ & "crm:P16_used_specific_object" & dmQ & " --> rf;");
        if (len(dmSoftware)) {
            arrayAppend(dmM, "asx --> sw;");
        }
        arrayAppend(dmM, "r1 -- " & dmQ & "crm:P141_assigned" & dmQ & " --> h1;");
        for (dmI = 2; dmI LTE 5; dmI++) {
            arrayAppend(dmM, "r" & dmI & " --> h" & dmI & ";");
        }
        for (dmI = 1; dmI LTE 4; dmI++) {
            arrayAppend(dmM, "r" & dmI & " --> dep;");
        }
        arrayAppend(dmM, "r5 -- " & dmQ & "crm:P140_assigned_attribute_to" & dmQ & " --> dep;");
        arrayAppend(dmM, "asm -- " & dmQ & "crm:P12i_was_present_at" & dmQ & " --> dep;");
        arrayAppend(dmM, "asm -- " & dmQ & "dm:hasCount" & dmQ & " --> c1;");
        for (dmI = 2; dmI LTE 5; dmI++) {
            arrayAppend(dmM, "asm --> c" & dmI & ";");
        }
        arrayAppend(dmM, "c1 -- " & dmQ & "aeont:serviceType" & dmQ & " --> s1;");
        for (dmI = 2; dmI LTE 5; dmI++) {
            arrayAppend(dmM, "c" & dmI & " --> s" & dmI & ";");
        }

        dmEnts = "asm,dep,asx,rf,";
        if (len(dmSoftware)) {
            dmEnts = dmEnts & "sw,";
        }
        arrayAppend(dmM, "class " & dmEnts & "c1,c2,c3,c4,c5,r1,r2,r3,r4,r5,h1,h2,h3,h4,h5 ent;");
        arrayAppend(dmM, "class s1,s2,s3,s4,s5 term;");
        arrayAppend(dmM, "class r" & dmBest & ",h" & dmBest & " best;");
        dmMermaid = arrayToList(dmM, dmNL);
    }
</cfscript>

<cfif dmReady>
<cfoutput>
<link rel="stylesheet" href="#dmgraphPath#dm-graph.css">

<section id="dm-graph" class="dm-graph">
    <h2>The result as a graph</h2>
    <p class="note">
        The same result as RDF triples. The assemblage is a blank node: it has no
        identifier of its own and exists only in this graph. Each result is a
        CIDOC-CRM attribute assignment (E13) that assigns one horizon to the deposition of the
        assemblage. Arrows follow the triples; a label is printed once per group
        of parallel arrows. Concepts (purple) and horizons are those of the
        CAA2026-alligator graphs, so this graph can be merged with them. The
        heavy outline marks the most probable horizon.
    </p>

    <div id="dm-graph-canvas" class="dm-graph__canvas">Drawing the graph &hellip;</div>

    <p class="dm-graph__actions">
        <label>Layout
            <select id="dm-layout">
                <option value="TB">Top to bottom</option>
                <option value="LR">Left to right</option>
                <option value="BT">Bottom to top</option>
                <option value="RL">Right to left</option>
            </select>
        </label>
        <button type="button" id="dm-download-ttl">Download Turtle</button>
        <button type="button" id="dm-download-svg">Download SVG</button>
    </p>

    <details class="dm-graph__source">
        <summary>Show the triples (Turtle)</summary>
        <pre id="dm-turtle-view"></pre>
    </details>

    <script type="text/turtle" id="dm-turtle">#dmTurtle#</script>
    <script type="text/x-mermaid" id="dm-mermaid">#dmMermaid#</script>
</section>

<script src="#dmgraphPath#vendor/mermaid.min.js"></script>
<script src="#dmgraphPath#dm-graph.js"></script>
</cfoutput>
</cfif>
