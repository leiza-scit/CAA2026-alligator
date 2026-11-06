<!--- services_output.cfm
     Dirichlet-multinomial horizon attribution.

     Reference data:
       public.tblservicesquantities
       id, horizon, bordoblique, ia, ib, ic, ii

     The actual archaeological counts are used as Dirichlet parameters.
     A Jeffreys pseudocount of 0.5 is added to each service category to
     handle zero counts without assigning zero probability.

     Equal prior probability is assigned to each horizon.
     Results are displayed in fixed chronological order h1 ... h5.
--->
<cfscript>
    if (structKeyExists(application, "servicesDSN")) {
        servicesDSN = application.servicesDSN;
    } else {
        servicesDSN = "IPS";
    }

    // Jeffreys smoothing: 0.5 added to each of the five service categories.
    pseudocount = 0.5;

    serviceFields = ["bordoblique", "ia", "ib", "ic", "ii"];
    serviceLabels = ["BordOblique", "Ia", "Ib", "Ic", "II"];
    observed = [];
    inputErrors = [];

    for (fieldName in serviceFields) {
        if (!structKeyExists(form, fieldName) OR !reFind("^\d+$", trim(form[fieldName]))) {
            arrayAppend(inputErrors, "Please enter a non-negative whole number for " & fieldName & ".");
            arrayAppend(observed, 0);
        } else {
            arrayAppend(observed, val(form[fieldName]));
        }
    }

    totalObserved = 0;
    for (countValue in observed) {
        totalObserved += countValue;
    }

    if (totalObserved LTE 0) {
        arrayAppend(inputErrors, "The total number of sherds must be greater than zero.");
    }

    // Lanczos approximation for log(Gamma(z)).
    function logGamma(required numeric z) {
        var coefficients = [
            676.5203681218851,
            -1259.1392167224028,
            771.3234287776531,
            -176.6150291621406,
            12.507343278686905,
            -0.13857109526572012,
            9.984369578019572E-6,
            1.5056327351493116E-7
        ];
        var x = 0.99999999999980993;
        var zz = arguments.z - 1;
        var t = 0;
        var i = 0;

        if (arguments.z LT 0.5) {
            return log(pi()) - log(sin(pi() * arguments.z)) - logGamma(1 - arguments.z);
        }

        for (i = 1; i LTE arrayLen(coefficients); i++) {
            x += coefficients[i] / (zz + i);
        }

        t = zz + 7.5;
        return 0.5 * log(2 * pi()) + (zz + 0.5) * log(t) - t + log(x);
    }

    results = [];
    maxLogPosterior = -1E300;
    sumExp = 0;

    if (!arrayLen(inputErrors)) {
        sql = "
            SELECT id, horizon, bordoblique, ia, ib, ic, ii
            FROM public.tblservicesquantities
            ORDER BY CASE horizon
                WHEN 'h1' THEN 1
                WHEN 'h2' THEN 2
                WHEN 'h3' THEN 3
                WHEN 'h4' THEN 4
                WHEN 'h5' THEN 5
                ELSE 99
            END, id
        ";

        qServices = queryExecute(sql, {}, { datasource = servicesDSN });

        if (qServices.recordCount NEQ 5) {
            arrayAppend(inputErrors, "Expected exactly five reference records (h1 to h5) in public.tblservicesquantities; found " & qServices.recordCount & ".");
        } else {
            // Calculate one Dirichlet-multinomial log likelihood per horizon.
            for (r = 1; r LTE qServices.recordCount; r++) {
                referenceCounts = [
                    val(qServices.bordoblique[r]),
                    val(qServices.ia[r]),
                    val(qServices.ib[r]),
                    val(qServices.ic[r]),
                    val(qServices.ii[r])
                ];

                alpha = [];
                alphaTotal = 0;

                for (j = 1; j LTE arrayLen(referenceCounts); j++) {
                    if (referenceCounts[j] LT 0) {
                        arrayAppend(inputErrors, "Negative reference quantity found in horizon " & qServices.horizon[r] & ".");
                    }
                    a = referenceCounts[j] + pseudocount;
                    arrayAppend(alpha, a);
                    alphaTotal += a;
                }

                // Multinomial coefficient is omitted because it is identical
                // for every candidate horizon.
                logLikelihood = logGamma(alphaTotal) - logGamma(alphaTotal + totalObserved);

                for (j = 1; j LTE arrayLen(observed); j++) {
                    logLikelihood += logGamma(alpha[j] + observed[j]) - logGamma(alpha[j]);
                }

                // Equal prior probability for each of the five horizons.
                logPosterior = logLikelihood - log(5);

                arrayAppend(results, {
                    horizon = qServices.horizon[r],
                    referenceTotal = arraySum(referenceCounts),
                    logLikelihood = logLikelihood,
                    logPosterior = logPosterior,
                    probability = 0
                });

                if (logPosterior GT maxLogPosterior) {
                    maxLogPosterior = logPosterior;
                }
            }

            if (!arrayLen(inputErrors)) {
                // Stable log-sum-exp normalization.
                for (item in results) {
                    sumExp += exp(item.logPosterior - maxLogPosterior);
                }

                for (i = 1; i LTE arrayLen(results); i++) {
                    results[i].probability = exp(results[i].logPosterior - maxLogPosterior) / sumExp;
                }
            }
        }
    }
</cfscript>

<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Horizon likelihoods</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 850px; margin: 2.5rem auto; padding: 0 1rem; color: #222; }
        h1 { font-size: 1.55rem; }
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
        th, td { padding: .65rem; border-bottom: 1px solid #ddd; text-align: right; }
        th:first-child, td:first-child { text-align: left; }
        th { background: #f3f3f3; }
        .bar { background: #9e9b9b; height: 12px; }
        .error { color: #9b1c1c; }
        .note { color: #555; font-size: .92rem; line-height: 1.45; }
        a { display: inline-block; margin-top: 1rem; }
    </style>
</head>
<body>
    <h1>Likelihood of the ceramic horizons</h1>

    <cfif arrayLen(inputErrors)>
        <div class="error">
            <cfoutput><cfloop array="#inputErrors#" item="err"><p>#encodeForHTML(err)#</p></cfloop></cfoutput>
        </div>
    <cfelse>
        <h2>Observed assemblage</h2>
        <table>
            <tr><th>Service</th><th>Number of sherds</th></tr>
            <cfoutput>
            <cfloop from="1" to="#arrayLen(serviceLabels)#" index="i">
                <tr><td>#encodeForHTML(serviceLabels[i])#</td><td>#numberFormat(observed[i], "9,999,999")#</td></tr>
            </cfloop>
            <tr><th>Total</th><th>#numberFormat(totalObserved, "9,999,999")#</th></tr>
            </cfoutput>
        </table>

        <h2>Relative posterior probabilities</h2>
        <p class="note">These are normalized posterior probabilities across h1–h5. They sum to 100% and assume equal prior probabilities for the five horizons. The rows are shown in chronological order, not ranked by probability.</p>

        <table>
            <tr><th>Horizon</th><th>Reference sample size</th><th>Probability</th><th>Relative display</th></tr>
            <cfoutput>
            <cfloop array="#results#" item="item">
                <tr>
                    <td><strong>#encodeForHTML(item.horizon)#</strong></td>
                    <td>#numberFormat(item.referenceTotal, "9,999,999")#</td>
                    <td>#numberFormat(item.probability * 100, "99.99")#%</td>
                    <td style="width:35%">
                        <cfif round(item.probability * 10000) GT 0>
                            <div class="bar" style="width:#numberFormat(item.probability * 100, "99.99")#%;"></div>
                        </cfif>
                    </td>
                </tr>
            </cfloop>
            </cfoutput>
        </table>

        <p class="note">
            Statistical model: Dirichlet-multinomial predictive likelihood.
            Each Dirichlet parameter is the corresponding reference count in
            servicesquantities plus a pseudocount of <cfoutput>#pseudocount#</cfoutput>.
            The reference sample size therefore contributes directly to the
            concentration of each horizon's Dirichlet distribution.
        </p>

        <!--- The result as RDF triples, drawn as a graph. Remove this line to switch it off. --->
        <cfinclude template="dmgraph/dm_graph.cfm">
    </cfif>

    <p><a href="services_input.cfm">Enter another assemblage</a></p>
</body>
</html>
