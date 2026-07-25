#!/usr/bin/env python3
"""Build the interactive query page from ``queries.yaml``, in three forms.

    docs/sparql.html              generated directly; opens anywhere, no Quarto
    docs/downloads/queries/*.rq   the same queries as plain files
    qmd/<name>.qmd                the quarto-live variant, for reuse as an OER

A query may additionally carry a ``viz:`` list naming short Python files under
``py/viz/``. Those turn that query's rows into a browser figure, and both the
page and the notebook draw them: the figure code is the same file in either
case, so a figure cannot be right in one place and stale in the other.

All three are rendered from the same ``queries.yaml``, so they cannot drift
apart. The HTML page is what the repository publishes and what ``py/main.py``
always builds; the ``.qmd`` is always *written* (so it is versioned and citable)
but only *rendered* if Quarto is installed.

Why generate the HTML directly rather than depend on Quarto: this is a
supplementary-materials repository. Somebody who clones it, or unpacks the
Zenodo archive in five years, should be able to build the site with
``pip install -r requirements.txt`` and nothing else. Quarto stays optional.

All variants run rdflib under Pyodide against a static Turtle file. There is no
SPARQL endpoint and no server to keep alive - deliberate for an archive.

Every query is executed against the real graph here, at build time, before any
page is written. **A query that returns no rows fails the build.** That is not
pedantry: SPARQL does not fail on a mistyped IRI, it returns nothing, so an
empty result is the normal symptom of a broken graph rather than of a boring
question. In this family it has twice been exactly that.

Run standalone (``python py/build_sparql.py``) or as the ``sparql`` step of
``python py/main.py``. Requires ``pyyaml`` and ``jinja2``; the graph must exist.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import textwrap
from pathlib import Path

# --- deviation from the verbatim asset (keep on re-sync) --------------------
# The graph dates events in the first century BC, so it carries xsd:gYear
# literals with negative years ("-0015"). That is valid XSD, but rdflib tries to
# map every literal onto a Python datetime.date, whose minimum year is 1 — the
# conversion raises and rdflib logs a full traceback per literal at WARNING.
# Dozens of tracebacks scroll past on every build and look like a failure, while
# nothing is actually wrong: the literals keep their lexical form, and both the
# queries here and the notebook read them via STR() rather than .value.
# Silencing this one logger keeps real warnings visible.
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wd_paths  # noqa: E402

import yaml  # noqa: E402

QUERIES_YAML = wd_paths.ROOT / "queries.yaml"
META_YAML = wd_paths.ROOT / "metadata.yaml"
QMD_DIR = wd_paths.ROOT / "qmd"
RQ_DIRNAME = "queries"

# Pinned so an archived copy keeps working. An unpinned CDN path follows whatever
# Pyodide ships next, and an rdflib that no longer parses this Turtle would break
# the page silently, years after anyone is watching.
PYODIDE_VERSION = "0.26.4"
RDFLIB_VERSION = "7.1.1"

# How many result rows the browser renders. Some queries are deliberately
# unbounded and the graphs run to tens of thousands of triples; an unlimited
# table can hang a phone.
MAX_ROWS = 500


def _docs_dir():
    """docs/ under whichever name this repo's wd_paths uses."""
    return getattr(wd_paths, "DOCS")


def _downloads_dir():
    return getattr(wd_paths, "DOWNLOADS", _docs_dir() / "downloads")


def load_config():
    if not QUERIES_YAML.exists():
        sys.exit(f"  !!  {QUERIES_YAML.name} is missing.")
    cfg = yaml.safe_load(QUERIES_YAML.read_text(encoding="utf-8")) or {}
    for key in ("graph", "prefixes", "queries"):
        if key not in cfg:
            sys.exit(f"  !!  queries.yaml must contain '{key}'.")
    graph_file = wd_paths.ROOT / cfg["graph"]["file"]
    if not graph_file.exists():
        sys.exit(f"  !!  {graph_file} is missing. Build the graph first.")
    return cfg


def _inline_json(value):
    """JSON for embedding in an inline <script> block.

    The figure code contains HTML, and HTML contains ``</script>``. Inside an
    inline script that sequence ends the script element, wherever it appears -
    including in the middle of a string literal - and the rest of the page's
    JavaScript is then parsed as text. Escaping the slash prevents the match;
    ``\\/`` is a legal JSON escape for ``/`` and reads back as itself.
    """
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def load_viz(cfg):
    """Read the figure code referenced from queries.yaml into the config.

    Deviation from the verbatim asset (keep on re-sync): the shared generator
    has no figure mechanism. The code lives in files under ``py/viz/`` rather
    than inline in the YAML so it stays lintable, diffable and editable with
    syntax highlighting, and it is read here so that a missing or renamed file
    fails the build rather than a reader's browser.

    Returns the number of figures found.
    """
    def read(rel):
        path = wd_paths.ROOT / rel
        if not path.exists():
            sys.exit(f"  !!  queries.yaml refers to {rel}, which is missing.")
        return path.read_text(encoding="utf-8").rstrip("\n")

    qmd_cfg = cfg.setdefault("qmd", {})
    prelude = qmd_cfg.get("viz_prelude")
    qmd_cfg["viz_prelude_code"] = read(prelude) if prelude else ""

    total = 0
    for q in cfg["queries"]:
        for figure in q.get("viz") or []:
            code = read(figure["file"])
            # Both consumers need the value of the last expression. The
            # notebook gets it for free - it is the cell's value - but the HTML
            # page has to evaluate that line on its own, which only works if it
            # is a single line. Checking here means the contract is enforced
            # once, at build time, rather than failing in someone's browser.
            last = code.rstrip().rsplit("\n", 1)[-1]
            if not last.startswith("Frame("):
                sys.exit(f"  !!  {figure['file']} must end in a single-line "
                         f"Frame(...) call; it ends in: {last[:40]!r}")
            figure["code"] = code
            total += 1
    if total:
        print(f"  OK  {total} figure(s) from py/viz/")
    return total


def check_queries(cfg, graph_file):
    """Run every query against the real graph before shipping it.

    A page whose examples do not run is worse than no page: the reader cannot
    tell whether they broke it or it arrived broken. An empty result counts as a
    failure - see the module docstring.
    """
    from rdflib import Graph

    base = Graph().parse(graph_file, format="turtle")
    print(f"  ..  {graph_file.name}: {len(base)} triples")

    extra_graphs = {}
    ok = True
    for q in cfg["queries"]:
        target = base
        key = q.get("needs")
        if key:
            if key not in extra_graphs:
                url = cfg["graph"]["extra"][key]["url"]
                try:
                    import urllib.request
                    with urllib.request.urlopen(url, timeout=30) as response:
                        loaded = Graph()
                        loaded.parse(data=response.read().decode("utf-8"),
                                     format="turtle")
                    extra_graphs[key] = loaded
                except Exception as exc:               # noqa: BLE001
                    print(f"  ..  {q['id']}: could not fetch '{key}' to verify "
                          f"({type(exc).__name__}); shipped unverified.")
                    extra_graphs[key] = None
            if extra_graphs[key] is not None:
                target = base + extra_graphs[key]
        try:
            rows = list(target.query(cfg["prefixes"] + "\n" + q["sparql"]))
        except Exception as exc:                       # noqa: BLE001
            print(f"  !!  query {q['id']}: {type(exc).__name__}: {exc}")
            ok = False
            continue
        q["rows_at_build"] = len(rows)
        if rows:
            print(f"  OK  query {q['id']:24s} {len(rows):4d} rows")
        else:
            print(f"  !!  query {q['id']:24s}    0 rows - parses, matches nothing")
            ok = False
    if not ok:
        sys.exit("  !!  a query does not work; nothing written.")


def write_rq_files(cfg):
    """Write each query as a plain .rq file, for use outside the browser."""
    out_dir = _downloads_dir() / RQ_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.rq"):                 # drop renamed leftovers
        stale.unlink()
    for q in cfg["queries"]:
        intro = "\n".join(f"# {line}" for line in
                          textwrap.wrap(" ".join(str(q.get("intro", "")).split()), 76))
        text = (f"# {q['title']}\n{intro}\n\n"
                f"{cfg['prefixes'].rstrip()}\n\n{q['sparql'].rstrip()}\n")
        (out_dir / f"{q['id']}.rq").write_text(text, encoding="utf-8")
    print(f"  OK  {out_dir}  ({len(cfg['queries'])} .rq file(s))")


def build():
    wd_paths.ensure_dirs()
    cfg = load_config()
    graph_cfg = dict(cfg["graph"])
    graph_file = wd_paths.ROOT / graph_cfg["file"]

    repo = {}
    if META_YAML.exists():
        repo = yaml.safe_load(META_YAML.read_text(encoding="utf-8")).get("repo", {})

    check_queries(cfg, graph_file)
    n_figures = load_viz(cfg)
    viz_prelude = cfg["qmd"]["viz_prelude_code"]
    write_rq_files(cfg)

    docs = _docs_dir()

    # The browser fetches the graph relative to the page. Repos whose graph is
    # not already inside docs/ publish a copy here; the rest just point at it.
    if graph_cfg.get("publish"):
        target = docs / graph_cfg["url"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(graph_file, target)
        print(f"  OK  docs/{graph_cfg['url']}  "
              f"({target.stat().st_size / 1e6:.1f} MB)")
    graph_cfg["megabytes"] = f"{graph_file.stat().st_size / 1e6:.1f}"

    queries = []
    for q in cfg["queries"]:
        item = dict(q)
        item["sparql"] = q["sparql"].rstrip("\n")
        # Size the editor to the query, so nothing hides behind a scrollbar the
        # reader has to discover.
        item["rows"] = max(6, item["sparql"].count("\n") + 2)
        queries.append(item)

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(wd_paths.TEMPLATES)),
                      autoescape=False)

    page = cfg.get("page", {})
    html = env.get_template("sparql.html.j2").render(
        repo=repo, page=page, graph=graph_cfg, queries=queries,
        pyodide_version=PYODIDE_VERSION, rdflib_version=RDFLIB_VERSION,
        max_rows=MAX_ROWS,
        prefixes_json=json.dumps(cfg["prefixes"]),
        viz_prelude_json=_inline_json(viz_prelude if n_figures else ""),
        figures_json=_inline_json(
            {q["id"]: [{"title": f["title"], "code": f["code"]}
                       for f in q.get("viz") or []]
             for q in queries if q.get("viz")}),
        graph_json=json.dumps(graph_cfg, ensure_ascii=False),
        queries_json=json.dumps({q["id"]: q["sparql"] for q in queries},
                                ensure_ascii=False))
    (docs / "sparql.html").write_text(html, encoding="utf-8")
    print(f"  OK  docs/sparql.html  ({len(queries)} queries, "
          f"{n_figures} figure(s))")

    qmd_cfg = dict(cfg.get("qmd", {}))
    if qmd_cfg.get("file"):
        QMD_DIR.mkdir(exist_ok=True)
        qmd_cfg.setdefault("title", page.get("title", "Querying the graph"))
        qmd_cfg["graph_url"] = qmd_cfg.get("graph_url") or graph_cfg["url"]
        qmd_cfg["megabytes"] = graph_cfg["megabytes"]
        qmd = env.get_template("sparql.qmd.j2").render(
            graph=graph_cfg, queries=queries, qmd=qmd_cfg,
            rdflib_version=RDFLIB_VERSION, has_viz=bool(n_figures),
            prefixes=cfg["prefixes"].rstrip("\n"))
        (QMD_DIR / qmd_cfg["file"]).write_text(qmd, encoding="utf-8")
        print(f"  OK  qmd/{qmd_cfg['file']}  ({len(queries)} queries, "
              f"{n_figures} figure(s); render with Quarto, optional)")


def main():
    print("=" * 60)
    print("Build the interactive query page (queries.yaml -> docs/ + qmd/)")
    print("=" * 60)
    for module in ("jinja2", "rdflib"):
        try:
            __import__(module)
        except Exception as exc:                       # noqa: BLE001
            sys.exit(f"  !!  {module} is required: {type(exc).__name__}: {exc}")
    build()
    print("=" * 60 + "\nDone.\n" + "=" * 60)


if __name__ == "__main__":
    main()
