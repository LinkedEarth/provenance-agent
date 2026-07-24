"""
Bibliography assembly and APA rendering - the output stage where the software
and data workflows converge into one bibliography.

Two layers, and only one of them is software-specific:
  - Collection (software-specific here): looks up a notebook's imported
    libraries in Citations/ (a YAML index plus per-library .bib files) and
    merges their BibTeX, deduped by DOI. Dataset citations are collected
    elsewhere - by the data workflow (data_workflow.py), which runs
    get_bibtex()/get_publications() in the notebook's live kernel.
  - Rendering/assembly (shared by both workflows): turns BibTeX into APA via
    the LLM and assembles the final bibliography. Dataset BibTeX from the data
    workflow is folded in as strings via the dataset_bibtex argument.

Implementation:
    - load_citation_index(): loads library_citations.yml (library name -> its
      inline "paper" BibTeX and/or a "software" .bib file).
    - collect_library_entries(libraries, citation_types): merges the matching
      software BibTeX entries into a DataFrame, deduped by DOI, optionally
      filtered to "paper" and/or "software".
    - render_apa(entries): takes the DataFrame from collect_library_entries()
      and converts each entry's BibTeX to APA via the LLM.
    - render_bibtex_strings_to_apa(): turns raw BibTeX strings (e.g. dataset
      citations) into APA text via the LLM.
    - generate_bibliography() / generate_bibliography_cell(): assemble the final
      bibliography (libraries + optional dataset BibTeX) as text or as a JSON
      notebook markdown cell.
"""

import json
import os
import sys

import bibtexparser
import nbformat
import pandas as pd
import yaml
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter


_STDLIB_MODULES = sys.stdlib_module_names
_CITATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "Citations")
_DATAFRAME_COLUMNS = [
    "library", "citation_type", "key", "title", "author", "year", "doi", "bibtex",
]


def load_citation_index() -> dict:
    """Loads library_citations.yml mapping library names to BibTeX keys."""
    yml_path = os.path.join(_CITATIONS_DIR, "library_citations.yml")
    with open(yml_path) as f:
        return yaml.safe_load(f)


def _bibtex_parser() -> BibTexParser:
    """
    Returns a BibTexParser configured to keep @software entries.

    bibtexparser's default parser silently drops non-"standard" BibTeX entry
    types (e.g. @software, used by every Zenodo software citation in
    Citations/*.bib) unless ignore_nonstandard_types is disabled.
    """
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    return parser


def _entry_to_bibtex(entry: dict) -> str:
    """Re-serializes a single bibtexparser entry dict back to BibTeX text."""
    db = BibDatabase()
    db.entries = [entry]
    return BibTexWriter().write(db).strip()


def _add_entry_row(
    rows: list[dict],
    seen_dois: set[str],
    library: str,
    citation_type: str,
    entry: dict,
) -> None:
    """
    Appends one DataFrame row for entry, deduplicating by DOI.

    Entries sharing a DOI with an already-added row are skipped; entries
    without a DOI are never deduplicated against each other.
    """
    doi = entry.get("doi", "")
    if doi and doi in seen_dois:
        return
    if doi:
        seen_dois.add(doi)
    rows.append({
        "library": library,
        "citation_type": citation_type,
        "key": entry.get("ID", ""),
        "title": entry.get("title", ""),
        "author": entry.get("author", ""),
        "year": entry.get("year", ""),
        "doi": doi,
        "bibtex": _entry_to_bibtex(entry),
    })


def collect_library_entries(
    libraries: list[str],
    citation_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collects citation entries for each library, deduplicating by DOI.

    Args:
        libraries: library names to look up in library_citations.yml
        citation_types: optional filter - "paper" and/or "software"; None
            means both

    Returns:
        a DataFrame with one row per citation entry (columns: library,
        citation_type, key, title, author, year, doi, bibtex); a library
        with both a paper and a software citation produces two rows
    """
    index = load_citation_index()
    seen_dois: set[str] = set()
    rows: list[dict] = []

    for lib in libraries:
        lib_lower = lib.lower()
        if lib_lower not in index:
            continue

        lib_entry = index[lib_lower] or {}

        if (not citation_types or "paper" in citation_types) and "paper" in lib_entry:
            entry = bibtexparser.loads(lib_entry["paper"], parser=_bibtex_parser()).entries[0]
            _add_entry_row(rows, seen_dois, lib_lower, "paper", entry)

        if not citation_types or "software" in citation_types:
            bib_path = os.path.join(_CITATIONS_DIR, f"{lib_lower}.bib")
            if os.path.exists(bib_path):
                with open(bib_path) as f:
                    entries = bibtexparser.load(f, parser=_bibtex_parser()).entries
                for entry in entries:
                    _add_entry_row(rows, seen_dois, lib_lower, "software", entry)

    return pd.DataFrame(rows, columns=_DATAFRAME_COLUMNS)


def render_apa(entries: pd.DataFrame) -> str:
    """
    Converts collected citation entries to APA 7th edition plain text by
    sending each entry's BibTeX through the LLM.

    Args:
        entries: DataFrame from collect_library_entries(), must have a
            "bibtex" column

    Returns:
        APA-formatted citation string with entries separated by blank lines
    """
    from llm import bibtex_to_apa

    citations = [bibtex_to_apa(bibtex) for bibtex in entries["bibtex"]]
    return "\n\n".join(citations)


def render_bibtex_strings_to_apa(bibtex_strings: list[str]) -> str:
    """
    Converts raw BibTeX strings to APA via the LLM.
    Used for dataset citations produced by the data workflow (data_workflow.py),
    which returns BibTeX as strings from get_bibtex() / get_publications().

    Args:
        bibtex_strings: list of BibTeX entry strings

    Returns:
        APA-formatted citation string with entries separated by blank lines
    """
    from llm import bibtex_to_apa

    citations = []
    for bibtex_str in bibtex_strings:
        apa = bibtex_to_apa(bibtex_str.strip())
        citations.append(apa)

    return "\n\n".join(citations)


def render_bibtex_strings_to_df(bibtex_strings: list[str], source: str) -> pd.DataFrame:
    """
    Parses raw BibTeX strings into the uniform citation DataFrame.

    The DataFrame twin of render_bibtex_strings_to_apa: where that renders
    dataset BibTeX to APA text, this renders it to the same 8-column schema
    collect_library_entries produces, so software and dataset citations share
    one shape and can be concatenated into the combined bibliography.

    Args:
        bibtex_strings: raw BibTeX entry strings (e.g. the data workflow's
            _bib_{var} list from get_bibtex()/get_publications())
        source: label written to the "library" column (the notebook variable
            the datasets came from)

    Returns:
        a DataFrame with columns library, citation_type, key, title, author,
        year, doi, bibtex; citation_type is "dataset"; rows are deduped by DOI
    """
    seen_dois: set[str] = set()
    rows: list[dict] = []
    for bibtex_str in bibtex_strings:
        for entry in bibtexparser.loads(bibtex_str, parser=_bibtex_parser()).entries:
            _add_entry_row(rows, seen_dois, source, "dataset", entry)
    return pd.DataFrame(rows, columns=_DATAFRAME_COLUMNS)


_COMBINE_MARKER = "# provenance-combine-cell"


def build_combine_cell() -> str:
    """
    Builds the source for the combined-bibliography cell.

    The cell scans the kernel namespace at run time for every _provbib_*
    DataFrame (software binds _provbib_software; each data cell binds
    _provbib_data_{var}) and concatenates them into provenance_bibliography.
    Scanning at run time is what lets any subset resolve: software-only,
    data-only, or both all produce the right combined frame without the
    injector needing to know which segments ran.

    Returns:
        Python source that builds and display()s provenance_bibliography
    """
    columns = "['library', 'citation_type', 'key', 'title', 'author', 'year', 'doi', 'bibtex']"
    return (
        f"{_COMBINE_MARKER}\n"
        "import pandas as pd\n"
        "_frames = [v for k, v in sorted(globals().items())\n"
        "           if k.startswith('_provbib_') and isinstance(v, pd.DataFrame)]\n"
        "provenance_bibliography = (pd.concat(_frames, ignore_index=True)\n"
        f"                          if _frames else pd.DataFrame(columns={columns}))\n"
        "display(provenance_bibliography)"
    )


def ensure_combine_cell(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """
    Guarantees exactly one combine cell, positioned as the notebook's last cell.

    Removes any existing marked combine cell, then appends a fresh one. Both
    workflows call this after appending their own cells, so no matter which
    workflow(s) ran or in what order, the notebook ends with a single combine
    cell that runs after every segment cell.

    Args:
        nb: an nbformat notebook node (modified in place)

    Returns:
        the same notebook node, ending with exactly one combine cell
    """
    nb.cells = [
        c for c in nb.cells
        if not (c.cell_type == "code" and _COMBINE_MARKER in c.source)
    ]
    nb.cells.append(nbformat.v4.new_code_cell(build_combine_cell()))
    return nb


def generate_bibliography(
    libraries: list[str],
    citation_types: list[str] | None = None,
    dataset_bibtex: list[str] | None = None,
) -> str:
    """
    Produces a bibliography string for a notebook's imported libraries.
    Collects library citations from local YAML/.bib files and renders them
    to APA. Dataset citations are not collected here — the data workflow
    produces them and they can be folded in via dataset_bibtex.

    Args:
        libraries: imported library names (from parse_notebook())
        citation_types: optional filter for library citations —
            "paper" and/or "software"
        dataset_bibtex: optional list of BibTeX strings for dataset
            citations, produced by the data workflow (data_workflow.py)

    Returns:
        formatted bibliography string
    """
    parts = []

    # Library citations
    index = load_citation_index()
    entries = collect_library_entries(libraries, citation_types)

    if not entries.empty:
        parts.append(render_apa(entries))

    not_found = [lib for lib in libraries
                 if lib.lower() not in index and lib not in _STDLIB_MODULES]
    if not_found:
        parts.append("\n".join(f"[No citation found for: {lib}]" for lib in not_found))

    # Dataset citations (produced by the data workflow)
    if dataset_bibtex:
        parts.append(render_bibtex_strings_to_apa(dataset_bibtex))

    return "\n\n".join(parts)


def generate_bibliography_cell(
    libraries: list[str],
    citation_types: list[str] | None = None,
    dataset_bibtex: list[str] | None = None,
) -> str:
    """
    Produces a JSON-formatted Jupyter markdown cell containing the bibliography.
    The returned JSON string can be parsed and appended to a notebook's cell
    list.

    Args:
        libraries: imported library names (from parse_notebook())
        citation_types: optional filter for library citations
        dataset_bibtex: optional list of BibTeX strings for dataset
            citations, produced by the data workflow (data_workflow.py)

    Returns:
        JSON string representing a notebook markdown cell with the bibliography
    """
    bib_text = generate_bibliography(libraries, citation_types, dataset_bibtex)
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## Bibliography\n\n{bib_text}"],
    }
    return json.dumps(cell, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bibliography.py <path_to_notebook.ipynb>")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(__file__))
    from notebook_parser import parse_notebook

    notebook_path = sys.argv[1]
    result = parse_notebook(notebook_path)
    bib_text = generate_bibliography(result)
    print(bib_text)
