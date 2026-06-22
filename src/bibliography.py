"""
Generates an APA 7th edition bibliography from a parsed notebook result.
Handles two citation categories:
  - Software libraries: looked up in Citations/ YAML and .bib files,
    converted to APA via the LLM
  - Datasets: not fetched here — parse_notebook() returns a datasets list
    with actionable instructions for the PaleoPAL Code Agent (PyLiPD,
    PyleoTUPS) and SPARQL Agent (LiPDGraph) to retrieve citations

All library citations are deduplicated by DOI. The final bibliography
is returned as a JSON-formatted Jupyter markdown cell for PaleoPAL.
"""

import json
import os
import sys
import yaml
from pybtex.database import parse_file, parse_string, BibliographyData


_STDLIB_MODULES = sys.stdlib_module_names
_CITATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "Citations")


def load_citation_index() -> dict:
    """Loads library_citations.yml mapping library names to BibTeX keys."""
    yml_path = os.path.join(_CITATIONS_DIR, "library_citations.yml")
    with open(yml_path) as f:
        return yaml.safe_load(f)


def _add_entries(source: BibliographyData, dest: BibliographyData, seen_dois: set) -> None:
    """Merges entries from source into dest, deduplicating by DOI."""
    for key, entry in source.entries.items():
        doi = entry.fields.get("doi", "")
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        dest.entries[key] = entry


def collect_library_entries(
    libraries: list[str],
    citation_types: list[str] | None = None,
) -> BibliographyData:
    """Collects BibTeX entries for each library, deduplicating by DOI.

    citation_types filters by type (e.g. ["paper"], ["software"]).
    None means all types.
    """
    index = load_citation_index()
    seen_dois = set()
    merged = BibliographyData()

    for lib in libraries:
        lib_lower = lib.lower()
        if lib_lower not in index:
            continue

        lib_entry = index[lib_lower] or {}

        if (not citation_types or "paper" in citation_types) and "paper" in lib_entry:
            paper_bib = parse_string(lib_entry["paper"], bib_format="bibtex")
            _add_entries(paper_bib, merged, seen_dois)

        if not citation_types or "software" in citation_types:
            bib_path = os.path.join(_CITATIONS_DIR, f"{lib_lower}.bib")
            if os.path.exists(bib_path):
                software_bib = parse_file(bib_path)
                _add_entries(software_bib, merged, seen_dois)

    return merged


def render_apa(bib_data: BibliographyData) -> str:
    """
    Converts BibliographyData to APA 7th edition plain text by sending
    each BibTeX entry through the LLM.

    Args:
        bib_data: collected BibTeX entries

    Returns:
        APA-formatted citation string with entries separated by blank lines
    """
    from llm import bibtex_to_apa

    citations = []
    for key, entry in bib_data.entries.items():
        single = BibliographyData(entries={key: entry})
        bibtex_str = single.to_string(bib_format="bibtex").strip()
        apa = bibtex_to_apa(bibtex_str)
        citations.append(apa)

    return "\n\n".join(citations)


def _render_apa_from_strings(bibtex_strings: list[str]) -> str:
    """
    Converts raw BibTeX strings to APA via the LLM.
    Used for dataset citations from PyLiPD and PyleoTUPS.

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


def generate_bibliography(
    parse_result: dict,
    citation_types: list[str] | None = None,
    dataset_bibtex: list[str] | None = None,
) -> str:
    """
    Produces a bibliography string from a parse_notebook() result.
    Collects library citations from local YAML/.bib files and converts
    to APA. Dataset citations are not fetched here — they come from the
    Code Agent or SPARQL Agent and are passed in via dataset_bibtex.

    Args:
        parse_result: dict returned by parse_notebook()
        citation_types: optional filter for library citations —
            "paper" and/or "software"
        dataset_bibtex: optional list of BibTeX strings for dataset
            citations, provided by the Code Agent / SPARQL Agent after
            executing the actions in parse_result["datasets"]

    Returns:
        formatted bibliography string
    """
    parts = []

    # Library citations
    libraries = parse_result["libraries"]
    index = load_citation_index()
    entries = collect_library_entries(libraries, citation_types)

    if entries.entries:
        parts.append(render_apa(entries))

    not_found = [lib for lib in libraries
                 if lib.lower() not in index and lib not in _STDLIB_MODULES]
    if not_found:
        parts.append("\n".join(f"[No citation found for: {lib}]" for lib in not_found))

    # Dataset citations (provided by agents)
    if dataset_bibtex:
        parts.append(_render_apa_from_strings(dataset_bibtex))

    return "\n\n".join(parts)


def generate_bibliography_cell(
    parse_result: dict,
    citation_types: list[str] | None = None,
    dataset_bibtex: list[str] | None = None,
) -> str:
    """
    Produces a JSON-formatted Jupyter markdown cell containing the bibliography.
    This is the final step in the pipeline — the returned JSON string
    can be parsed and appended to a notebook's cell list by PaleoPAL.

    Args:
        parse_result: dict returned by parse_notebook()
        citation_types: optional filter for library citations
        dataset_bibtex: optional list of BibTeX strings for dataset
            citations, provided by the Code Agent / SPARQL Agent

    Returns:
        JSON string representing a notebook markdown cell with the bibliography
    """
    bib_text = generate_bibliography(parse_result, citation_types, dataset_bibtex)
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
