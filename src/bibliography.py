"""
Generates an APA 7th edition bibliography for libraries found in a Jupyter
Notebook.

Citation data is stored in Citations/library_citations.yml, a nested dict
(loaded via PyYAML) that maps each library name to an optional inlined
paper BibTeX entry and an optional software .bib filename. Paper DOIs are
extracted from the inlined BibTeX using pybtex; software DOIs are read
from the referenced .bib files, also via pybtex.

Once DOIs are collected, each is sent to doi.org with an Accept header
requesting APA-formatted plain text (content negotiation). This avoids
storing pre-formatted citations and always returns the canonical APA
string from the DOI registrar. The trade-off is that an HTTP request is
made per DOI, so bibliography generation requires network access.

The main entry point is add_bibliography_to_notebook(), which chains
notebook_parser.parse_notebook() -> collect_dois() -> doi_to_apa() and
appends the result as a markdown cell to the notebook via nbformat.
"""

import os
import re
import sys
import yaml
import requests
import nbformat
from pybtex.database import parse_file, parse_string


_STDLIB_MODULES = sys.stdlib_module_names
_CITATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "Citations")


def load_citation_index(citations_dir: str = _CITATIONS_DIR) -> dict:
    """
    Loads library_citations.yml and returns it as a nested dict mapping
    library names to their citation entries (paper BibTeX and/or software
    .bib filename).

    Args:
        citations_dir: path to the Citations/ directory

    Returns:
        dict mapping library name strings to dicts with optional "paper"
        and "software" keys
    """
    yml_path = os.path.join(citations_dir, "library_citations.yml")
    with open(yml_path) as f:
        return yaml.safe_load(f)


def _extract_doi_from_bibtex(bibtex_str: str) -> str | None:
    """
    Extracts the DOI field from a BibTeX string. Parses the BibTeX with
    pybtex and returns the first DOI found across all entries.

    Args:
        bibtex_str: raw BibTeX string (e.g. from the "paper" field in the YAML)

    Returns:
        DOI string, or None if no DOI field is present
    """
    bib = parse_string(bibtex_str, bib_format="bibtex")
    for entry in bib.entries.values():
        doi = entry.fields.get("doi", "")
        if doi:
            return doi
    return None


def doi_to_apa(doi: str) -> str | None:
    """
    Fetches an APA 7th edition formatted citation from doi.org using
    content negotiation. Sends a GET request with an Accept header for
    APA-style plain text.

    Args:
        doi: DOI string (e.g. "10.1038/s41586-020-2649-2")

    Returns:
        APA citation string, or None if the request fails
    """
    r = requests.get(
        f"https://doi.org/{doi}",
        headers={"Accept": "text/x-bibliography; style=apa"},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    r.encoding = "utf-8"
    text = r.text.strip()
    # doi.org sometimes returns HTML italic tags for software titles
    text = re.sub(r"</?i>", "", text)
    return text


def collect_dois(
    libraries: list[str],
    citations_dir: str = _CITATIONS_DIR,
    citation_types: list[str] | None = None,
) -> list[str]:
    """
    Collects unique DOIs for a list of libraries by looking up each in
    library_citations.yml. Paper DOIs are extracted from inlined BibTeX;
    software DOIs are read from the .bib file referenced in the YAML.

    Args:
        libraries: list of library name strings (e.g. from notebook_parser)
        citations_dir: path to the Citations/ directory
        citation_types: optional filter — list containing "paper" and/or
            "software". If None, both types are collected.

    Returns:
        list of unique DOI strings, in the order they were encountered
    """
    index = load_citation_index(citations_dir)
    seen = set()
    dois = []

    for lib in libraries:
        lib_lower = lib.lower()
        if lib_lower not in index:
            continue

        lib_entry = index[lib_lower] or {}

        if (not citation_types or "paper" in citation_types) and "paper" in lib_entry:
            doi = _extract_doi_from_bibtex(lib_entry["paper"])
            if doi and doi not in seen:
                seen.add(doi)
                dois.append(doi)

        if not citation_types or "software" in citation_types:
            software_file = lib_entry.get("software")
            if software_file:
                bib_path = os.path.join(citations_dir, software_file)
                if os.path.exists(bib_path):
                    bib_data = parse_file(bib_path)
                    for entry in bib_data.entries.values():
                        doi = entry.fields.get("doi", "")
                        if doi and doi not in seen:
                            seen.add(doi)
                            dois.append(doi)

    return dois


def generate_bibliography(
    libraries: list[str],
    citations_dir: str = _CITATIONS_DIR,
    citation_types: list[str] | None = None,
) -> str:
    """
    Produces a full APA bibliography string for a list of library names.
    Collects DOIs, fetches APA citations from doi.org, and appends
    placeholder lines for any libraries not found in the citation index.

    Args:
        libraries: list of library name strings
        citations_dir: path to the Citations/ directory
        citation_types: optional filter — "paper" and/or "software"

    Returns:
        formatted bibliography string with citations separated by blank
        lines, followed by any "[No citation found]" placeholders
    """
    index = load_citation_index(citations_dir)
    dois = collect_dois(libraries, citations_dir, citation_types)

    citations = []
    for doi in dois:
        apa = doi_to_apa(doi)
        if apa:
            citations.append(apa)

    not_found = [lib for lib in libraries
                 if lib.lower() not in index and lib not in _STDLIB_MODULES]

    parts = []
    if citations:
        parts.append("\n\n".join(citations))
    if not_found:
        placeholders = "\n".join(f"[No citation found for: {lib}]" for lib in not_found)
        parts.append(placeholders)

    return "\n\n".join(parts)


def add_bibliography_to_notebook(notebook_path: str, citations_dir: str = _CITATIONS_DIR) -> None:
    """
    End-to-end entry point. Parses a notebook for library imports,
    generates an APA bibliography, and appends it as a new markdown
    cell at the end of the notebook.

    Args:
        notebook_path: file path to a .ipynb notebook
        citations_dir: path to the Citations/ directory

    Returns:
        None (modifies the notebook file in place)
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from notebook_parser import parse_notebook

    result = parse_notebook(notebook_path)
    bib_text = generate_bibliography(result["libraries"], citations_dir)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    cell_source = "## Bibliography\n\n" + bib_text
    new_cell = nbformat.v4.new_markdown_cell(source=cell_source)
    nb.cells.append(new_cell)

    with open(notebook_path, "w") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bibliography.py <path_to_notebook.ipynb>")
        sys.exit(1)

    add_bibliography_to_notebook(sys.argv[1])
    print(f"Bibliography appended to {sys.argv[1]}")
