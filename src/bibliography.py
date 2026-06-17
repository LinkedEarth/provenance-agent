"""
Generates an APA 7th edition bibliography from library names extracted by
notebook_parser. Looks up each library in Citations/, collects .bib entries,
deduplicates by DOI, and renders via pandoc + the bundled apa.csl.
"""

import os
import sys
import yaml
import nbformat
from pybtex.database import parse_file, parse_string, BibliographyData


_STDLIB_MODULES = sys.stdlib_module_names
_CITATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "Citations")


def load_citation_index() -> dict:
    """Loads library_citations.yml mapping library names to BibTeX keys."""
    yml_path = os.path.join(_CITATIONS_DIR, "library_citations.yml")
    with open(yml_path) as f:
        return yaml.safe_load(f)


def _add_entries(source: BibliographyData, dest: BibliographyData, seen_dois: set) -> None:
    """Merges entries from source into dest, deduplicating by DOI.

    Deduplication is not needed with the current hardcoded citations,
    but will matter when citations are fetched from the web where
    duplicate DOIs could appear.
    """
    for key, entry in source.entries.items():
        doi = entry.fields.get("doi", "")
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        dest.entries[key] = entry


def collect_entries(
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
    Converts BibliographyData to APA 7th edition plain text.

    Currently returns raw BibTeX as a placeholder. This will be replaced
    by an LLM call that converts BibTeX to APA format.

    Args:
        bib_data: collected BibTeX entries from collect_entries()

    Returns:
        formatted citation string (currently raw BibTeX)
    """
    # TODO: replace with LLM-based BibTeX-to-APA conversion
    return bib_data.to_string(bib_format="bibtex").strip()


def generate_bibliography(
    libraries: list[str],
    citation_types: list[str] | None = None,
) -> str:
    """
    Produces a bibliography string for a list of library names.

    Args:
        libraries: list of library name strings
        citation_types: optional filter — "paper" and/or "software"

    Returns:
        formatted bibliography string with citations and placeholders
        for libraries not found in the citation index
    """
    index = load_citation_index()
    entries = collect_entries(libraries, citation_types)

    not_found = [lib for lib in libraries
                 if lib.lower() not in index and lib not in _STDLIB_MODULES]

    parts = []
    if entries.entries:
        parts.append(render_apa(entries))
    if not_found:
        placeholders = "\n".join(f"[No citation found for: {lib}]" for lib in not_found)
        parts.append(placeholders)

    return "\n\n".join(parts)


def add_bibliography_to_notebook(bib_text: str, notebook_path: str | None = None) -> None:
    """
    Appends a bibliography as a new markdown cell at the end of a notebook.

    If no path is given, attempts to auto-detect the current notebook
    path using ipynbname (must be called from inside a running notebook).

    Note: this only writes to disk via nbformat. The cell won't appear
    in an already-open VS Code/Jupyter session until the notebook is
    reloaded. A future improvement is to also use
    IPython.get_ipython().set_next_input() to inject the cell into the
    live UI, then combine both disk write and UI update in one call.
    VS Code may show a file conflict warning on save — select "Overwrite"
    since the editor state will already match the disk.

    Args:
        bib_text: pre-generated bibliography string to insert
        notebook_path: file path to a .ipynb notebook, or None for
            auto-detection

    Returns:
        None (modifies the notebook file in place)
    """
    if notebook_path is None:
        try:
            import ipynbname
            notebook_path = str(ipynbname.path())
        except Exception:
            raise RuntimeError(
                "Could not auto-detect the current notebook path. "
                "Install ipynbname (`pip install ipynbname`) and call from inside a running notebook, "
                "or pass an explicit notebook_path."
            )

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

    sys.path.insert(0, os.path.dirname(__file__))
    from notebook_parser import parse_notebook

    notebook_path = sys.argv[1]
    result = parse_notebook(notebook_path)
    bib_text = generate_bibliography(result["libraries"])
    add_bibliography_to_notebook(bib_text, notebook_path)
    print(f"Bibliography appended to {notebook_path}")
