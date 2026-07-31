"""
Software citation lookup plus the generated-cell lifecycle helpers both
workflows share.

Collection is software-specific: it looks up a notebook's imported libraries in
Citations/ (a YAML index plus per-library .bib files) and merges their BibTeX
into one DataFrame, deduped by DOI. Dataset citations are collected elsewhere -
by the data workflow (data_workflow.py), which runs get_bibtex() /
get_publications() in the notebook's live kernel.

Implementation:
    - load_citation_index(): loads library_citations.yml (library name -> its
      inline "paper" BibTeX and/or a "software" .bib file).
    - collect_library_entries(libraries, citation_types): merges the matching
      software BibTeX entries into a DataFrame, deduped by DOI, optionally
      filtered to "paper" and/or "software". This is what the injected software
      cell calls, so the DataFrame is the citation output the user sees.
    - remove_provenance_cells() / remove_legacy_combine_cells(): drop a
      workflow's previously injected cell so a re-run replaces it instead of
      stacking a second, stale one.

Design decisions:
    - There is no rendering layer here. APA rendering was removed along with the
      LLM chain that produced it: citations are surfaced as the injected cell's
      DataFrame output, not as assembled text. The data workflow still accepts
      an `fmt` argument for compatibility, but it is ignored, so no module
      renders citation text in any format.
    - The cell-lifecycle helpers live here rather than in notebook_parser
      because they are written against the frame names this module's output is
      bound to. Relocating them is deferred to the module-rename task.
"""

import os
import sys

import bibtexparser
import pandas as pd
import yaml
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from notebook_parser import PROVENANCE_CELL_MARKER


_STDLIB_MODULES = sys.stdlib_module_names
_CITATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "Citations")
_DATAFRAME_COLUMNS = [
    "library", "citation_type", "key", "title", "author", "year", "doi", "bibtex", "note",
]

_NO_CITATION_NOTE = "No citation found for imported library"


def load_citation_index() -> dict:
    """Loads library_citations.yml mapping library names to BibTeX keys."""
    yml_path = os.path.join(_CITATIONS_DIR, "library_citations.yml")
    with open(yml_path) as f:
        return yaml.safe_load(f)


def is_stdlib(library: str) -> bool:
    """
    Reports whether a library name is part of the Python standard library.

    Standard-library modules ship with the interpreter and nobody cites them, so
    they are dropped from an import list outright rather than being reported as
    libraries whose citation is missing.

    Args:
        library: a top-level module name from parse_notebook()

    Returns:
        True if the name is a standard-library module
    """
    return library.lower() in _STDLIB_MODULES


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
        "note": "",
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
        a DataFrame with one row per citation entry and the uniform citation
        columns (including ``note``); a library with both a paper and a
        software citation produces two rows
    """
    index = load_citation_index()
    seen_dois: set[str] = set()
    rows: list[dict] = []

    for lib in libraries:
        lib_lower = lib.lower()
        if is_stdlib(lib_lower):
            continue
        found_citation = False
        lib_entry = index.get(lib_lower) or {}

        if (not citation_types or "paper" in citation_types) and "paper" in lib_entry:
            entry = bibtexparser.loads(lib_entry["paper"], parser=_bibtex_parser()).entries[0]
            found_citation = True
            _add_entry_row(rows, seen_dois, lib_lower, "paper", entry)

        if not citation_types or "software" in citation_types:
            bib_path = os.path.join(_CITATIONS_DIR, f"{lib_lower}.bib")
            if os.path.exists(bib_path):
                with open(bib_path) as f:
                    entries = bibtexparser.load(f, parser=_bibtex_parser()).entries
                found_citation = bool(entries) or found_citation
                for entry in entries:
                    _add_entry_row(rows, seen_dois, lib_lower, "software", entry)

        if not found_citation:
            rows.append({
                "library": lib_lower,
                "citation_type": None,
                "key": None,
                "title": None,
                "author": None,
                "year": None,
                "doi": None,
                "bibtex": None,
                "note": _NO_CITATION_NOTE,
            })

    return pd.DataFrame(rows, columns=_DATAFRAME_COLUMNS)


_LEGACY_COMBINE_MARKER = "# provenance-combine-cell"

SOFTWARE_FRAME = "provenance_software"
DATASETS_FRAME = "provenance_datasets"


def remove_provenance_cells(nb, frame: str) -> None:
    """
    Drops any previously injected cell that binds the given provenance frame.

    Injection appends, so without this a second run would leave the notebook
    with two software cells (or two dataset cells) that disagree - the stale one
    still displaying whatever the earlier run found. Each workflow calls this
    for its own frame before appending, so re-running is idempotent and the
    notebook holds exactly one cell per workflow.

    Args:
        nb: an nbformat notebook node (modified in place)
        frame: the bound frame name, SOFTWARE_FRAME or DATASETS_FRAME
    """
    def belongs_to_frame(cell) -> bool:
        if cell.cell_type != "code":
            return False
        if f"{frame} = " in cell.source:
            return True
        return (
            frame == DATASETS_FRAME
            and PROVENANCE_CELL_MARKER in cell.source
            and f"{SOFTWARE_FRAME} = " not in cell.source
        )

    nb.cells = [cell for cell in nb.cells if not belongs_to_frame(cell)]


def remove_legacy_combine_cells(nb) -> None:
    """
    Strips the combined-bibliography cell earlier versions used to inject.

    The two workflows now each own one self-displaying cell, so nothing creates
    this cell anymore. Notebooks run against an older version still carry one,
    and since no code manages it now it would linger forever and re-display a
    stale bibliography. Both workflows call this before writing.

    Args:
        nb: an nbformat notebook node (modified in place)
    """
    nb.cells = [
        cell for cell in nb.cells
        if not (cell.cell_type == "code" and _LEGACY_COMBINE_MARKER in cell.source)
    ]
