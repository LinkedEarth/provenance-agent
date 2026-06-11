"""
bibliography.py

Purpose:
    Generates an APA 7th edition bibliography from a list of library names
    (as returned by notebook_parser). Looks up each library in the Citations/
    directory, collects the relevant .bib entries, deduplicates by DOI, and
    renders them through pandoc + apa.csl.

Implementation:
    Uses PyYAML to read library_citations.yml (the library-to-BibTeX-key mapping)
    and pybtex to parse/filter .bib entries. Rendering is delegated to pandoc with
    the bundled apa.csl file via a single subprocess call — this handles all APA
    formatting edge cases (author truncation, italics, entry-type-specific layout)
    without reimplementing a citation style engine in Python.

Design Decisions:
    - pandoc + apa.csl over pybtex-apa7-style or hand-written formatting: pandoc is
      a widely-installed tool with battle-tested CSL support; the apa.csl file is
      bundled in the repo so rendering is fully offline.
    - Deduplication is by DOI: if a library's paper and software citations share the
      same DOI, only one entry is written to the temp .bib file.
    - Citation order follows notebook import order, not APA alphabetical, so the
      bibliography reflects the narrative structure of the notebook.
    - A temporary .bib file is assembled with only the needed entries and passed to
      pandoc, rather than merging all .bib files, so the output is scoped to exactly
      what the notebook imports.
"""

import os
import subprocess
import sys
import tempfile
import yaml
from pybtex.database import parse_file, BibliographyData


_STDLIB_MODULES = sys.stdlib_module_names


_CITATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "Citations")


def load_citation_index(citations_dir: str = _CITATIONS_DIR) -> dict:
    """
    @brief:       Loads library_citations.yml and returns the mapping of library
                  names to their BibTeX citation keys.
    @params[in]:  citations_dir - path to the Citations/ directory
    @params[out]: dict mapping library name (str) to dict of citation type -> key,
                  e.g. {"pandas": {"paper": "mckinney2010data", "software": "pandas_software"}}
    """
    yml_path = os.path.join(citations_dir, "library_citations.yml")
    with open(yml_path) as f:
        return yaml.safe_load(f)


def collect_entries(libraries: list[str], citations_dir: str = _CITATIONS_DIR) -> BibliographyData:
    """
    @brief:       For each library in the ordered input list, loads the matching .bib
                  file and extracts the entries listed in library_citations.yml.
                  Deduplicates entries that share a DOI. Preserves insertion order
                  so the output follows notebook appearance order.
    @params[in]:  libraries     - ordered list of library name strings
                  citations_dir - path to Citations/ directory
    @params[out]: a pybtex BibliographyData containing only the needed entries
    """
    index = load_citation_index(citations_dir)
    seen_dois = set()
    merged = BibliographyData()

    for lib in libraries:
        lib_lower = lib.lower()
        if lib_lower not in index:
            continue

        bib_path = os.path.join(citations_dir, f"{lib_lower}.bib")
        if not os.path.exists(bib_path):
            continue

        bib_data = parse_file(bib_path)
        lib_keys = index[lib_lower]

        for cite_key in lib_keys.values():
            if cite_key not in bib_data.entries:
                continue

            entry = bib_data.entries[cite_key]
            doi = entry.fields.get("doi", "")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)

            merged.entries[cite_key] = entry

    return merged


def render_apa(bib_data: BibliographyData, citations_dir: str = _CITATIONS_DIR) -> str:
    """
    @brief:       Renders a pybtex BibliographyData object as APA 7th edition
                  plain text using pandoc and the bundled apa.csl file.
    @params[in]:  bib_data      - pybtex BibliographyData with entries to render
                  citations_dir - path to Citations/ directory (for locating apa.csl)
    @params[out]: APA-formatted bibliography as a plain text string
    """
    csl_path = os.path.join(citations_dir, "apa.csl")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as tmp:
        tmp_path = tmp.name
        bib_data.to_file(tmp_path, bib_format="bibtex")

    try:
        result = subprocess.run(
            [
                "pandoc",
                "--csl", csl_path,
                "--bibliography", tmp_path,
                "-t", "plain",
                "--citeproc",
            ],
            input='---\nnocite: "@*"\n---\n',
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pandoc failed: {result.stderr}")
        return result.stdout.strip()
    finally:
        os.unlink(tmp_path)


def generate_bibliography(libraries: list[str], citations_dir: str = _CITATIONS_DIR) -> str:
    """
    @brief:       Main entry point. Takes an ordered list of library names (from
                  notebook_parser) and produces an APA-style bibliography string.
                  Includes both paper and software citations for each library.
                  Entries sharing a DOI are deduplicated. Output order matches
                  the input list (notebook order of appearance).
    @params[in]:  libraries     - ordered list of library name strings
                  citations_dir - path to Citations/ directory
    @params[out]: APA-formatted bibliography as a plain text string
    """
    index = load_citation_index(citations_dir)
    entries = collect_entries(libraries, citations_dir)

    not_found = [lib for lib in libraries
                 if lib.lower() not in index and lib not in _STDLIB_MODULES]

    parts = []
    if entries.entries:
        parts.append(render_apa(entries, citations_dir))
    if not_found:
        placeholders = "\n".join(f"[No citation found for: {lib}]" for lib in not_found)
        parts.append(placeholders)

    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from notebook_parser import parse_notebook

    if len(sys.argv) != 2:
        print("Usage: python bibliography.py <path_to_notebook.ipynb>")
        sys.exit(1)

    result = parse_notebook(sys.argv[1])
    print("Libraries found:", result["libraries"])
    print("\n--- Bibliography ---\n")
    print(generate_bibliography(result["libraries"]))
