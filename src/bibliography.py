"""
Generates an APA 7th edition bibliography from library names extracted by
notebook_parser. Looks up each library in Citations/, collects .bib entries,
deduplicates by DOI, and renders via pandoc + the bundled apa.csl.
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
    """Loads library_citations.yml mapping library names to BibTeX keys."""
    yml_path = os.path.join(citations_dir, "library_citations.yml")
    with open(yml_path) as f:
        return yaml.safe_load(f)


def collect_entries(libraries: list[str], citations_dir: str = _CITATIONS_DIR) -> BibliographyData:
    """Collects .bib entries for each library, deduplicating by DOI."""
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
    """Renders BibliographyData as APA 7th edition plain text via pandoc."""
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
    """Main entry point: produces APA bibliography for a list of library names."""
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
