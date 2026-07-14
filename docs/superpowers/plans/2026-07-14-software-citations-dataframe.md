# Software Citations DataFrame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `bibliography.collect_library_entries()` to return a `pandas.DataFrame` (one row per citation entry) instead of a `pybtex.database.BibliographyData`, and update every downstream consumer, so the software workflow's intermediate citation representation is tabular.

**Architecture:** Swap the BibTeX parsing library in `bibliography.py` from `pybtex` to `bibtexparser` (already installed transitively via `pylipd`/`pyleotups`/`doi2bib`), which hands back plain dicts that map directly onto DataFrame columns. `render_apa()` and `generate_bibliography()` are updated to consume the DataFrame; `orchestrator.cite_software()`'s `fmt="bibtex"` branch is updated to join the DataFrame's `bibtex` column instead of calling `BibliographyData.to_string()`.

**Tech Stack:** Python 3.12, `bibtexparser` 1.4.4, `pandas`, `pytest`. Run everything with `/opt/anaconda3/envs/lang/bin/python` (the `lang` conda env - the only one with all required packages).

## Global Constraints

- DataFrame columns, in order: `library, citation_type, key, title, author, year, doi, bibtex`.
- One row per citation entry - a library with both a paper and a software citation produces two rows.
- DOI dedup: skip an entry whose `doi` is non-empty and already seen in an earlier row; entries with an empty `doi` are never deduped against each other. Preserves the existing `_add_entries` behavior in `bibliography.py`.
- `bibtexparser`'s default parser silently drops `@software{...}` entries (logged as "Entry type software not standard. Not considered.") because it's not a "standard" BibTeX type. Every parse call MUST use a `BibTexParser` instance with `ignore_nonstandard_types = False`, or every software citation in `Citations/*.bib` silently disappears.
- No new dependencies - `bibtexparser` and `pandas` are already installed in the `lang` env.

---

### Task 1: `collect_library_entries()` returns a DataFrame

**Files:**
- Modify: `src/bibliography.py:28-88` (imports through `collect_library_entries`)
- Test: `tests/test_bibliography.py` (new file)

**Interfaces:**
- Produces: `collect_library_entries(libraries: list[str], citation_types: list[str] | None = None) -> pd.DataFrame` with columns `["library", "citation_type", "key", "title", "author", "year", "doi", "bibtex"]`.
- Produces (internal helpers later tasks don't need, but Task 2 reads the `bibtex` column): each row's `bibtex` value is a complete, parseable single-entry BibTeX string including the `@type{key, ...}` wrapper.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bibliography.py`:

```python
"""
Unit tests for bibliography.py's collect_library_entries(). Uses real
Citations/ data (pyleoclim has both a paper and a software citation) plus
a monkeypatched citation index for the DOI-dedup case.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bibliography
from bibliography import collect_library_entries

FAKE_INDEX_SHARED_DOI = {
    "liba": {
        "paper": """@article{liba_paper,
  author = {A. One},
  title = {Lib A Paper},
  year = {2020},
  doi = {10.1234/shared}
}"""
    },
    "libb": {
        "paper": """@article{libb_paper,
  author = {B. Two},
  title = {Lib B Paper},
  year = {2021},
  doi = {10.1234/shared}
}"""
    },
}


def test_returns_dataframe_with_expected_columns():
    df = collect_library_entries(["pyleoclim"])
    assert list(df.columns) == [
        "library", "citation_type", "key", "title", "author", "year", "doi", "bibtex",
    ]


def test_library_with_paper_and_software_yields_two_rows():
    df = collect_library_entries(["pyleoclim"])
    assert len(df) == 2
    assert set(df["citation_type"]) == {"paper", "software"}
    assert set(df["key"]) == {"khider2022pyleoclim", "pyleoclim_software"}


def test_citation_types_filter_keeps_only_requested_rows():
    df = collect_library_entries(["pyleoclim"], citation_types=["software"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["citation_type"] == "software"
    assert row["key"] == "pyleoclim_software"


def test_software_entry_bibtex_column_is_parseable_software_type():
    df = collect_library_entries(["pyleoclim"], citation_types=["software"])
    bibtex = df.iloc[0]["bibtex"]
    assert bibtex.startswith("@software{pyleoclim_software,")
    assert "doi" in bibtex


def test_doi_dedup_keeps_first_entry_only(monkeypatch):
    monkeypatch.setattr(bibliography, "load_citation_index", lambda: FAKE_INDEX_SHARED_DOI)
    df = collect_library_entries(["liba", "libb"])
    assert len(df) == 1
    assert df.iloc[0]["key"] == "liba_paper"


def test_unknown_library_yields_no_rows():
    df = collect_library_entries(["definitely_not_a_real_library"])
    assert df.empty
    assert list(df.columns) == [
        "library", "citation_type", "key", "title", "author", "year", "doi", "bibtex",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_bibliography.py -v`
Expected: FAIL (`AttributeError` or `AssertionError` - `collect_library_entries` still returns a `BibliographyData`, which has no `.columns`/`.iloc`).

- [ ] **Step 3: Rewrite `collect_library_entries()` and its helpers**

Replace lines 28-88 of `src/bibliography.py` (the imports block through the end of `collect_library_entries`) with:

```python
import json
import os
import sys

import bibtexparser
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
    parser = _bibtex_parser()
    seen_dois: set[str] = set()
    rows: list[dict] = []

    for lib in libraries:
        lib_lower = lib.lower()
        if lib_lower not in index:
            continue

        lib_entry = index[lib_lower] or {}

        if (not citation_types or "paper" in citation_types) and "paper" in lib_entry:
            entry = bibtexparser.loads(lib_entry["paper"], parser=parser).entries[0]
            _add_entry_row(rows, seen_dois, lib_lower, "paper", entry)

        if not citation_types or "software" in citation_types:
            bib_path = os.path.join(_CITATIONS_DIR, f"{lib_lower}.bib")
            if os.path.exists(bib_path):
                with open(bib_path) as f:
                    entries = bibtexparser.load(f, parser=parser).entries
                for entry in entries:
                    _add_entry_row(rows, seen_dois, lib_lower, "software", entry)

    return pd.DataFrame(rows, columns=_DATAFRAME_COLUMNS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_bibliography.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bibliography.py tests/test_bibliography.py
git commit -m "bibliography: collect_library_entries returns a DataFrame (#26)"
```

---

### Task 2: `render_apa()` consumes the DataFrame

**Files:**
- Modify: `src/bibliography.py:90-110` (the `render_apa` function, now shifted - locate by function name, not line number, since Task 1 changed line counts)
- Test: `tests/test_bibliography.py` (append)

**Interfaces:**
- Consumes: `collect_library_entries(...) -> pd.DataFrame` from Task 1, specifically the `bibtex` column.
- Produces: `render_apa(entries: pd.DataFrame) -> str` (signature changes from `BibliographyData` to `pd.DataFrame`; same return type and join behavior as before).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bibliography.py`:

```python
def test_render_apa_calls_llm_per_row_and_joins(monkeypatch):
    import llm
    monkeypatch.setattr(llm, "bibtex_to_apa", lambda bibtex: f"APA[{bibtex[:20]}]")

    df = collect_library_entries(["pyleoclim"], citation_types=["software"])
    out = bibliography.render_apa(df)
    assert out.startswith("APA[@software{pyleoclim_software")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_bibliography.py::test_render_apa_calls_llm_per_row_and_joins -v`
Expected: FAIL (`render_apa` still does `bib_data.entries.items()`, which raises `AttributeError` on a DataFrame).

- [ ] **Step 3: Rewrite `render_apa()`**

Replace the existing `render_apa` function body in `src/bibliography.py` with:

```python
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
```

Also update the module docstring's description of `render_apa()` (around line 21 of the original file) to say it takes the DataFrame from `collect_library_entries()` rather than a `BibliographyData`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_bibliography.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bibliography.py tests/test_bibliography.py
git commit -m "bibliography: render_apa consumes the citations DataFrame (#26)"
```

---

### Task 3: `generate_bibliography()`'s truthiness check

**Files:**
- Modify: `src/bibliography.py` (the `generate_bibliography` function - locate by function name)
- Test: `tests/test_bibliography.py` (append)

**Interfaces:**
- Consumes: `collect_library_entries(...) -> pd.DataFrame` from Task 1.
- Produces: `generate_bibliography(...)` behavior unchanged from the caller's perspective (same signature, same return type).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bibliography.py`:

```python
def test_generate_bibliography_includes_library_citation(monkeypatch):
    import llm
    monkeypatch.setattr(llm, "bibtex_to_apa", lambda bibtex: "APA_CITATION")

    out = bibliography.generate_bibliography(["pyleoclim"])
    assert "APA_CITATION" in out


def test_generate_bibliography_reports_unknown_library():
    out = bibliography.generate_bibliography(["definitely_not_a_real_library"])
    assert "No citation found for: definitely_not_a_real_library" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_bibliography.py -k generate_bibliography -v`
Expected: FAIL on `test_generate_bibliography_includes_library_citation` - `if entries.entries:` raises `AttributeError` on a DataFrame (DataFrame has no `.entries`).

- [ ] **Step 3: Fix the truthiness check**

In `src/bibliography.py`, inside `generate_bibliography()`, change:

```python
    if entries.entries:
        parts.append(render_apa(entries))
```

to:

```python
    if not entries.empty:
        parts.append(render_apa(entries))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_bibliography.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bibliography.py tests/test_bibliography.py
git commit -m "bibliography: fix generate_bibliography truthiness check for DataFrame (#26)"
```

---

### Task 4: `orchestrator.cite_software()`'s bibtex branch, full regression pass

**Files:**
- Modify: `src/orchestrator.py:70-73`
- Test: run the existing `tests/test_orchestrator.py` (no changes expected, but must be verified)

**Interfaces:**
- Consumes: `collect_library_entries(...) -> pd.DataFrame` from Task 1, specifically the `bibtex` column.

- [ ] **Step 1: Confirm the existing orchestrator tests currently pass on this branch before touching orchestrator.py**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: at this point (Tasks 1-3 done, orchestrator.py untouched), `test_cite_software_all_bibtex_contains_a_library`, `test_cite_software_one_library_bibtex`, `test_cite_software_by_citation_type_software_only`, and `test_cite_software_reports_not_imported_library` FAIL, because `cite_software`'s `fmt="bibtex"` branch still calls `entries.to_string(bib_format="bibtex")`, which doesn't exist on a DataFrame.

- [ ] **Step 2: Fix the bibtex branch**

In `src/orchestrator.py`, inside `cite_software()`, change:

```python
    if fmt == "apa":
        body = render_apa(entries)
    else:
        body = entries.to_string(bib_format="bibtex")
```

to:

```python
    if fmt == "apa":
        body = render_apa(entries)
    else:
        body = "\n\n".join(entries["bibtex"])
```

Also update the module docstring line describing the flow (`parse_notebook -> collect_library_entries -> render_apa (when fmt="apa")`, around line 12) to note that `collect_library_entries` now returns a DataFrame.

- [ ] **Step 3: Run the full test suite**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/ -v`
Expected: all tests PASS, including every test in `tests/test_orchestrator.py` and the new `tests/test_bibliography.py`.

- [ ] **Step 4: Commit**

```bash
git add src/orchestrator.py
git commit -m "orchestrator: join the citations DataFrame's bibtex column for fmt=bibtex (#26)"
```

---

## Post-plan verification

After all four tasks:
- `/opt/anaconda3/envs/lang/bin/python -m pytest tests/ -v` passes in full.
- Manually re-run `notebooks/workflow.ipynb` in the `lang` env (per CLAUDE.md's testing convention - this is the software workflow's dev/demo notebook) to confirm both `fmt="bibtex"` and `fmt="apa"` still produce readable bibliographies end-to-end, not just unit-test substrings.
