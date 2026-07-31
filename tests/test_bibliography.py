"""
Unit tests for bibliography.py's collect_library_entries(). Uses real
Citations/ data (pyleoclim has both a paper and a software citation) plus
a monkeypatched citation index for the DOI-dedup case. The injected data
workflow uses source metadata directly; the software schema also includes a
note row for imported libraries without citations, except for standard-library
imports, which are dropped entirely.

The APA rendering and bibliography-assembly functions were removed along with
the LLM chain behind them, so this module no longer covers render_apa,
render_bibtex_strings_to_apa, render_bibtex_strings_to_df, or
generate_bibliography.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nbformat

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

EXPECTED_COLUMNS = [
    "library", "citation_type", "key", "title", "author", "year", "doi",
    "bibtex", "note",
]


def test_returns_dataframe_with_expected_columns():
    df = collect_library_entries(["pyleoclim"])
    assert list(df.columns) == EXPECTED_COLUMNS


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


def test_unknown_library_yields_note_row():
    import pandas as pd

    df = collect_library_entries(["definitely_not_a_real_library"])
    assert len(df) == 1
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.iloc[0]["library"] == "definitely_not_a_real_library"
    assert df.iloc[0]["note"] == "No citation found for imported library"
    assert pd.isna(df.iloc[0]["bibtex"])


def test_multiple_libraries_with_distinct_papers_are_not_corrupted():
    df = collect_library_entries(
        ["pandas", "numpy", "pyleoclim"], citation_types=["paper"]
    )
    assert len(df) == 3
    by_library = df.set_index("library")
    assert by_library.loc["pandas", "key"] == "mckinney2010data"
    assert by_library.loc["pandas", "doi"] == "10.25080/Majora-92bf1922-00a"
    assert by_library.loc["numpy", "key"] == "harris2020array"
    assert by_library.loc["numpy", "doi"] == "10.1038/s41586-020-2649-2"
    assert by_library.loc["pyleoclim", "key"] == "khider2022pyleoclim"
    assert by_library.loc["pyleoclim", "doi"] == "10.1029/2022PA004509"


# --- APA rendering is gone ---------------------------------------------------

def test_apa_rendering_surface_is_removed():
    """The LLM rendering path is deleted, not merely unused."""
    for name in (
        "render_apa",
        "render_bibtex_strings_to_apa",
        "render_bibtex_strings_to_df",
        "generate_bibliography",
        "generate_bibliography_cell",
    ):
        assert not hasattr(bibliography, name)

    import llm

    assert not hasattr(llm, "bibtex_to_apa")


# --- standard-library imports ------------------------------------------------

def test_is_stdlib_recognizes_interpreter_modules():
    assert bibliography.is_stdlib("sys")
    assert bibliography.is_stdlib("json")
    assert bibliography.is_stdlib("PATHLIB")  # case-insensitive
    assert not bibliography.is_stdlib("numpy")


def test_stdlib_produces_no_rows():
    df = collect_library_entries(["sys", "os", "json", "io", "ast", "pathlib"])
    assert df.empty
    assert list(df.columns) == EXPECTED_COLUMNS


def test_stdlib_dropped_but_uncited_third_party_still_noted():
    df = collect_library_entries(["sys", "definitely_not_a_real_library"])
    assert list(df["library"]) == ["definitely_not_a_real_library"]
    assert df.iloc[0]["note"] == "No citation found for imported library"


def test_stdlib_does_not_crowd_out_real_citations():
    df = collect_library_entries(["sys", "pyleoclim", "pathlib"])
    assert set(df["library"]) == {"pyleoclim"}


