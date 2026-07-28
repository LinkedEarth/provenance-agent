"""
Unit tests for bibliography.py's collect_library_entries(). Uses real
Citations/ data (pyleoclim has both a paper and a software citation) plus
a monkeypatched citation index for the DOI-dedup case. Also covers
render_bibtex_strings_to_df as a standalone BibTeX utility. The injected data
workflow uses source metadata directly; the software schema also includes a
note row for imported libraries without citations, except for standard-library
imports, which are dropped entirely.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nbformat

import bibliography
from bibliography import (
    collect_library_entries,
    render_bibtex_strings_to_df,
)

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


def test_render_apa_calls_llm_per_row_and_joins(monkeypatch):
    import llm
    monkeypatch.setattr(llm, "bibtex_to_apa", lambda bibtex: f"APA[{bibtex[:20]}]")

    df = collect_library_entries(["pyleoclim"], citation_types=["software"])
    out = bibliography.render_apa(df)
    assert out.startswith("APA[@software{pyleoclim_")


def test_generate_bibliography_includes_library_citation(monkeypatch):
    import llm
    monkeypatch.setattr(llm, "bibtex_to_apa", lambda bibtex: "APA_CITATION")

    out = bibliography.generate_bibliography(["pyleoclim"])
    assert "APA_CITATION" in out


def test_generate_bibliography_reports_unknown_library():
    out = bibliography.generate_bibliography(["definitely_not_a_real_library"])
    assert "No citation found for: definitely_not_a_real_library" in out


_BIB_A = """@article{smith2020,
  author = {Smith, J.},
  title = {Dataset A},
  year = {2020},
  doi = {10.1000/a}
}"""
_BIB_B = """@article{jones2021,
  author = {Jones, K.},
  title = {Dataset B},
  year = {2021},
  doi = {10.1000/b}
}"""


def test_strings_to_df_has_uniform_schema():
    df = render_bibtex_strings_to_df([_BIB_A], source="D")
    assert list(df.columns) == EXPECTED_COLUMNS
    row = df.iloc[0]
    assert row["library"] == "D"
    assert row["citation_type"] == "dataset"
    assert row["key"] == "smith2020"


def test_strings_to_df_dedupes_by_doi():
    df = render_bibtex_strings_to_df([_BIB_A, _BIB_A], source="D")
    assert len(df) == 1


def test_strings_to_df_multiple_entries():
    df = render_bibtex_strings_to_df([_BIB_A, _BIB_B], source="D")
    assert len(df) == 2


def test_strings_to_df_empty_input_is_empty_framed():
    df = render_bibtex_strings_to_df([], source="D")
    assert df.empty
    assert list(df.columns) == EXPECTED_COLUMNS


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


