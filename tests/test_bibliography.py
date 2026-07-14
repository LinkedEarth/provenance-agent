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
