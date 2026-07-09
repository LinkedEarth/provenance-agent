"""
Unit tests for orchestrator.py. Offline where possible: cite_software's bibtex
path reads only local Citations/, and the apa path is exercised by monkeypatching
bibliography.render_apa so no Gemini call is made.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator import _check_fmt, cite_software

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sample.ipynb")


def _write_lipdgraph_notebook(path):
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
        "filtered_df2 = None"
    ))
    with open(path, "w") as f:
        nbformat.write(nb, f)


def test_check_fmt_rejects_unknown():
    with pytest.raises(ValueError):
        _check_fmt("markdown")


def test_cite_software_all_bibtex_contains_a_library():
    out = cite_software(SAMPLE, fmt="bibtex")
    assert "pyleoclim" in out.lower()


def test_cite_software_one_library_bibtex():
    out = cite_software(SAMPLE, libraries="pyleoclim", fmt="bibtex")
    assert "pyleoclim" in out.lower()
    assert "numpy" not in out.lower()


def test_cite_software_by_citation_type_software_only():
    out = cite_software(SAMPLE, libraries="pyleoclim", citation_types=["software"], fmt="bibtex")
    assert "pyleoclim_software" in out


def test_cite_software_reports_not_imported_library():
    out = cite_software(SAMPLE, libraries=["definitely_not_here"], fmt="bibtex")
    assert "definitely_not_here" in out


def test_cite_software_apa_routes_to_render(monkeypatch):
    import bibliography
    monkeypatch.setattr(bibliography, "render_apa", lambda entries: "APA_SENTINEL")
    out = cite_software(SAMPLE, libraries="pyleoclim", fmt="apa")
    assert out.startswith("APA_SENTINEL")


def test_cite_data_injects_apa_cell(tmp_path, monkeypatch):
    import dataset_detection
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda code: [["filtered_df2", "LiPDGraph"]],
    )
    nb_in = tmp_path / "in.ipynb"
    nb_out = tmp_path / "out.ipynb"
    _write_lipdgraph_notebook(str(nb_in))

    from orchestrator import cite_data
    pairs = cite_data(str(nb_in), fmt="apa", output_path=str(nb_out))
    assert pairs == [["filtered_df2", "LiPDGraph"]]

    import nbformat
    out = nbformat.read(str(nb_out), as_version=4)
    injected = out.cells[-1].source
    assert "render_bibtex_strings_to_apa(_bib_filtered_df2)" in injected
    assert "repositories/LiPDVerse-dynamic" in injected


def test_cite_data_rejects_bad_fmt(tmp_path):
    from orchestrator import cite_data
    with pytest.raises(ValueError):
        cite_data(str(tmp_path / "x.ipynb"), fmt="html")
