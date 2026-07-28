"""
Unit tests for orchestrator.py. Fully offline: cite_software injects a metadata
cell (reads imports + writes a notebook, no Gemini), and cite_data is exercised
with a monkeypatched detector so no network call is made. Both write to a tmp
output_path so the fixture notebooks are never mutated.
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


def test_cite_software_all_injects_and_returns_libraries(tmp_path):
    out = tmp_path / "out.ipynb"
    wanted = cite_software(SAMPLE, output_path=str(out))
    assert "pyleoclim" in wanted

    import nbformat
    nb = nbformat.read(str(out), as_version=4)
    assert any("collect_library_entries(" in c.source for c in nb.cells)


def test_cite_software_one_library_filters(tmp_path):
    out = tmp_path / "out.ipynb"
    wanted = cite_software(SAMPLE, libraries="pyleoclim", output_path=str(out))
    assert wanted == ["pyleoclim"]


def test_cite_software_passes_citation_type_into_cell(tmp_path):
    out = tmp_path / "out.ipynb"
    cite_software(SAMPLE, libraries="pyleoclim", citation_types=["software"], output_path=str(out))

    import nbformat
    nb = nbformat.read(str(out), as_version=4)
    assert any("['pyleoclim'], ['software']" in c.source for c in nb.cells)


def test_cite_software_unimported_library_returns_empty(tmp_path):
    out = tmp_path / "out.ipynb"
    assert cite_software(SAMPLE, libraries=["definitely_not_here"], output_path=str(out)) == []


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
    assert any("render_bibtex_strings_to_apa(_bib_filtered_df2)" in c.source for c in out.cells)
    assert any("repositories/LiPDVerse-dynamic" in c.source for c in out.cells)
    assert "display(provenance_datasets)" in out.cells[-1].source


def test_cite_data_reuses_precomputed_detection(tmp_path, monkeypatch):
    import dataset_detection

    monkeypatch.setattr(
        dataset_detection,
        "detect_datasets",
        lambda _code: (_ for _ in ()).throw(AssertionError("detector reran")),
    )
    notebook = tmp_path / "in.ipynb"
    _write_lipdgraph_notebook(str(notebook))

    from orchestrator import cite_data
    pairs = cite_data(
        str(notebook),
        detected_pairs=[["filtered_df2", "LiPDGraph"]],
    )

    assert pairs == [["filtered_df2", "LiPDGraph"]]


def test_cite_data_rejects_bad_fmt(tmp_path):
    from orchestrator import cite_data
    with pytest.raises(ValueError):
        cite_data(str(tmp_path / "x.ipynb"), fmt="html")


def test_tools_are_structured_tools():
    from langchain_core.tools import StructuredTool
    from orchestrator import cite_software_tool, cite_data_tool
    assert isinstance(cite_software_tool, StructuredTool)
    assert isinstance(cite_data_tool, StructuredTool)
    assert "detected_pairs" not in cite_data_tool.args


def test_tool_names_and_descriptions():
    from orchestrator import cite_software_tool, cite_data_tool
    assert cite_software_tool.name == "cite_software"
    assert cite_data_tool.name == "cite_data"
    assert "software" in cite_software_tool.description.lower()
    assert "dataset" in cite_data_tool.description.lower()


def test_cite_software_tool_invokes(tmp_path):
    from orchestrator import cite_software_tool
    out = cite_software_tool.invoke(
        {"notebook_path": SAMPLE, "libraries": "pyleoclim",
         "output_path": str(tmp_path / "out.ipynb")}
    )
    assert out == ["pyleoclim"]
