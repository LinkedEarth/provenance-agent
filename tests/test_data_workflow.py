"""
test_data_workflow.py

Purpose:
    Unit tests for the pure logic in data_workflow.py: generating the per-tool
    retrieval-cell source (build_retrieval_cell), filtering detected pairs
    (filter_datasets), and injecting retrieval cells into a notebook
    (inject_retrieval_cells).

Implementation:
    build_retrieval_cell and filter_datasets are pure string/list functions.
    inject_retrieval_cells operates on an in-memory nbformat NotebookNode built
    with nbformat.v4.new_notebook(), so most tests do no file I/O and no live
    kernel. The one exception is test_software_then_data_leaves_one_combine_cell_last,
    an integration test that writes a notebook to tmp_path and reads it back to
    confirm the software and data workflows leave a single combine cell last
    across a real write/read roundtrip. The live-kernel execution (the user
    running the injected cell) is out of scope for these tests.

Design Decisions:
    - Each tool gets its own cell-generation test so a failure pinpoints which
      retrieval path broke.
    - LiPDGraph retrieval must convert the terminal DataFrame to a LiPD object,
      so its cell references the DataFrame's dataSetName column and the LiPDVerse
      endpoint - asserted explicitly.
    - Dataset provenance frames reuse the metadata DataFrame returned by the
      source library, preserving its full schema for the combined output.
      Dataset-name targets filter that metadata after retrieval.
    - Unsupported tools raise ValueError rather than silently emitting nothing.
"""

import os
import sys

import nbformat
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_workflow import (
    build_retrieval_cell,
    extract_lipdgraph_endpoint,
    filter_datasets,
    inject_retrieval_cells,
    split_targets,
)


# --- build_retrieval_cell ----------------------------------------------------

def test_pylipd_cell_calls_get_bibtex_on_variable():
    cell = build_retrieval_cell("D", "PyLiPD")
    assert "D.get_bibtex(remote=True)" in cell


def test_pylipd_cell_filters_metadata_after_retrieval():
    cell = build_retrieval_cell("D", "PyLiPD", dataset_names=["tr04evli"])
    assert "D.get_bibtex(remote=True)" in cell
    assert '_meta_D["dsname"].astype("string").str.casefold()' in cell
    assert ".isin(_want_D)" in cell
    assert "get_all_dataset_names" not in cell


def test_pyleotups_cell_calls_get_publications_on_variable():
    cell = build_retrieval_cell("ds", "PyleoTUPS")
    assert "ds.get_publications()" in cell


def test_pyleotups_cell_filters_metadata_when_available():
    cell = build_retrieval_cell("ds", "PyleoTUPS", dataset_names=["TR04EVLI"])
    assert "ds.get_publications()" in cell
    assert 'if "dsname" in _meta_ds.columns:' in cell
    assert '_meta_ds["dsname"].astype("string").str.casefold()' in cell


def test_lipdgraph_cell_converts_dataframe_to_lipd():
    cell = build_retrieval_cell("filtered_df2", "LiPDGraph")
    assert 'filtered_df2["dataSetName"]' in cell
    assert "load_remote_datasets" in cell
    assert "linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic" in cell
    assert "get_bibtex(remote=True)" in cell


def test_lipdgraph_cell_filters_metadata_after_retrieval():
    cell = build_retrieval_cell(
        "filtered_df2", "LiPDGraph", dataset_names=["tr04evli"]
    )
    assert "filtered_df2[\"dataSetName\"].unique().tolist()" in cell
    assert "_names_filtered_df2 = filtered_df2" in cell
    assert '_meta_filtered_df2["dsname"].astype("string").str.casefold()' in cell
    assert "load_remote_datasets(_names_filtered_df2)" in cell


def test_lipdgraph_cell_uses_supplied_endpoint():
    cell = build_retrieval_cell("df", "LiPDGraph", endpoint="https://example.org/repositories/Other")
    assert 'set_endpoint("https://example.org/repositories/Other")' in cell


def test_unsupported_tool_raises():
    with pytest.raises(ValueError):
        build_retrieval_cell("iso_ds", "xarray")


@pytest.mark.parametrize("tool", ["PyLiPD", "PyleoTUPS", "LiPDGraph"])
def test_every_cell_binds_provbib_data_and_drops_meta_display(tool):
    cell = build_retrieval_cell("D", tool)
    assert "_provbib_data_D = _meta_D" in cell
    assert "render_bibtex_strings_to_df" not in cell
    assert "display(_meta_D)" not in cell


# --- extract_lipdgraph_endpoint ----------------------------------------------

def test_extract_endpoint_from_url_assignment():
    code = "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
    assert extract_lipdgraph_endpoint(code) == (
        "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic"
    )


def test_extract_endpoint_skips_bare_host_url():
    code = (
        "base = 'https://linkedearth.graphdb.mint.isi.edu'\n"
        "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
    )
    assert extract_lipdgraph_endpoint(code) == (
        "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic"
    )


def test_extract_endpoint_returns_none_when_absent():
    assert extract_lipdgraph_endpoint("x = 1\nimport pandas as pd") is None


def test_extract_endpoint_ignores_bare_host_only():
    assert extract_lipdgraph_endpoint("url = 'https://linkedearth.graphdb.mint.isi.edu'") is None


def test_extract_endpoint_survives_syntax_error():
    assert extract_lipdgraph_endpoint("def broken(:\n  pass") is None


# --- filter_datasets ---------------------------------------------------------

PAIRS = [["D", "PyLiPD"], ["ds", "PyleoTUPS"], ["filtered_df2", "LiPDGraph"]]


def test_filter_by_tool_is_case_insensitive():
    assert filter_datasets(PAIRS, tool="pylipd") == [["D", "PyLiPD"]]


def test_filter_by_variable():
    assert filter_datasets(PAIRS, variable="ds") == [["ds", "PyleoTUPS"]]


def test_no_filter_returns_all():
    assert filter_datasets(PAIRS) == PAIRS


def test_filter_by_variable_list():
    assert filter_datasets(PAIRS, variable=["D", "filtered_df2"]) == [
        ["D", "PyLiPD"], ["filtered_df2", "LiPDGraph"]
    ]


def test_split_targets_none_keeps_all_without_dataset_names():
    assert split_targets(PAIRS, None) == (PAIRS, [])


def test_split_targets_empty_keeps_all_without_dataset_names():
    assert split_targets(PAIRS, []) == (PAIRS, [])


def test_split_targets_variable_target_keeps_matching_pair():
    assert split_targets(PAIRS, "ds") == ([PAIRS[1]], [])


def test_split_targets_dataset_name_applies_to_all_pairs():
    assert split_targets(PAIRS, "TR04EVLI") == (PAIRS, ["TR04EVLI"])


def test_split_targets_mixed_targets_keeps_all_for_name_filter():
    assert split_targets(PAIRS, ["ds", "TR04EVLI"]) == (PAIRS, ["TR04EVLI"])


def test_filter_by_variable_str_still_works():
    assert filter_datasets(PAIRS, variable="ds") == [["ds", "PyleoTUPS"]]


def test_generate_data_workflow_accepts_dataset_name_target(tmp_path, monkeypatch):
    import dataset_detection

    monkeypatch.setattr(
        dataset_detection,
        "detect_datasets",
        lambda code: [["filtered_df2", "LiPDGraph"]],
    )

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
        "filtered_df2 = None"
    ))
    source = tmp_path / "input.ipynb"
    output = tmp_path / "output.ipynb"
    with open(source, "w") as f:
        nbformat.write(nb, f)

    from data_workflow import generate_data_workflow
    pairs = generate_data_workflow(
        str(source), targets="tr04evli", output_path=str(output)
    )

    assert pairs == [["filtered_df2", "LiPDGraph"]]
    generated = nbformat.read(str(output), as_version=4)
    retrieval = generated.cells[1].source
    assert '_meta_filtered_df2["dsname"].astype("string").str.casefold()' in retrieval


def test_generate_data_workflow_rejects_both_target_aliases(tmp_path):
    from data_workflow import generate_data_workflow

    with pytest.raises(ValueError, match="either targets or variable"):
        generate_data_workflow(
            str(tmp_path / "missing.ipynb"),
            variable="D",
            targets="TR04EVLI",
        )


# --- inject_retrieval_cells --------------------------------------------------

def test_inject_appends_one_code_cell_per_pair():
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("df_res = pd.read_csv(data)"))
    inject_retrieval_cells(nb, [["D", "PyLiPD"], ["ds", "PyleoTUPS"]])
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 3  # original + 2 injected
    assert "D.get_bibtex(remote=True)" in code_cells[1].source
    assert "ds.get_publications()" in code_cells[2].source


# --- fmt parameter for APA rendering -----------------------------------------

def test_pylipd_cell_apa_renders_via_bibliography():
    cell = build_retrieval_cell("D", "PyLiPD", fmt="apa")
    assert "_bib_D, _meta_D = D.get_bibtex(remote=True)" in cell
    assert "from bibliography import render_bibtex_strings_to_apa" in cell
    assert "print(render_bibtex_strings_to_apa(_bib_D))" in cell


def test_pyleotups_cell_apa_wraps_publications():
    cell = build_retrieval_cell("ds", "PyleoTUPS", fmt="apa")
    assert "ds.get_publications()" in cell
    assert "render_bibtex_strings_to_apa(_bib_ds)" in cell


def test_bibtex_fmt_is_unchanged_default():
    cell = build_retrieval_cell("D", "PyLiPD")
    assert 'print("\\n".join(_bib_D))' in cell
    assert "render_bibtex_strings_to_apa" not in cell


# --- cross-workflow integration -----------------------------------------------

def test_software_then_data_leaves_one_combine_cell_last(tmp_path, monkeypatch):
    import dataset_detection
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda code: [["filtered_df2", "LiPDGraph"]],
    )

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        "import pyleoclim\n"
        "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
        "filtered_df2 = None"
    ))
    path = tmp_path / "nb.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    from software_workflow import generate_software_workflow
    from data_workflow import generate_data_workflow

    generate_software_workflow(str(path))
    generate_data_workflow(str(path))

    final = nbformat.read(str(path), as_version=4)
    marked = [c for c in final.cells if "# provenance-combine-cell" in c.source]
    assert len(marked) == 1
    assert final.cells[-1] is marked[0]
    assert any("_provbib_software" in c.source for c in final.cells)
    assert any("_provbib_data_filtered_df2" in c.source for c in final.cells)
