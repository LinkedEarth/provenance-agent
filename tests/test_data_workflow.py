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
    kernel. The one exception is test_software_then_data_leaves_one_cell_each,
    an integration test that writes a notebook to tmp_path and reads it back to
    confirm the two workflows leave exactly one self-displaying cell each across
    a real write/read roundtrip. The live-kernel execution (the user
    running the injected cell) is out of scope for these tests.

Design Decisions:
    - Each tool gets its own cell-generation test so a failure pinpoints which
      retrieval path broke.
    - LiPDGraph retrieval must convert the terminal DataFrame to a LiPD object,
      so its cell references the DataFrame's dataSetName column and the LiPDVerse
      endpoint - asserted explicitly.
    - Targeted PyLiPD and LiPDGraph cells load only the requested names; the
      untargeted paths remain unchanged. PyleoTUPS keeps its existing
      post-retrieval metadata filter.
    - Unsupported tools raise ValueError rather than silently emitting nothing.
"""

import warnings

import nbformat
import pytest

from provenance_agent.data_workflow import (
    build_retrieval_cell,
    extract_lipdgraph_endpoint,
    filter_datasets,
    build_dataset_cell,
    inject_retrieval_cells,
    split_targets,
)


# --- build_retrieval_cell ----------------------------------------------------

def test_pylipd_cell_calls_get_bibtex_on_variable():
    cell = build_retrieval_cell("D", "PyLiPD")
    assert "D.get_bibtex(remote=True)" in cell


def test_targeted_pylipd_cell_loads_requested_names_directly():
    cell = build_retrieval_cell("D", "PyLiPD", dataset_names=["tr04evli"])
    assert "_lipd_D.load_remote_datasets(['tr04evli'])" in cell
    assert "_bib_D, _meta_D = _lipd_D.get_bibtex(remote=True)" in cell
    assert "_bib_D, _meta_D = D.get_bibtex(remote=True)" not in cell
    assert "_want_D" not in cell


def test_targeted_pylipd_loads_only_requested_names():
    cell = build_retrieval_cell("D", "PyLiPD", dataset_names=["TR04EVLI"])
    assert "_lipd_D.load_remote_datasets(['TR04EVLI'])" in cell
    assert "_bib_D, _meta_D = _lipd_D.get_bibtex(remote=True)" in cell
    assert "_bib_D, _meta_D = D.get_bibtex(remote=True)" not in cell


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


def test_targeted_lipdgraph_cell_loads_requested_names_directly():
    cell = build_retrieval_cell(
        "filtered_df2", "LiPDGraph", dataset_names=["tr04evli"]
    )
    assert "_names_filtered_df2 = ['tr04evli']" in cell
    assert 'filtered_df2["dataSetName"]' not in cell
    assert "load_remote_datasets(_names_filtered_df2)" in cell


def test_targeted_lipdgraph_loads_only_requested_names():
    cell = build_retrieval_cell(
        "filtered_df2", "LiPDGraph", dataset_names=["TR04EVLI", "SP13CHPE"]
    )
    assert "_names_filtered_df2 = ['TR04EVLI', 'SP13CHPE']" in cell
    assert "load_remote_datasets(_names_filtered_df2)" in cell
    assert 'filtered_df2["dataSetName"]' not in cell


def test_lipdgraph_cell_uses_supplied_endpoint():
    cell = build_retrieval_cell("df", "LiPDGraph", endpoint="https://example.org/repositories/Other")
    assert 'set_endpoint("https://example.org/repositories/Other")' in cell


def test_unsupported_tool_raises():
    with pytest.raises(ValueError):
        build_retrieval_cell("iso_ds", "xarray")


@pytest.mark.parametrize("tool", ["PyLiPD", "PyleoTUPS", "LiPDGraph"])
def test_every_block_binds_bib_and_meta_without_printing(tool):
    """Retrieval blocks bind citation metadata without printing."""
    block = build_retrieval_cell("D", tool)
    assert "_meta_D" in block and "_bib_D" in block
    assert "print(" not in block
    assert "display(" not in block


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


def test_split_targets_treats_variable_like_any_dataset_name():
    assert split_targets(PAIRS, "ds") == (PAIRS, ["ds"])


def test_split_targets_dataset_name_applies_to_all_pairs():
    assert split_targets(PAIRS, "TR04EVLI") == (PAIRS, ["TR04EVLI"])


def test_split_targets_mixed_targets_keeps_all_for_name_filter():
    assert split_targets(PAIRS, ["ds", "TR04EVLI"]) == (
        PAIRS,
        ["ds", "TR04EVLI"],
    )


@pytest.mark.parametrize("target", ["TR04EVLI", "ds", "830587"])
def test_pyleotups_target_warns_without_mutating_notebook(
    tmp_path, monkeypatch, target
):
    from provenance_agent import dataset_detection

    monkeypatch.setattr(
        dataset_detection,
        "detect_datasets",
        lambda _code: [["ds", "PyleoTUPS"]],
    )

    notebook = tmp_path / "pyleotups.ipynb"
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("ds = PangaeaDataset()")]
    )
    with open(notebook, "w") as handle:
        nbformat.write(nb, handle)
    before = notebook.read_bytes()

    from provenance_agent.data_workflow import generate_data_workflow

    with pytest.warns(UserWarning, match="specific PyleoTUPS"):
        pairs = generate_data_workflow(
            str(notebook),
            targets=target,
        )

    assert pairs == []
    assert notebook.read_bytes() == before


def test_pyleotups_all_datasets_request_still_injects(tmp_path, monkeypatch):
    from provenance_agent import dataset_detection

    monkeypatch.setattr(
        dataset_detection,
        "detect_datasets",
        lambda _code: [["ds", "PyleoTUPS"]],
    )

    notebook = tmp_path / "pyleotups.ipynb"
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("ds = PangaeaDataset()")]
    )
    with open(notebook, "w") as handle:
        nbformat.write(nb, handle)

    from provenance_agent.data_workflow import generate_data_workflow

    pairs = generate_data_workflow(str(notebook))

    assert pairs == [["ds", "PyleoTUPS"]]
    written = nbformat.read(str(notebook), as_version=4)
    assert "ds.get_publications()" in written.cells[-1].source


def test_pyleotups_warning_only_applies_to_requested_tool(tmp_path, monkeypatch):
    from provenance_agent import dataset_detection

    monkeypatch.setattr(
        dataset_detection,
        "detect_datasets",
        lambda _code: [["ds", "PyleoTUPS"], ["df", "LiPDGraph"]],
    )

    notebook = tmp_path / "lipdgraph.ipynb"
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("df = pandas.DataFrame()")]
    )
    with open(notebook, "w") as handle:
        nbformat.write(nb, handle)

    from provenance_agent.data_workflow import generate_data_workflow

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pairs = generate_data_workflow(
            str(notebook),
            tool="LiPDGraph",
            targets="TR04EVLI",
        )

    assert not caught
    assert pairs == [["df", "LiPDGraph"]]


def test_filter_by_variable_str_still_works():
    assert filter_datasets(PAIRS, variable="ds") == [["ds", "PyleoTUPS"]]


def test_generate_data_workflow_accepts_dataset_name_target(tmp_path, monkeypatch):
    from provenance_agent import dataset_detection

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

    from provenance_agent.data_workflow import generate_data_workflow
    pairs = generate_data_workflow(
        str(source), targets="tr04evli", output_path=str(output)
    )

    assert pairs == [["filtered_df2", "LiPDGraph"]]
    generated = nbformat.read(str(output), as_version=4)
    retrieval = generated.cells[1].source
    assert "_names_filtered_df2 = ['tr04evli']" in retrieval
    assert 'filtered_df2["dataSetName"]' not in retrieval


def test_generate_data_workflow_rejects_variable_targets(tmp_path):
    from provenance_agent.data_workflow import generate_data_workflow

    with pytest.raises(ValueError, match="source-variable targeting"):
        generate_data_workflow(
            str(tmp_path / "missing.ipynb"),
            variable="D",
        )


# --- inject_retrieval_cells --------------------------------------------------

def test_inject_appends_exactly_one_cell_for_all_datasets():
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("df_res = pd.read_csv(data)"))
    inject_retrieval_cells(nb, [["D", "PyLiPD"], ["ds", "PyleoTUPS"]])
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 2  # original + 1 injected
    injected = code_cells[1].source
    assert "D.get_bibtex(remote=True)" in injected
    assert "ds.get_publications()" in injected
    assert "display(_meta_D)" in injected
    assert "display(_meta_ds)" in injected
    assert "pd.concat" not in injected
    assert "provenance_datasets" not in injected


def test_inject_with_no_pairs_appends_nothing():
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("x = 1"))
    inject_retrieval_cells(nb, [])
    assert len(nb.cells) == 1


# --- BibTeX DataFrame output and fmt compatibility --------------------------

def test_bibtex_cell_keeps_the_returned_metadata_binding():
    cell = build_dataset_cell([["D", "PyLiPD"]])
    assert "_bib_D, _meta_D = D.get_bibtex(remote=True)" in cell
    assert "display(_meta_D)" in cell
    assert "provenance_datasets" not in cell


def test_dataset_cell_displays_metadata_frame_without_combining():
    cell = build_dataset_cell([["D", "PyLiPD"]])
    assert "D.get_bibtex(remote=True)" in cell
    assert "import pandas as pd" not in cell
    assert "pd.concat" not in cell
    assert "provenance_datasets" not in cell
    assert cell.rstrip().endswith("display(_meta_D)")


def test_apa_fmt_is_accepted_but_output_neutral_until_typed_api():
    bibtex = build_dataset_cell([["D", "PyLiPD"]], fmt="bibtex")
    apa = build_dataset_cell([["D", "PyLiPD"]], fmt="apa")
    assert apa == bibtex


def test_multi_dataset_cell_displays_each_metadata_frame_without_concatenating():
    cell = build_dataset_cell([["D", "PyLiPD"], ["ds", "PyleoTUPS"]])
    assert "_bib_D, _meta_D = D.get_bibtex(remote=True)" in cell
    assert "ds.get_publications()" in cell
    assert "pd.concat" not in cell
    assert "display(_meta_D)" in cell
    assert "display(_meta_ds)" in cell


# --- cross-workflow integration -----------------------------------------------

def test_software_then_data_leaves_one_cell_each(tmp_path, monkeypatch):
    from provenance_agent import dataset_detection
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

    from provenance_agent.software_workflow import generate_software_workflow
    from provenance_agent.data_workflow import generate_data_workflow

    generate_software_workflow(str(path))
    generate_data_workflow(str(path))

    final = nbformat.read(str(path), as_version=4)
    software = [c for c in final.cells if "provenance_software" in c.source]
    data = [
        c for c in final.cells
        if "# provenance-agent-generated" in c.source
        and "provenance_software" not in c.source
    ]
    assert len(software) == 1
    assert len(data) == 1
    assert "# provenance-combine-cell" not in "".join(c.source for c in final.cells)
    assert "display(provenance_software)" in software[0].source
    assert "display(_meta_filtered_df2)" in data[0].source


def test_legacy_combine_cell_is_stripped(tmp_path, monkeypatch):
    """Notebooks from older runs carry a combine cell nothing manages anymore."""
    from provenance_agent import dataset_detection
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda code: [["filtered_df2", "LiPDGraph"]],
    )

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
        "filtered_df2 = None"
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
        "# provenance-combine-cell\nimport pandas as pd\ndisplay(provenance_bibliography)"
    ))
    path = tmp_path / "legacy.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    from provenance_agent.data_workflow import generate_data_workflow
    generate_data_workflow(str(path))

    final = nbformat.read(str(path), as_version=4)
    assert "# provenance-combine-cell" not in "".join(c.source for c in final.cells)
    assert sum(
        "# provenance-agent-generated" in c.source
        and "provenance_software" not in c.source
        for c in final.cells
    ) == 1


def test_repeated_runs_replace_the_dataset_cell(tmp_path, monkeypatch):
    """One dataset cell means one, however many times the workflow runs."""
    from provenance_agent import dataset_detection
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda code: [["filtered_df2", "LiPDGraph"]],
    )

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
        "filtered_df2 = None"
    ))
    path = tmp_path / "nb.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    from provenance_agent.data_workflow import generate_data_workflow
    for _ in range(3):
        generate_data_workflow(str(path))

    written = nbformat.read(str(path), as_version=4)
    assert sum(
        "# provenance-agent-generated" in c.source
        and "provenance_software" not in c.source
        for c in written.cells
    ) == 1
    assert len(written.cells) == 2
