"""
Unit tests for orchestrator.py. Fully offline: cite_software injects a metadata
cell (reads imports + writes a notebook, no Gemini), and cite_data is exercised
with a monkeypatched detector so no network call is made. Both write to a tmp
output_path so the fixture notebooks are never mutated.

fmt is accepted and ignored rather than validated, so the tests here pin
acceptance of arbitrary values and identical output across them, not rejection.
"""

import os
import inspect

import pytest

from provenance_agent.orchestrator import cite_software

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


def test_format_validation_is_removed():
    from provenance_agent import orchestrator

    assert not hasattr(orchestrator, "_check_fmt")
    assert not hasattr(orchestrator, "_VALID_FMT")


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


def test_cite_data_accepts_apa_compatibility_mode(tmp_path, monkeypatch):
    from provenance_agent import dataset_detection
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda code: [["filtered_df2", "LiPDGraph"]],
    )
    nb_in = tmp_path / "in.ipynb"
    nb_out = tmp_path / "out.ipynb"
    _write_lipdgraph_notebook(str(nb_in))

    from provenance_agent.orchestrator import cite_data
    pairs = cite_data(str(nb_in), fmt="apa", output_path=str(nb_out))
    assert pairs == [["filtered_df2", "LiPDGraph"]]

    import nbformat
    out = nbformat.read(str(nb_out), as_version=4)
    assert any("repositories/LiPDVerse-dynamic" in c.source for c in out.cells)
    assert all("render_bibtex_strings_to_apa" not in c.source for c in out.cells)
    assert all("print(" not in c.source for c in out.cells)
    assert "provenance_datasets" not in out.cells[-1].source
    assert "display(_meta_filtered_df2)" in out.cells[-1].source


def test_cite_data_defaults_to_bibtex():
    from provenance_agent.orchestrator import _cite_data_tool_entry, cite_data

    assert inspect.signature(cite_data).parameters["fmt"].default == "bibtex"
    assert inspect.signature(_cite_data_tool_entry).parameters["fmt"].default == "bibtex"


def test_cite_data_reuses_precomputed_detection(tmp_path, monkeypatch):
    from provenance_agent import dataset_detection

    monkeypatch.setattr(
        dataset_detection,
        "detect_datasets",
        lambda _code: (_ for _ in ()).throw(AssertionError("detector reran")),
    )
    notebook = tmp_path / "in.ipynb"
    _write_lipdgraph_notebook(str(notebook))

    from provenance_agent.orchestrator import cite_data
    pairs = cite_data(
        str(notebook),
        detected_pairs=[["filtered_df2", "LiPDGraph"]],
    )

    assert pairs == [["filtered_df2", "LiPDGraph"]]


@pytest.mark.parametrize("fmt", ["bibtex", "apa", "html", "not-a-format"])
def test_cite_data_accepts_any_fmt_with_identical_output(tmp_path, monkeypatch, fmt):
    """Every fmt value is accepted and produces the same cell as the default."""
    from provenance_agent import dataset_detection
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda _path: [["filtered_df2", "LiPDGraph"]],
    )
    import nbformat
    from provenance_agent.orchestrator import cite_data

    def inject(suffix, **kwargs):
        nb_in = tmp_path / f"in_{suffix}.ipynb"
        nb_out = tmp_path / f"out_{suffix}.ipynb"
        _write_lipdgraph_notebook(str(nb_in))
        pairs = cite_data(str(nb_in), output_path=str(nb_out), **kwargs)
        return pairs, nbformat.read(str(nb_out), as_version=4).cells[-1].source

    default_pairs, default_cell = inject("default")
    pairs, cell = inject(fmt.replace("-", "_"), fmt=fmt)

    assert pairs == default_pairs == [["filtered_df2", "LiPDGraph"]]
    assert cell == default_cell


def test_cite_data_tool_accepts_any_fmt(tmp_path, monkeypatch):
    """The StructuredTool boundary accepts fmt too, so tool callers do not break."""
    from provenance_agent import dataset_detection
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda _path: [["filtered_df2", "LiPDGraph"]],
    )
    notebook = tmp_path / "tool.ipynb"
    _write_lipdgraph_notebook(str(notebook))

    from provenance_agent.orchestrator import cite_data_tool
    pairs = cite_data_tool.invoke(
        {"notebook_path": str(notebook), "fmt": "apa"}
    )
    assert pairs == [["filtered_df2", "LiPDGraph"]]


def test_cite_data_does_not_use_the_deprecated_llm_detector(tmp_path, monkeypatch):
    """
    The active data path is deterministic. Detonate the deprecated LLM helpers
    and the shared client: reaching any of them fails the test.
    """
    from provenance_agent import dataset_detection
    from provenance_agent import llm

    def detonate(*_args, **_kwargs):
        raise AssertionError("the deprecated LLM detection path was used")

    class _NoClient:
        """Stands in for the Gemini client; any use of it fails the test."""

        def __getattr__(self, name):
            detonate()

    monkeypatch.setattr(dataset_detection, "build_detection_prompt", detonate)
    monkeypatch.setattr(dataset_detection, "parse_detection_response", detonate)
    monkeypatch.setattr(llm, "llm", _NoClient())

    # A notebook the real analyzer resolves, so this asserts the deterministic
    # detector produced the answer rather than that nothing ran at all.
    import nbformat
    notebook = tmp_path / "deterministic.ipynb"
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "from pylipd.lipd import LiPD\n"
        "import pyleoclim as pyleo\n"
        "D = LiPD()\n"
        'D.load("dataset.lpd")\n'
        "timeseries = D.get_timeseries(D.get_all_dataset_names())\n"
        "series = pyleo.Series(time=timeseries.time, value=timeseries.value)\n"
        "result = series.pca()\n"
    )])
    with open(notebook, "w") as handle:
        nbformat.write(nb, handle)

    from provenance_agent.orchestrator import cite_data
    assert cite_data(str(notebook)) == [["D", "PyLiPD"]]


@pytest.mark.parametrize("target", ["830587", "TR04EVLI"])
def test_cite_data_pyleotups_target_warns_and_leaves_notebook_alone(
    tmp_path, monkeypatch, target
):
    """A specific PyleoTUPS study, by numeric ID or name, is a warned no-op."""
    from provenance_agent import dataset_detection
    import nbformat
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda _path: [["ds", "PyleoTUPS"]],
    )
    notebook = tmp_path / "pyleotups.ipynb"
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("ds = PangaeaDataset()")]
    )
    with open(notebook, "w") as handle:
        nbformat.write(nb, handle)
    before = notebook.read_bytes()

    from provenance_agent.orchestrator import cite_data
    with pytest.warns(UserWarning, match="specific PyleoTUPS"):
        pairs = cite_data(str(notebook), targets=target)

    assert pairs == []
    assert notebook.read_bytes() == before


def test_tools_are_structured_tools():
    from langchain_core.tools import StructuredTool
    from provenance_agent.orchestrator import cite_software_tool, cite_data_tool
    assert isinstance(cite_software_tool, StructuredTool)
    assert isinstance(cite_data_tool, StructuredTool)
    assert "detected_pairs" not in cite_data_tool.args


def test_tool_names_and_descriptions():
    from provenance_agent.orchestrator import cite_software_tool, cite_data_tool
    assert cite_software_tool.name == "cite_software"
    assert cite_data_tool.name == "cite_data"
    assert "software" in cite_software_tool.description.lower()
    assert "dataset" in cite_data_tool.description.lower()


def test_cite_software_tool_invokes(tmp_path):
    from provenance_agent.orchestrator import cite_software_tool
    out = cite_software_tool.invoke(
        {"notebook_path": SAMPLE, "libraries": "pyleoclim",
         "output_path": str(tmp_path / "out.ipynb")}
    )
    assert out == ["pyleoclim"]
