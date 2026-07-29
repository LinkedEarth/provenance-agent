"""
Unit tests for agent.py.

The classifier normally uses Gemini, so these tests supply a fake LangChain
Runnable and exercise the offline LCEL stages: typed decisions, warning/no-op
resolution, sequential software/data dispatch, static cell verification, and
the public result envelope. The tool registry remains covered for direct/API
callers even though the model is no longer bound to those tools.
"""

import json
import nbformat
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sample.ipynb")


def test_classifier_prompt_is_json_only_and_uses_context():
    prompt = agent._CLASSIFIER_PROMPT.invoke({
        "notebook_path": "nb.ipynb",
        "imports": ["pandas"],
        "request": "cite pandas",
    })
    system = prompt.messages[0].content

    assert "The notebook path is `nb.ipynb`" in system
    assert "Imported libraries: ['pandas']" in system
    assert "TR04EVLI" in system
    assert "Return only JSON" in system
    assert "Walk through" not in system
    assert "cite_software" not in system
    assert "cite_data" not in system


def test_build_messages_injects_notebook_and_request():
    msgs = agent.build_messages("cite the software", "nb.ipynb")
    assert "nb.ipynb" in msgs[0].content        # system message carries the path
    assert "cite the software" in msgs[1].content  # human message carries the request


def test_tools_by_name_has_both():
    assert set(agent._TOOLS_BY_NAME) == {"cite_software", "cite_data"}


def _fake_model(decision):
    return RunnableLambda(
        lambda _prompt: AIMessage(content=json.dumps(decision))
    )


def test_build_chain_exposes_lcel_graph():
    assert hasattr(agent, "chain")
    assert hasattr(agent, "build_chain")
    assert hasattr(agent.chain, "get_graph")
    names = {node.name for node in agent.chain.get_graph().nodes.values()}
    assert {
        "prepare_context", "classify", "resolve_targets", "dispatch", "verify"
    } <= names


def test_chain_parses_typed_software_target_and_verifies_injection(tmp_path):
    notebook = tmp_path / "sample.ipynb"
    shutil.copyfile(SAMPLE, notebook)
    decision = {
        "action": "cite",
        "scope": "selected",
        "targets": [{"kind": "software", "name": "pyleoclim"}],
        "fmt": "apa",
        "warning": None,
    }

    result = agent.build_chain(_fake_model(decision)).invoke({
        "request": "cite Pyleoclim",
        "notebook_path": str(notebook),
    })

    assert result["status"] == "ok"
    assert result["decision"]["targets"] == [
        {"kind": "software", "name": "pyleoclim"}
    ]
    assert result["dispatch"][0]["name"] == "cite_software"
    assert result["verification"]["combine_cell_present"] is True
    assert result["verification"]["combine_cell_last"] is True


def test_ambiguous_classification_returns_warning_without_mutation(tmp_path):
    notebook = tmp_path / "sample.ipynb"
    shutil.copyfile(SAMPLE, notebook)
    before = notebook.read_bytes()
    decision = {
        "action": "warning",
        "scope": "selected",
        "targets": [],
        "fmt": "apa",
        "warning": "Could not determine whether the request is software or data.",
    }

    result = agent.build_chain(_fake_model(decision)).invoke({
        "request": "cite it",
        "notebook_path": str(notebook),
    })

    assert result["status"] == "warning"
    assert result["dispatch"] == []
    assert notebook.read_bytes() == before


def test_missing_imported_software_target_warns_without_mutation(tmp_path):
    notebook = tmp_path / "sample.ipynb"
    shutil.copyfile(SAMPLE, notebook)
    before = notebook.read_bytes()
    decision = {
        "action": "cite",
        "scope": "selected",
        "targets": [{"kind": "software", "name": "not_imported"}],
        "fmt": "apa",
    }

    result = agent.build_chain(_fake_model(decision)).invoke({
        "request": "cite not_imported",
        "notebook_path": str(notebook),
    })

    assert result["status"] == "warning"
    assert "not imported" in result["warning"]
    assert result["dispatch"] == []
    assert notebook.read_bytes() == before


def test_data_without_detected_pairs_warns_without_mutation(tmp_path, monkeypatch):
    notebook = tmp_path / "sample.ipynb"
    shutil.copyfile(SAMPLE, notebook)
    before = notebook.read_bytes()
    monkeypatch.setattr(agent, "_detect_dataset_pairs", lambda _path: [])
    decision = {
        "action": "cite",
        "scope": "selected",
        "targets": [{"kind": "data", "name": "LR04"}],
        "fmt": "apa",
    }

    result = agent.build_chain(_fake_model(decision)).invoke({
        "request": "cite LR04",
        "notebook_path": str(notebook),
    })

    assert result["status"] == "warning"
    assert "No datasets" in result["warning"]
    assert result["dispatch"] == []
    assert notebook.read_bytes() == before


def test_both_targets_dispatch_in_order_and_reuse_detection(tmp_path, monkeypatch):
    notebook = tmp_path / "sample.ipynb"
    shutil.copyfile(SAMPLE, notebook)
    detected = [["filtered_df2", "LiPDGraph"]]
    detection_calls = []
    dispatch_calls = []

    def fake_detect(path):
        detection_calls.append(path)
        return detected

    def append_segment(path, source):
        with open(path) as handle:
            current = nbformat.read(handle, as_version=4)
        current.cells.append(nbformat.v4.new_code_cell(source))
        from bibliography import ensure_combine_cell
        ensure_combine_cell(current)
        with open(path, "w") as handle:
            nbformat.write(current, handle)

    def fake_software(path, libraries=None):
        dispatch_calls.append(("software", libraries))
        append_segment(path, "_provbib_software = software_frame")
        return ["pyleoclim"]

    def fake_data(path, targets=None, fmt="apa", detected_pairs=None):
        dispatch_calls.append(("data", targets, fmt, detected_pairs))
        append_segment(path, "_provbib_data_filtered_df2 = data_frame")
        return detected_pairs

    monkeypatch.setattr(agent, "_detect_dataset_pairs", fake_detect)
    monkeypatch.setattr(agent, "cite_software", fake_software)
    monkeypatch.setattr(agent, "cite_data", fake_data)
    decision = {
        "action": "cite",
        "scope": "all",
        "kinds": ["software", "data"],
        "targets": [],
        "fmt": "bibtex",
    }

    result = agent.build_chain(_fake_model(decision)).invoke({
        "request": "cite everything",
        "notebook_path": str(notebook),
    })

    assert detection_calls == [str(notebook)]
    assert dispatch_calls == [
        ("software", None),
        ("data", None, "bibtex", detected),
    ]
    assert [call["name"] for call in result["dispatch"]] == [
        "cite_software", "cite_data"
    ]
    assert result["status"] == "ok"
    assert result["verification"]["combine_cell_last"] is True
    assert {cell["tool"] for cell in result["verification"]["cells"]} == {
        "software", "data", "combine"
    }


def test_run_returns_the_chain_envelope(monkeypatch):
    expected = {
        "status": "warning",
        "decision": None,
        "dispatch": [],
        "verification": {},
    }
    monkeypatch.setattr(
        agent,
        "chain",
        RunnableLambda(lambda _input: expected),
    )
    assert agent.run("whatever", SAMPLE) == expected
