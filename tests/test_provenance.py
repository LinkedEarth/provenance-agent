"""
Unit tests for provenance.py (the %provenance IPython magic).

The magic is a presentation layer over agent.run(), which needs Gemini, so
agent.run and the ipynbname auto-detect are monkeypatched here - no test makes
a network call. What's covered: notebook-path precedence and its error
message, the two different result shapes the tools return, and the
empty-request / unroutable-request paths.
"""

import os
import sys

import pytest
from IPython.core.error import UsageError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import provenance


@pytest.fixture(autouse=True)
def clear_notebook_path():
    """Each test starts with no session override set."""
    provenance._notebook_path = None
    yield
    provenance._notebook_path = None


# --- resolve_notebook_path ---------------------------------------------------

def test_override_wins_over_autodetect(monkeypatch):
    monkeypatch.setattr(provenance, "_autodetect_path", lambda: "/auto/detected.ipynb")
    provenance.set_notebook_path("explicit.ipynb")
    assert provenance.resolve_notebook_path() == "explicit.ipynb"


def test_falls_back_to_autodetect(monkeypatch):
    monkeypatch.setattr(provenance, "_autodetect_path", lambda: "/auto/detected.ipynb")
    assert provenance.resolve_notebook_path() == "/auto/detected.ipynb"


def test_unresolvable_path_raises_usage_error_naming_the_fix(monkeypatch):
    monkeypatch.setattr(provenance, "_autodetect_path", lambda: None)
    with pytest.raises(UsageError) as excinfo:
        provenance.resolve_notebook_path()
    assert "%provenance_notebook" in str(excinfo.value)


def test_set_notebook_path_echoes_what_it_stored():
    assert "explicit.ipynb" in provenance.set_notebook_path("explicit.ipynb")


# --- result formatting -------------------------------------------------------

def test_software_result_is_printed_unchanged():
    call = {
        "name": "cite_software",
        "args": {"notebook_path": "nb.ipynb"},
        "result": "Tierney, J. E. (2015). A title. Journal, 1(2), 3-4.",
    }
    assert provenance._format_result(call) == call["result"]


def test_data_result_summarizes_injected_cells():
    call = {
        "name": "cite_data",
        "args": {"notebook_path": "paleoPCAlite.ipynb"},
        "result": [["filtered_df2", "LiPDGraph"], ["D", "PyLiPD"]],
    }
    out = provenance._format_result(call)
    assert "paleoPCAlite.ipynb" in out
    assert "filtered_df2 (LiPDGraph)" in out
    assert "D (PyLiPD)" in out
    assert "run" in out.lower()  # tells the user the citations aren't produced yet


def test_data_result_with_no_datasets_mentions_reading_from_disk():
    call = {
        "name": "cite_data",
        "args": {"notebook_path": "empty.ipynb"},
        "result": [],
    }
    out = provenance._format_result(call)
    assert "no datasets" in out.lower()
    assert "unsaved" in out.lower()


def test_empty_software_result_mentions_reading_from_disk():
    call = {"name": "cite_software", "args": {"notebook_path": "nb.ipynb"}, "result": ""}
    assert "unsaved" in provenance._format_result(call).lower()


# --- cite --------------------------------------------------------------------

def test_empty_request_raises_usage_error(monkeypatch):
    monkeypatch.setattr(provenance, "_autodetect_path", lambda: "nb.ipynb")
    with pytest.raises(UsageError):
        provenance.cite("   ")


def test_unroutable_request_explains_what_the_tools_cover(monkeypatch):
    monkeypatch.setattr(provenance, "_autodetect_path", lambda: "nb.ipynb")
    monkeypatch.setattr(provenance, "run", lambda request, notebook_path: [])
    out = provenance.cite("what is the weather")
    assert "software" in out.lower() and "datasets" in out.lower()


def test_cite_passes_resolved_path_to_the_agent(monkeypatch):
    seen = {}

    def fake_run(request, notebook_path):
        seen["request"] = request
        seen["notebook_path"] = notebook_path
        return [{"name": "cite_software", "args": {}, "result": "a citation"}]

    monkeypatch.setattr(provenance, "_autodetect_path", lambda: None)
    monkeypatch.setattr(provenance, "run", fake_run)
    provenance.set_notebook_path("explicit.ipynb")

    assert provenance.cite("cite the software") == "a citation"
    assert seen == {"request": "cite the software", "notebook_path": "explicit.ipynb"}


def test_multiple_tool_calls_are_joined():
    calls = [
        {"name": "cite_software", "args": {"notebook_path": "nb.ipynb"}, "result": "SW"},
        {"name": "cite_data", "args": {"notebook_path": "nb.ipynb"},
         "result": [["D", "PyLiPD"]]},
    ]
    out = provenance._format_results(calls)
    assert "SW" in out and "D (PyLiPD)" in out


# --- extension registration --------------------------------------------------

def test_load_ipython_extension_registers_the_magics():
    registered = []
    load_ipython_extension = provenance.load_ipython_extension
    load_ipython_extension(type("Shell", (), {"register_magics": lambda self, m: registered.append(m)})())
    assert registered == [provenance.ProvenanceMagics]


def test_magic_methods_exist():
    assert hasattr(provenance.ProvenanceMagics, "provenance")
    assert hasattr(provenance.ProvenanceMagics, "provenance_notebook")
