"""
Unit tests for provenance_agent.magic (the %provenance IPython magic).

The magic is a presentation layer over agent.run(), which calls a model, so
agent.run and the ipynbname auto-detect are monkeypatched here - no test makes
a network call. What's covered: notebook-path precedence and its error
message, single-call formatting, the structured result envelope, and the
empty-request / unroutable-request paths.

These target the implementation module directly. The top-level `provenance`
shim that `%load_ext provenance` loads is covered separately in
test_provenance_shim.py, so a shim regression cannot hide behind a passing
implementation (or the reverse).
"""

import pytest
from IPython.core.error import UsageError

from provenance_agent import magic


@pytest.fixture(autouse=True)
def clear_notebook_path():
    """Each test starts with no session override set."""
    magic._notebook_path = None
    yield
    magic._notebook_path = None


# --- resolve_notebook_path ---------------------------------------------------

def test_override_wins_over_autodetect(monkeypatch):
    monkeypatch.setattr(magic, "_autodetect_path", lambda: "/auto/detected.ipynb")
    magic.set_notebook_path("explicit.ipynb")
    assert magic.resolve_notebook_path() == "explicit.ipynb"


def test_falls_back_to_autodetect(monkeypatch):
    monkeypatch.setattr(magic, "_autodetect_path", lambda: "/auto/detected.ipynb")
    assert magic.resolve_notebook_path() == "/auto/detected.ipynb"


def test_unresolvable_path_raises_usage_error_naming_the_fix(monkeypatch):
    monkeypatch.setattr(magic, "_autodetect_path", lambda: None)
    with pytest.raises(UsageError) as excinfo:
        magic.resolve_notebook_path()
    assert "%provenance_notebook" in str(excinfo.value)


def test_set_notebook_path_echoes_what_it_stored():
    assert "explicit.ipynb" in magic.set_notebook_path("explicit.ipynb")


# --- result formatting -------------------------------------------------------

def test_software_result_summarizes_injected_cell():
    call = {
        "name": "cite_software",
        "args": {"notebook_path": "paleoPCAlite.ipynb"},
        "result": ["pyleoclim", "pandas"],
    }
    out = magic._format_result(call)
    assert "paleoPCAlite.ipynb" in out
    assert "pyleoclim" in out
    assert "pandas" in out
    assert "run" in out.lower()  # tells the user to run the injected cell


def test_data_result_summarizes_injected_cells():
    call = {
        "name": "cite_data",
        "args": {"notebook_path": "paleoPCAlite.ipynb"},
        "result": [["filtered_df2", "LiPDGraph"], ["D", "PyLiPD"]],
    }
    out = magic._format_result(call)
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
    out = magic._format_result(call)
    assert "no datasets" in out.lower()
    assert "unsaved" in out.lower()


def test_empty_software_result_mentions_reading_from_disk():
    call = {"name": "cite_software", "args": {"notebook_path": "nb.ipynb"}, "result": []}
    assert "unsaved" in magic._format_result(call).lower()


# --- cite --------------------------------------------------------------------

def test_empty_request_raises_usage_error(monkeypatch):
    monkeypatch.setattr(magic, "_autodetect_path", lambda: "nb.ipynb")
    with pytest.raises(UsageError):
        magic.cite("   ")


def test_unroutable_request_explains_what_the_tools_cover(monkeypatch):
    monkeypatch.setattr(magic, "_autodetect_path", lambda: "nb.ipynb")
    monkeypatch.setattr(magic, "run", lambda request, notebook_path: [])
    out = magic.cite("what is the weather")
    assert "software" in out.lower() and "datasets" in out.lower()


def test_cite_passes_resolved_path_to_the_agent(monkeypatch):
    seen = {}

    def fake_run(request, notebook_path):
        seen["request"] = request
        seen["notebook_path"] = notebook_path
        return [{"name": "cite_software",
                 "args": {"notebook_path": notebook_path}, "result": ["pyleoclim"]}]

    monkeypatch.setattr(magic, "_autodetect_path", lambda: None)
    monkeypatch.setattr(magic, "run", fake_run)
    magic.set_notebook_path("explicit.ipynb")

    assert "pyleoclim" in magic.cite("cite the software")
    assert seen == {"request": "cite the software", "notebook_path": "explicit.ipynb"}


def test_multiple_tool_calls_are_joined():
    calls = [
        {"name": "cite_software", "args": {"notebook_path": "nb.ipynb"},
         "result": ["pyleoclim"]},
        {"name": "cite_data", "args": {"notebook_path": "nb.ipynb"},
         "result": [["D", "PyLiPD"]]},
    ]
    out = magic._format_results(calls)
    assert "pyleoclim" in out and "D (PyLiPD)" in out


def test_envelope_format_reports_warning_without_route():
    out = magic._format_results({
        "status": "warning",
        "decision": None,
        "dispatch": [],
        "verification": {"mutated": False},
        "warning": "The request was ambiguous.",
    })
    assert out.startswith("Warning:")
    assert "ambiguous" in out


def test_envelope_format_reports_static_verification():
    out = magic._format_results({
        "status": "ok",
        "decision": {"action": "cite"},
        "dispatch": [{
            "name": "cite_software",
            "args": {"notebook_path": "nb.ipynb"},
            "result": ["pyleoclim"],
        }],
        "verification": {
            "cells": [{"tool": "software", "injected": True}],
            "present": ["software"],
            "mutated": True,
            "runtime_unverified": False,
        },
    })
    assert "pyleoclim" in out
    assert "Static verification passed" in out


def test_unchanged_rerun_still_reports_verification_passed():
    """A re-run rewrites identical cells, so nothing is added but all is well."""
    out = magic._format_results({
        "status": "ok",
        "decision": {"action": "cite"},
        "dispatch": [{
            "name": "cite_software",
            "args": {"notebook_path": "nb.ipynb"},
            "result": ["pyleoclim"],
        }],
        "verification": {
            "cells": [],
            "present": ["software"],
            "mutated": False,
            "runtime_unverified": False,
        },
    })
    assert "Static verification passed" in out


# --- extension registration --------------------------------------------------

def test_load_ipython_extension_registers_the_magics():
    registered = []
    load_ipython_extension = magic.load_ipython_extension
    load_ipython_extension(type("Shell", (), {"register_magics": lambda self, m: registered.append(m)})())
    assert registered == [magic.ProvenanceMagics]


def test_magic_methods_exist():
    assert hasattr(magic.ProvenanceMagics, "provenance")
    assert hasattr(magic.ProvenanceMagics, "provenance_notebook")
