"""
Unit tests for agent.py.

The routing step itself needs Gemini (network), so these tests cover the offline
pieces: system-prompt content, message construction, the tool registry, and that
run() dispatches a routed tool call to the correct tool (route is monkeypatched so
no model call happens). The software dispatch path runs fully offline (it injects
a metadata cell, reading imports and writing to a tmp path - no Gemini).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sample.ipynb")


def test_system_prompt_mentions_both_tools():
    assert "cite_software" in agent.SYSTEM_PROMPT
    assert "cite_data" in agent.SYSTEM_PROMPT


def test_build_messages_injects_notebook_and_request():
    msgs = agent.build_messages("cite the software", "nb.ipynb")
    assert "nb.ipynb" in msgs[0].content        # system message carries the path
    assert "cite the software" in msgs[1].content  # human message carries the request


def test_tools_by_name_has_both():
    assert set(agent._TOOLS_BY_NAME) == {"cite_software", "cite_data"}


def test_run_dispatches_to_chosen_tool(monkeypatch, tmp_path):
    out_nb = str(tmp_path / "out.ipynb")
    monkeypatch.setattr(
        agent, "route",
        lambda request, notebook_path: [
            {"name": "cite_software",
             "args": {"notebook_path": SAMPLE, "libraries": "pyleoclim",
                      "output_path": out_nb}}
        ],
    )
    out = agent.run("cite pyleoclim", SAMPLE)
    assert len(out) == 1
    assert out[0]["name"] == "cite_software"
    assert out[0]["result"] == ["pyleoclim"]


def test_run_skips_unknown_tool(monkeypatch):
    monkeypatch.setattr(
        agent, "route",
        lambda request, notebook_path: [{"name": "nonexistent", "args": {}}],
    )
    assert agent.run("whatever", SAMPLE) == []
