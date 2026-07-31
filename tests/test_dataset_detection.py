"""
test_dataset_detection.py

Purpose:
    Unit tests for the pure legacy helpers in dataset_detection.py: building the
    old LLM detection prompt (build_detection_prompt), parsing model replies into
    [variable, tool] pairs (parse_detection_response), and exercising the active
    deterministic notebook-path entry point.

Implementation:
    The legacy prompt/parser tests operate on plain strings. The active
    detect_datasets test writes a small notebook and verifies deterministic
    source-to-analysis tracing without network or LLM calls.

Design Decisions:
    - parse_detection_response must tolerate the ways an LLM wraps JSON: bare
      arrays, markdown code fences, and surrounding prose. Each variant gets its
      own test so a failure pinpoints which wrapping broke.
    - Malformed output returns [] rather than raising, so the data workflow can
      degrade gracefully instead of crashing.
"""

import os
import sys

import nbformat
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset_detection import (
    build_detection_prompt,
    detect_datasets,
    parse_detection_response,
)


# --- parse_detection_response ------------------------------------------------

def test_parses_bare_json_list():
    text = '[["df_res", "LiPDGraph"], ["D", "PyLiPD"]]'
    assert parse_detection_response(text) == [["df_res", "LiPDGraph"], ["D", "PyLiPD"]]


def test_parses_json_wrapped_in_markdown_fence():
    text = '```json\n[["ds", "PyleoTUPS"]]\n```'
    assert parse_detection_response(text) == [["ds", "PyleoTUPS"]]


def test_parses_json_wrapped_in_bare_fence():
    text = '```\n[["df_res", "LiPDGraph"]]\n```'
    assert parse_detection_response(text) == [["df_res", "LiPDGraph"]]


def test_empty_array_returns_empty_list():
    assert parse_detection_response("[]") == []


def test_ignores_surrounding_prose():
    text = 'Here are the pairs I found:\n[["iso_ds", "xarray"]]\nHope that helps.'
    assert parse_detection_response(text) == [["iso_ds", "xarray"]]


def test_malformed_output_returns_empty_list():
    assert parse_detection_response("I could not find any datasets.") == []


def test_non_pair_items_are_dropped():
    text = '[["D", "PyLiPD"], ["only_one"], ["a", "b", "c"]]'
    assert parse_detection_response(text) == [["D", "PyLiPD"]]


# --- build_detection_prompt --------------------------------------------------

def test_prompt_embeds_notebook_code():
    code = "df_res = pd.read_csv(io.StringIO(response.text))"
    assert code in build_detection_prompt(code)


def test_prompt_includes_output_format_instruction():
    prompt = build_detection_prompt("x = 1")
    assert "JSON list of [variable, tool] pairs" in prompt


def test_detect_datasets_uses_deterministic_notebook_path(tmp_path):
    notebook = tmp_path / "analysis.ipynb"
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "import io\n"
        "import pandas as pd\n"
        "import requests\n"
        "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
        "response = requests.post(url, data={'query': 'SELECT ...'})\n"
        "df_res = pd.read_csv(io.StringIO(response.text))\n"
        "filtered_df2 = df_res[df_res['varID'].notna()]\n"
        "filtered_df2.pca()\n"
    )])
    with open(notebook, "w") as handle:
        nbformat.write(nb, handle)

    assert detect_datasets(str(notebook)) == [["filtered_df2", "LiPDGraph"]]


def test_detect_datasets_warns_for_analysis_without_source_lineage(tmp_path):
    notebook = tmp_path / "unknown_loader.ipynb"
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "import custom_loader\n"
        "df = custom_loader.load_data('remote://example')\n"
        "result = df.pca()\n"
    )])
    with open(notebook, "w") as handle:
        nbformat.write(nb, handle)

    with pytest.warns(UserWarning, match="unsupported loader"):
        assert detect_datasets(str(notebook)) == []
