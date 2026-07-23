"""
test_dataset_detection.py

Purpose:
    Unit tests for the pure logic in dataset_detection.py: building the LLM
    detection prompt (build_detection_prompt) and parsing the model's reply into
    [variable, tool] pairs (parse_detection_response).

Implementation:
    These tests operate on plain strings with no network or LLM calls, so they
    run fast and deterministically. The live Gemini call in detect_datasets is
    exercised manually against a real notebook (it hits an external, non-
    deterministic service), so it is intentionally not covered here.

Design Decisions:
    - parse_detection_response must tolerate the ways an LLM wraps JSON: bare
      arrays, markdown code fences, and surrounding prose. Each variant gets its
      own test so a failure pinpoints which wrapping broke.
    - Malformed output returns [] rather than raising, so the data workflow can
      degrade gracefully instead of crashing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset_detection import build_detection_prompt, parse_detection_response


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
