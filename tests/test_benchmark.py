"""
test_benchmark.py

Purpose:
    Unit tests for the pure scoring logic in benchmark/run_benchmark.py:
    set-based scoring of software imports (score_sets), pair-based scoring of
    dataset detections (score_pairs), averaging across LLM trials
    (mean_scores), micro-averaging across notebooks (micro_average), and
    ground-truth loading (load_ground_truth).

Implementation:
    All scoring functions are pure - no LLM calls, no notebook parsing, no
    network. load_ground_truth is exercised against YAML files written to
    pytest's tmp_path. The benchmark runner's main() (which calls the real
    parse_notebook / detect_datasets) is out of scope; it is exercised by
    running the benchmark itself.

Design Decisions:
    - Precision/recall/F1 conventions are pinned by tests: an empty prediction
      against an empty expectation scores 1.0 across the board (the tool was
      right to find nothing), while any miss or spurious hit is penalized.
    - Tool names match case-insensitively ("pylipd" == "PyLiPD") because the
      LLM's casing varies; variable names match exactly because they must name
      a real kernel object.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))

from run_benchmark import (
    load_ground_truth,
    mean_scores,
    micro_average,
    score_pairs,
    score_sets,
)


# ---------------------------------------------------------------- score_sets

def test_score_sets_perfect_match():
    scores = score_sets({"pandas", "numpy"}, {"pandas", "numpy"})
    assert scores["tp"] == 2 and scores["fp"] == 0 and scores["fn"] == 0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_score_sets_partial_match():
    # predicted numpy (right), scipy (spurious); missed pandas
    scores = score_sets({"numpy", "scipy"}, {"numpy", "pandas"})
    assert scores["tp"] == 1 and scores["fp"] == 1 and scores["fn"] == 1
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5


def test_score_sets_both_empty_is_perfect():
    scores = score_sets(set(), set())
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_score_sets_empty_prediction_misses_everything():
    scores = score_sets(set(), {"pandas"})
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0


def test_score_sets_case_insensitive():
    scores = score_sets({"Pandas"}, {"pandas"})
    assert scores["tp"] == 1 and scores["fp"] == 0 and scores["fn"] == 0


# --------------------------------------------------------------- score_pairs

def test_score_pairs_tool_case_insensitive_variable_exact():
    predicted = [["D", "pylipd"], ["df_res", "LiPDGraph"]]
    expected = [
        {"variable": "D", "tool": "PyLiPD"},
        {"variable": "df_res", "tool": "LiPDGraph"},
    ]
    scores = score_pairs(predicted, expected)
    assert scores["tp"] == 2 and scores["fp"] == 0 and scores["fn"] == 0
    assert scores["f1"] == 1.0


def test_score_pairs_wrong_variable_case_is_a_miss():
    scores = score_pairs([["d", "PyLiPD"]], [{"variable": "D", "tool": "PyLiPD"}])
    assert scores["tp"] == 0 and scores["fp"] == 1 and scores["fn"] == 1


def test_score_pairs_wrong_tool_is_a_miss():
    scores = score_pairs([["D", "PyleoTUPS"]], [{"variable": "D", "tool": "PyLiPD"}])
    assert scores["tp"] == 0 and scores["fp"] == 1 and scores["fn"] == 1


def test_score_pairs_duplicate_predictions_count_once():
    predicted = [["D", "PyLiPD"], ["D", "PyLiPD"]]
    scores = score_pairs(predicted, [{"variable": "D", "tool": "PyLiPD"}])
    assert scores["tp"] == 1 and scores["fp"] == 0 and scores["fn"] == 0


def test_score_pairs_empty_expectation_penalizes_spurious_hits():
    scores = score_pairs([["ds", "xarray"]], [])
    assert scores["fp"] == 1
    assert scores["precision"] == 0.0


# --------------------------------------------------- mean_scores / micro_average

def test_mean_scores_averages_every_field():
    trials = [
        {"tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5},
    ]
    mean = mean_scores(trials)
    assert mean["tp"] == 1.5 and mean["fp"] == 0.5 and mean["fn"] == 0.5
    assert mean["precision"] == 0.75
    assert mean["f1"] == 0.75


def test_micro_average_pools_counts_before_computing_rates():
    # notebook A: 1 TP; notebook B: 1 TP, 2 FP. Pooled precision = 2/4,
    # NOT the mean of per-notebook precisions (1.0 and 1/3).
    per_notebook = [
        {"tp": 1, "fp": 0, "fn": 0},
        {"tp": 1, "fp": 2, "fn": 1},
    ]
    micro = micro_average(per_notebook)
    assert micro["tp"] == 2 and micro["fp"] == 2 and micro["fn"] == 1
    assert micro["precision"] == pytest.approx(0.5)
    assert micro["recall"] == pytest.approx(2 / 3)


# ----------------------------------------------------------- load_ground_truth

def test_load_ground_truth_reads_and_sorts_yaml_files(tmp_path):
    a = {
        "notebook": "notebooks/b.ipynb",
        "software": ["pandas"],
        "datasets": [{"variable": "df", "tool": "LiPDGraph"}],
    }
    b = {"notebook": "notebooks/a.ipynb", "software": [], "datasets": []}
    (tmp_path / "b.yml").write_text(yaml.safe_dump(a))
    (tmp_path / "a.yml").write_text(yaml.safe_dump(b))

    loaded = load_ground_truth(str(tmp_path))

    assert [gt["notebook"] for gt in loaded] == ["notebooks/a.ipynb", "notebooks/b.ipynb"]
    assert loaded[1]["datasets"][0]["variable"] == "df"
