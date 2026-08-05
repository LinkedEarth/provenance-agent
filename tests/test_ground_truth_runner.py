"""
Tests for the repeatable ground-truth evaluation runner.

The scoring and loading tests are fully offline. Detector calls are patched in
the evaluation and CLI tests so those tests validate report structure and
filesystem behavior without relying on a particular notebook's predictions.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml


BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark"
RUNNER_PATH = BENCHMARK / "run_ground_truth.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_ground_truth", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def test_score_sets_case_insensitive_and_reports_misses():
    result = runner.score_sets({"Pandas", "scipy"}, {"pandas", "numpy"})

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_score_pairs_deduplicates_and_keeps_variable_case_exact():
    result = runner.score_pairs(
        [["D", "pylipd"], ["D", "PyLiPD"], ["d", "PyLiPD"]],
        [{"variable": "D", "tool": "PyLiPD"}],
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_empty_prediction_and_expectation_is_perfect():
    assert runner.score_sets(set(), set())["f1"] == 1.0
    assert runner.score_pairs([], [])["f1"] == 1.0


def test_micro_average_pools_counts_before_rates():
    result = runner.micro_average([
        {"tp": 1, "fp": 0, "fn": 0},
        {"tp": 1, "fp": 2, "fn": 1},
    ])

    assert result["tp"] == 2
    assert result["fp"] == 2
    assert result["fn"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(2 / 3)


def _write_truth(directory: Path, name: str, record: dict) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return path


def test_load_ground_truth_reads_yml_and_yaml_sorted_by_notebook(tmp_path):
    _write_truth(
        tmp_path,
        "z.yml",
        {"notebook": "notebooks/b.ipynb", "software": [], "datasets": []},
    )
    _write_truth(
        tmp_path,
        "a.yaml",
        {"notebook": "notebooks/a.ipynb", "software": [], "datasets": []},
    )

    records = runner.load_ground_truth(tmp_path)

    assert [record["notebook"] for record in records] == [
        "notebooks/a.ipynb",
        "notebooks/b.ipynb",
    ]


@pytest.mark.parametrize(
    ("record", "field"),
    [
        ({"software": [], "datasets": []}, "notebook"),
        ({"notebook": "nb.ipynb", "software": {}, "datasets": []}, "software"),
        (
            {
                "notebook": "nb.ipynb",
                "software": [],
                "datasets": [{"variable": "df"}],
            },
            "datasets[0].tool",
        ),
    ],
)
def test_load_ground_truth_rejects_malformed_records(tmp_path, record, field):
    path = _write_truth(tmp_path, "bad.yml", record)

    with pytest.raises(ValueError, match=re.escape(field)):
        runner.load_ground_truth(tmp_path)


def test_evaluate_notebook_reports_predictions_and_mismatches(monkeypatch, tmp_path):
    record = {
        "notebook": "notebooks/example.ipynb",
        "software": ["numpy", "pandas"],
        "datasets": [{"variable": "df", "tool": "LiPDGraph"}],
    }
    monkeypatch.setattr(runner, "parse_notebook", lambda path: ["scipy", "numpy"])
    monkeypatch.setattr(
        runner,
        "detect_datasets",
        lambda path: [["extra", "xarray"], ["df", "lipdgraph"]],
    )

    result = runner.evaluate_notebook(record, tmp_path)

    assert result["notebook"] == "notebooks/example.ipynb"
    assert result["software"]["expected"] == ["numpy", "pandas"]
    assert result["software"]["predicted"] == ["numpy", "scipy"]
    assert result["software"]["missing"] == ["pandas"]
    assert result["software"]["unexpected"] == ["scipy"]
    assert result["software"]["score"]["f1"] == pytest.approx(0.5)
    assert result["data"]["expected"] == [["df", "LiPDGraph"]]
    assert result["data"]["predicted"] == [["df", "lipdgraph"], ["extra", "xarray"]]
    assert result["data"]["missing"] == []
    assert result["data"]["unexpected"] == [["extra", "xarray"]]
    assert result["data"]["score"]["f1"] == pytest.approx(2 / 3)


def test_evaluate_records_pools_software_and_data_counts(monkeypatch, tmp_path):
    records = [
        {"notebook": "a.ipynb", "software": ["numpy"], "datasets": []},
        {
            "notebook": "b.ipynb",
            "software": ["numpy", "pandas"],
            "datasets": [{"variable": "df", "tool": "LiPDGraph"}],
        },
    ]

    monkeypatch.setattr(runner, "parse_notebook", lambda path: ["numpy"])
    monkeypatch.setattr(
        runner,
        "detect_datasets",
        lambda path: [["df", "LiPDGraph"]] if path.endswith("b.ipynb") else [],
    )

    report = runner.evaluate_records(records, tmp_path)

    assert [item["notebook"] for item in report["notebooks"]] == ["a.ipynb", "b.ipynb"]
    assert report["software_total"]["tp"] == 2
    assert report["software_total"]["fn"] == 1
    assert report["data_total"]["tp"] == 1
    assert report["data_total"]["fn"] == 0
    assert report["combined"]["tp"] == 3
    assert report["combined"]["fn"] == 1


def test_main_filters_records_writes_json_and_does_not_mutate_inputs(
    monkeypatch, tmp_path, capsys
):
    ground_truth = tmp_path / "ground_truth"
    ground_truth.mkdir()
    records = [
        {"notebook": "a.ipynb", "software": [], "datasets": []},
        {"notebook": "b.ipynb", "software": [], "datasets": []},
    ]
    paths = [
        _write_truth(ground_truth, "a.yml", records[0]),
        _write_truth(ground_truth, "b.yml", records[1]),
    ]
    notebooks = []
    for record in records:
        notebook = tmp_path / record["notebook"]
        notebook.write_bytes(b"original notebook bytes")
        notebooks.append(notebook)
    before_yaml = [path.read_bytes() for path in paths]
    before_notebooks = [path.read_bytes() for path in notebooks]
    output = tmp_path / "results" / "report.json"

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "parse_notebook", lambda path: [])
    monkeypatch.setattr(runner, "detect_datasets", lambda path: [])

    status = runner.main(
        [
            "--ground-truth",
            str(ground_truth),
            "--output",
            str(output),
            "--notebook",
            "b.ipynb",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(output.read_text())
    assert status == 0
    assert "b.ipynb" in captured.out
    assert "a.ipynb" not in captured.out
    assert [item["notebook"] for item in report["notebooks"]] == ["b.ipynb"]
    assert set(("software_total", "data_total", "combined")) <= report.keys()
    assert [path.read_bytes() for path in paths] == before_yaml
    assert [path.read_bytes() for path in notebooks] == before_notebooks


def test_main_returns_nonzero_when_filter_matches_nothing(tmp_path):
    ground_truth = tmp_path / "ground_truth"
    ground_truth.mkdir()
    _write_truth(
        ground_truth,
        "one.yml",
        {"notebook": "one.ipynb", "software": [], "datasets": []},
    )

    status = runner.main(
        ["--ground-truth", str(ground_truth), "--notebook", "missing.ipynb"]
    )

    assert status != 0
