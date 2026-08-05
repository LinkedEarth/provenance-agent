"""
Offline evaluator for the manually curated notebook ground truth.

Purpose:
    Measure how closely the current provenance-agent software and deterministic
    dataset detectors match the records in ``benchmark/ground_truth/``. The
    command is intentionally separate from citation generation: it reads
    notebooks, computes predictions, compares them with YAML, prints a compact
    summary, and writes an inspectable JSON report.

Implementation:
    ``score_sets`` compares imported library names, while ``score_pairs``
    compares ``[variable, tool]`` dataset pairs. ``load_ground_truth`` validates
    and orders YAML records. Later orchestration functions run those scorers on
    repository-relative notebook paths and aggregate raw counts.

Design decisions:
    - The active path-based detector is evaluated directly; notebooks are never
      executed and the deprecated LLM detector is never called.
    - This module does not invoke citation workflows, write notebooks, or
      create bibliography files.
    - Software and dataset tools are case-insensitive, but dataset variable
      names are exact because they identify notebook objects.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from provenance_agent.dataset_detection import detect_datasets
from provenance_agent.notebook_io import parse_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]


def _rates(tp: float, fp: float, fn: float) -> dict[str, float]:
    """
    Computes precision, recall, and F1 from true/false positive counts.

    Args:
        tp: number of predictions that match expected values
        fp: number of predictions that were not expected
        fn: number of expected values that were not predicted

    Returns:
        A score mapping containing the three counts and three rates. An empty
        prediction against an empty expectation is scored as perfect.
    """
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_sets(predicted: Iterable[str], expected: Iterable[str]) -> dict[str, float]:
    """
    Scores predicted and expected names as case-insensitive sets.

    Args:
        predicted: names reported by the software detector
        expected: names listed in the ground truth

    Returns:
        True/false positive counts and precision, recall, and F1 rates.
    """
    predicted_set = {name.casefold() for name in predicted}
    expected_set = {name.casefold() for name in expected}
    return _rates(
        len(predicted_set & expected_set),
        len(predicted_set - expected_set),
        len(expected_set - predicted_set),
    )


def score_pairs(
    predicted: Iterable[Sequence[str]],
    expected: Iterable[Mapping[str, str]],
) -> dict[str, float]:
    """
    Scores dataset variable/tool pairs with exact variables and folded tools.

    Args:
        predicted: ``[variable, tool]`` pairs reported by dataset detection
        expected: ground-truth mappings with ``variable`` and ``tool`` keys

    Returns:
        True/false positive counts and precision, recall, and F1 rates.
    """
    predicted_set = {(pair[0], pair[1].casefold()) for pair in predicted}
    expected_set = {
        (entry["variable"], entry["tool"].casefold()) for entry in expected
    }
    return _rates(
        len(predicted_set & expected_set),
        len(predicted_set - expected_set),
        len(expected_set - predicted_set),
    )


def micro_average(scores: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """
    Pools raw counts across records before calculating corpus-level rates.

    Args:
        scores: score mappings containing at least ``tp``, ``fp``, and ``fn``

    Returns:
        One score mapping calculated from the pooled counts.
    """
    scores = list(scores)
    return _rates(
        sum(score["tp"] for score in scores),
        sum(score["fp"] for score in scores),
        sum(score["fn"] for score in scores),
    )


def _invalid_record(path: Path, field: str, detail: str) -> ValueError:
    """Builds a consistent validation error for one ground-truth YAML file."""
    return ValueError(f"{path}: invalid {field}: {detail}")


def _validate_record(path: Path, record: Any) -> dict[str, Any]:
    """Validates one parsed ground-truth mapping and returns it unchanged."""
    if not isinstance(record, dict):
        raise _invalid_record(path, "record", "expected a mapping")

    notebook = record.get("notebook")
    if not isinstance(notebook, str) or not notebook:
        raise _invalid_record(path, "notebook", "expected a non-empty string")

    software = record.get("software")
    if not isinstance(software, list):
        raise _invalid_record(path, "software", "expected a list")
    for index, name in enumerate(software):
        if not isinstance(name, str):
            raise _invalid_record(path, f"software[{index}]", "expected a string")

    datasets = record.get("datasets")
    if not isinstance(datasets, list):
        raise _invalid_record(path, "datasets", "expected a list")
    for index, entry in enumerate(datasets):
        field = f"datasets[{index}]"
        if not isinstance(entry, dict):
            raise _invalid_record(path, field, "expected a mapping")
        for key in ("variable", "tool"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise _invalid_record(path, f"{field}.{key}", "expected a non-empty string")

    return record


def load_ground_truth(directory: str | Path) -> list[dict[str, Any]]:
    """
    Loads and validates all YAML ground-truth records in a directory.

    Args:
        directory: directory containing ``.yml`` or ``.yaml`` records

    Returns:
        Records sorted by their repository-relative notebook path.

    Raises:
        ValueError: if a YAML document or required field has the wrong shape
    """
    directory_path = Path(directory)
    paths = sorted((*directory_path.glob("*.yml"), *directory_path.glob("*.yaml")))
    records = []
    for path in paths:
        try:
            record = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise _invalid_record(path, "record", f"invalid YAML ({exc})") from exc
        records.append(_validate_record(path, record))
    return sorted(records, key=lambda record: record["notebook"])


def _sorted_names(names: Iterable[str]) -> list[str]:
    """Returns unique names in stable, case-insensitive order."""
    unique = set(names)
    return sorted(unique, key=lambda name: (name.casefold(), name))


def _name_mismatches(predicted: Iterable[str], expected: Iterable[str]) -> tuple[list[str], list[str]]:
    """Returns expected names missed and predicted names not expected."""
    predicted_by_key = {name.casefold(): name for name in predicted}
    expected_by_key = {name.casefold(): name for name in expected}
    missing = [
        expected_by_key[key]
        for key in sorted(expected_by_key.keys() - predicted_by_key.keys())
    ]
    unexpected = [
        predicted_by_key[key]
        for key in sorted(predicted_by_key.keys() - expected_by_key.keys())
    ]
    return missing, unexpected


def _pair_list(entries: Iterable[Mapping[str, str] | Sequence[str]]) -> list[list[str]]:
    """Converts expected mappings or detector pairs to sorted JSON-safe pairs."""
    pairs = {
        (entry["variable"], entry["tool"])
        if isinstance(entry, Mapping)
        else (entry[0], entry[1])
        for entry in entries
    }
    return [
        [variable, tool]
        for variable, tool in sorted(pairs, key=lambda pair: (pair[0], pair[1].casefold(), pair[1]))
    ]


def _pair_mismatches(
    predicted: Iterable[Sequence[str]],
    expected: Iterable[Mapping[str, str]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Returns missing and unexpected dataset pairs using scoring normalization."""
    predicted_by_key = {
        (pair[0], pair[1].casefold()): [pair[0], pair[1]] for pair in predicted
    }
    expected_by_key = {
        (entry["variable"], entry["tool"].casefold()): [
            entry["variable"],
            entry["tool"],
        ]
        for entry in expected
    }
    missing = [
        expected_by_key[key]
        for key in sorted(expected_by_key.keys() - predicted_by_key.keys())
    ]
    unexpected = [
        predicted_by_key[key]
        for key in sorted(predicted_by_key.keys() - expected_by_key.keys())
    ]
    return missing, unexpected


def evaluate_notebook(record: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """
    Runs both active detectors for one ground-truth record and compares them.

    Args:
        record: validated ground-truth mapping
        repo_root: repository root used to resolve the record's notebook path

    Returns:
        JSON-serializable notebook predictions, mismatches, and scores
    """
    notebook = record["notebook"]
    notebook_path = repo_root / notebook
    predicted_software = parse_notebook(str(notebook_path))
    predicted_data = detect_datasets(str(notebook_path))
    expected_software = record["software"]
    expected_data = record["datasets"]

    software_missing, software_unexpected = _name_mismatches(
        predicted_software, expected_software
    )
    data_missing, data_unexpected = _pair_mismatches(predicted_data, expected_data)

    return {
        "notebook": notebook,
        "software": {
            "expected": _sorted_names(expected_software),
            "predicted": _sorted_names(predicted_software),
            "missing": software_missing,
            "unexpected": software_unexpected,
            "score": score_sets(predicted_software, expected_software),
        },
        "data": {
            "expected": _pair_list(expected_data),
            "predicted": _pair_list(predicted_data),
            "missing": data_missing,
            "unexpected": data_unexpected,
            "score": score_pairs(predicted_data, expected_data),
        },
    }


def evaluate_records(records: list[dict[str, Any]], repo_root: Path) -> dict[str, Any]:
    """
    Evaluates records and computes software, data, and combined micro-averages.

    Args:
        records: validated ground-truth mappings
        repo_root: repository root used to resolve notebook paths

    Returns:
        Complete JSON-serializable corpus report
    """
    notebooks = [evaluate_notebook(record, repo_root) for record in records]
    software_total = micro_average(item["software"]["score"] for item in notebooks)
    data_total = micro_average(item["data"]["score"] for item in notebooks)
    return {
        "notebooks": notebooks,
        "software_total": software_total,
        "data_total": data_total,
        "combined": micro_average((software_total, data_total)),
    }


def _format_score(score: Mapping[str, float]) -> str:
    """Formats a score mapping for the command-line summary."""
    return (
        f"P={score['precision']:.2f} "
        f"R={score['recall']:.2f} "
        f"F1={score['f1']:.2f}"
    )


def _print_mismatches(label: str, section: Mapping[str, Any]) -> None:
    """Prints missing and unexpected values for one notebook section."""
    if section["missing"]:
        print(f"  {label} missing: {json.dumps(section['missing'])}")
    if section["unexpected"]:
        print(f"  {label} unexpected: {json.dumps(section['unexpected'])}")


def _print_report(report: Mapping[str, Any], output: Path) -> None:
    """Prints per-notebook scores, mismatches, and aggregate scores."""
    for item in report["notebooks"]:
        print(
            f"{item['notebook']}: "
            f"software {_format_score(item['software']['score'])}; "
            f"data {_format_score(item['data']['score'])}"
        )
        _print_mismatches("software", item["software"])
        _print_mismatches("data", item["data"])

    print(f"\nMICRO-AVERAGE (software): {_format_score(report['software_total'])}")
    print(f"MICRO-AVERAGE (data): {_format_score(report['data_total'])}")
    print(f"COMBINED: {_format_score(report['combined'])}")
    print(f"\n{len(report['notebooks'])} notebooks evaluated")
    print(f"Details written to {output}")


def main(argv: Sequence[str] | None = None) -> int:
    """
    Runs the ground-truth evaluator from command-line arguments.

    Args:
        argv: optional argument list; ``None`` uses process command-line args

    Returns:
        zero on a completed evaluation, or two when inputs select no records
    """
    parser = argparse.ArgumentParser(
        description="Compare provenance-agent predictions with notebook ground truth."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REPO_ROOT / "benchmark" / "ground_truth",
        help="directory containing ground-truth YAML files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmark" / "results" / "ground_truth_results.json",
        help="JSON report path",
    )
    parser.add_argument(
        "--notebook",
        default=None,
        help="only evaluate records whose notebook path contains this text",
    )
    args = parser.parse_args(argv)

    try:
        records = load_ground_truth(args.ground_truth)
    except (OSError, ValueError) as exc:
        print(f"Unable to load ground truth: {exc}", file=sys.stderr)
        return 2

    if args.notebook:
        records = [record for record in records if args.notebook in record["notebook"]]
    if not records:
        print("No ground-truth files matched the requested filter.", file=sys.stderr)
        return 2

    report = evaluate_records(records, REPO_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
