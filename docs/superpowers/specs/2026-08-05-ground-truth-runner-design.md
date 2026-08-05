# Ground-Truth Evaluation Runner Design

## Goal

Provide a repeatable, offline command that evaluates the current provenance
agent against the manually curated records in `benchmark/ground_truth/`.

## Scope

The runner will:

- load every ground-truth YAML record by default;
- resolve each record's repository-relative notebook path;
- extract imported software with `provenance_agent.notebook_io.parse_notebook`;
- detect source-backed datasets with the active deterministic
  `provenance_agent.dataset_detection.detect_datasets` API;
- compare predictions with expected software names and `[variable, tool]`
  dataset pairs;
- print per-notebook precision, recall, F1, and missing/unexpected values;
- print micro-averaged software, dataset, and combined corpus scores; and
- write a detailed JSON report to the ignored `benchmark/results/` directory.

The runner will not execute notebooks, call an LLM, modify notebooks or YAML
labels, inject citation cells, or create bibliography files.

## Interface

The executable module will be `benchmark/run_ground_truth.py`, invoked with
the repository environment's Python as:

```text
/opt/anaconda3/envs/lang/bin/python benchmark/run_ground_truth.py
```

Optional arguments:

- `--notebook TEXT` filters records whose notebook path contains `TEXT`;
- `--ground-truth PATH` changes the YAML directory; and
- `--output PATH` changes the JSON report path.

An empty filter result is an error. The default report path is
`benchmark/results/ground_truth_results.json`.

## Scoring

Software names are compared case-insensitively as sets. Dataset variables must
match exactly, while dataset tool names are compared case-insensitively. Duplicate
predictions count once. Each side uses the standard true-positive,
false-positive, false-negative precision/recall/F1 calculation; an empty
prediction against an empty expectation is a perfect match. The corpus totals
pool counts before calculating rates (micro-average).

Each notebook report will retain the raw expected and predicted values, scores,
and missing/unexpected values so a mismatch can be investigated without
rerunning the detector.

## Testing

Tests will cover the pure scoring conventions, YAML loading and ordering,
notebook filtering, report contents, and the command's no-write behavior for
source notebooks and ground truth. The existing full test suite remains the
final verification.
