"""
Benchmark runner: scores the agent's dependency attribution against
hand-curated ground truth, one YAML file per notebook.

Purpose:
    Produce a defensible "correctly attributed X% of dataset and software
    dependencies across N notebooks" number. For each notebook in
    benchmark/ground_truth/, the runner compares:
      - software side: parse_notebook() (AST import extraction) against the
        curated `software` list
      - data side: detect_datasets() (LLM detection) against the curated
        `datasets` [variable, tool] pairs, run over several trials because
        the LLM is nondeterministic even at temperature 0

Implementation:
    - Ground truth YAML shape: {notebook, software: [names],
      datasets: [{variable, tool}], notes?}. Paths are repo-relative.
    - score_sets / score_pairs reduce a prediction to true/false
      positive/negative counts and precision/recall/F1. Library and tool
      names match case-insensitively; dataset variable names match exactly
      (they must name a real kernel object).
    - mean_scores averages per-trial scores for one notebook; micro_average
      pools raw counts across notebooks before computing rates, so notebooks
      with many dependencies weigh proportionally more.
    - The combined headline metric pools software and data counts into one
      micro-averaged score.
    - Results are written as JSON to benchmark/results/ (untracked) and
      printed as a per-notebook table plus corpus aggregates.

Design decisions:
    - Detection is scored on the notebook's code (read_notebook_code), never
      by executing the notebook - retrieval is out of scope, so the
      benchmark needs only a Gemini API key and runs in seconds per trial.
    - Gemini calls are paced (_INTER_CALL_DELAY between calls) and wrapped in
      call_with_retry, because the free tier's requests-per-minute quota
      otherwise kills a corpus-wide run partway through with a 429.
    - Empty-vs-empty scores 1.0 across the board: a notebook with no dataset
      sources is attributed correctly when the agent finds none.
    - src/ is added to sys.path here (mirroring the tests) because the repo
      is not an installed package; workflow imports are deferred into
      benchmark_notebook so the pure scoring helpers import without a
      GOOGLE_API_KEY.
"""

import argparse
import json
import os
import sys
import time

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

_SCORE_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1")

# Seconds between consecutive Gemini calls, to stay under the free tier's
# requests-per-minute quota (~20 RPM at the time of writing).
_INTER_CALL_DELAY = 5.0


def _rates(tp: float, fp: float, fn: float) -> dict:
    """
    Computes precision/recall/F1 from raw counts.

    Args:
        tp: true positives (correctly attributed dependencies)
        fp: false positives (spurious attributions)
        fn: false negatives (missed dependencies)

    Returns:
        dict with tp, fp, fn, precision, recall, f1. An empty prediction
        against an empty expectation scores 1.0 (finding nothing was correct).
    """
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def score_sets(predicted: set[str], expected: set[str]) -> dict:
    """
    Scores a predicted set of names against the expected set.

    Args:
        predicted: names the agent reported (e.g. imported libraries)
        expected: ground-truth names; comparison is case-insensitive

    Returns:
        score dict (see _rates)
    """
    pred = {name.lower() for name in predicted}
    exp = {name.lower() for name in expected}
    return _rates(len(pred & exp), len(pred - exp), len(exp - pred))


def score_pairs(predicted: list[list[str]], expected: list[dict]) -> dict:
    """
    Scores detected [variable, tool] pairs against ground-truth dataset entries.

    Args:
        predicted: [variable, tool] pairs from detect_datasets(); duplicates
            are collapsed
        expected: ground-truth dicts with "variable" and "tool" keys

    Returns:
        score dict (see _rates). Variables match exactly, tools
        case-insensitively.
    """
    pred = {(var, tool.lower()) for var, tool in predicted}
    exp = {(entry["variable"], entry["tool"].lower()) for entry in expected}
    return _rates(len(pred & exp), len(pred - exp), len(exp - pred))


def mean_scores(trials: list[dict]) -> dict:
    """
    Averages score dicts field-wise across detection trials of one notebook.

    Args:
        trials: per-trial score dicts (all with the standard score fields)

    Returns:
        dict with each field's mean across the trials
    """
    return {field: sum(t[field] for t in trials) / len(trials)
            for field in _SCORE_FIELDS}


def micro_average(counts: list[dict]) -> dict:
    """
    Pools raw counts across notebooks, then computes corpus-level rates.

    Args:
        counts: per-notebook score dicts (only tp/fp/fn are read, so
            fractional trial-averaged counts are fine)

    Returns:
        score dict over the pooled counts (see _rates)
    """
    return _rates(sum(c["tp"] for c in counts),
                  sum(c["fp"] for c in counts),
                  sum(c["fn"] for c in counts))


def load_ground_truth(directory: str) -> list[dict]:
    """
    Loads every ground-truth YAML in a directory.

    Args:
        directory: path containing one .yml file per notebook

    Returns:
        the parsed ground-truth dicts, sorted by their notebook path
    """
    truths = []
    for name in os.listdir(directory):
        if name.endswith((".yml", ".yaml")):
            with open(os.path.join(directory, name)) as f:
                truths.append(yaml.safe_load(f))
    return sorted(truths, key=lambda gt: gt["notebook"])


def call_with_retry(fn, attempts: int = 5, base_delay: float = 30.0,
                    sleep=time.sleep):
    """
    Calls fn(), retrying with exponential backoff on API quota errors.

    The Gemini free tier enforces a small requests-per-minute quota, so a
    corpus-wide benchmark run must absorb 429s rather than crash mid-run.
    Only quota errors (429 / RESOURCE_EXHAUSTED in the message) are retried;
    anything else propagates immediately.

    Args:
        fn: zero-argument callable to invoke
        attempts: maximum number of tries before giving up
        base_delay: first backoff delay in seconds; doubles per retry
        sleep: sleep function (injectable for tests)

    Returns:
        fn()'s return value
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            message = str(exc)
            quota = "429" in message or "RESOURCE_EXHAUSTED" in message
            if not quota or attempt == attempts - 1:
                raise
            sleep(base_delay * 2 ** attempt)


def benchmark_notebook(gt: dict, trials: int) -> dict:
    """
    Runs both attribution workflows on one notebook and scores them.

    Args:
        gt: the notebook's ground-truth dict
        trials: how many times to run LLM dataset detection

    Returns:
        dict with the notebook path, the software score, per-trial data
        scores and their mean, and the raw predictions for inspection
    """
    from notebook_parser import parse_notebook, read_notebook_code
    from dataset_detection import detect_datasets

    path = os.path.join(_REPO_ROOT, gt["notebook"])
    libraries = parse_notebook(path)
    software = score_sets(set(libraries), set(gt["software"]))

    code = read_notebook_code(path)
    data_trials, predictions = [], []
    for trial in range(trials):
        if trial:
            time.sleep(_INTER_CALL_DELAY)
        pairs = call_with_retry(lambda: detect_datasets(code))
        predictions.append(pairs)
        data_trials.append(score_pairs(pairs, gt["datasets"]))

    return {
        "notebook": gt["notebook"],
        "software": software,
        "software_predicted": libraries,
        "data": mean_scores(data_trials),
        "data_trials": data_trials,
        "data_predicted": predictions,
    }


def _fmt(score: dict) -> str:
    """Formats one score dict as 'P=0.92 R=1.00 F1=0.96' for the table."""
    return (f"P={score['precision']:.2f} R={score['recall']:.2f} "
            f"F1={score['f1']:.2f}")


def main() -> None:
    """
    Runs the benchmark over every ground-truth notebook and reports results.

    Prints a per-notebook table plus software, data, and combined
    micro-averages, and writes the full detail to a JSON file.
    """
    parser = argparse.ArgumentParser(description="Score dependency attribution.")
    parser.add_argument("--trials", type=int, default=3,
                        help="LLM detection trials per notebook (default 3)")
    parser.add_argument("--notebook", default=None,
                        help="only run notebooks whose path contains this substring")
    parser.add_argument("--ground-truth",
                        default=os.path.join(_REPO_ROOT, "benchmark", "ground_truth"),
                        help="directory of ground-truth YAML files")
    parser.add_argument("--output",
                        default=os.path.join(_REPO_ROOT, "benchmark", "results",
                                             "benchmark_results.json"),
                        help="where to write the JSON results")
    args = parser.parse_args()

    truths = load_ground_truth(args.ground_truth)
    if args.notebook:
        truths = [gt for gt in truths if args.notebook in gt["notebook"]]
    if not truths:
        sys.exit("No ground-truth files matched.")

    results = []
    for i, gt in enumerate(truths):
        if i:
            time.sleep(_INTER_CALL_DELAY)
        result = benchmark_notebook(gt, args.trials)
        results.append(result)
        name = os.path.basename(result["notebook"])
        print(f"{name:<45} software {_fmt(result['software'])}   "
              f"data {_fmt(result['data'])}")

    software_total = micro_average([r["software"] for r in results])
    data_total = micro_average([r["data"] for r in results])
    combined = micro_average([software_total, data_total])

    print(f"\n{'MICRO-AVERAGE (software)':<45} {_fmt(software_total)}")
    print(f"{'MICRO-AVERAGE (data, mean of trials)':<45} {_fmt(data_total)}")
    print(f"{'COMBINED (all dependencies)':<45} {_fmt(combined)}")
    print(f"\n{len(results)} notebooks, {args.trials} detection trials each")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"notebooks": results, "software_total": software_total,
                   "data_total": data_total, "combined": combined}, f, indent=2)
    print(f"Details written to {args.output}")


if __name__ == "__main__":
    main()
