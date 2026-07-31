# Deterministic Dataset Detection Design

**Status:** Implemented as a standalone detector; workflow integration remains a
separate transition task.

## Goal

Add a deterministic, source-aware dataset detector that returns the same
`[variable, tool]` pairs as the existing LLM pathway's public contract, while
preserving the LLM detector and avoiding a public pathway selector during
validation.
Only dataset sources whose data reaches a recognized analysis operation are
reported.

## Problem

The current detector asks Gemini to infer terminal dataset variables from the
whole notebook. The standalone `notebook_lineage.py` experiment provides useful
AST data-flow machinery, but it starts at analysis calls and searches for the
nearest DataFrame. It does not identify the external source or its tool, so it
cannot distinguish a LiPDGraph DataFrame from an unrelated pandas DataFrame and
cannot handle object-based sources such as PyLiPD or PyleoTUPS.

The deterministic detector must answer two linked questions:

1. Which notebook values originate from a recognized external dataset source?
2. Which of those source values have a data-flow path to a recognized analysis
   operation?

## Design

### Source-oriented static graph

Implement the detector in a new standalone module,
`src/deterministic_dataset_detection.py`, rather than changing the existing
LLM detector or workflow modules. It may reuse the experiment's versioned
dependency graph ideas, but source identity is a first-class fact on each graph
node.

The scanner processes code cells in notebook order, skips provenance-generated
cells using the existing notebook parser helpers, parses each valid cell with
`ast`, and records:

- variable versions and their dependencies;
- whether a value is a DataFrame, xarray object, or recognized dataset object;
- the canonical source tool attached to a source value;
- source position (cell and line) for stable ordering;
- recognized analysis calls and their dependencies.

Assignments, filtering/subscript expressions, DataFrame transformations,
method-return values, tuple/list accumulation, loops, and object merges are
tracked conservatively. Unsupported dynamic behavior is not executed and does
not produce a guessed citation.

### Source recognizers

The initial registry recognizes the repository's supported dataset families:

- `LiPD()` followed by `load()`, `load_from_dir()`,
  `load_remote_datasets()`, `load_datasets()`, or equivalent configured load
  methods → `PyLiPD`;
- `PangaeaDataset()` or `NOAADataset()` followed by `search_studies()` →
  `PyleoTUPS`;
- a `requests` call whose resolved URL contains the LinkedEarth GraphDB
  repository prefix, followed by a pandas read of the response → `LiPDGraph`;
- configured xarray loaders such as `open_dataset`, `open_mfdataset`,
  `open_dataarray`, `open_zarr`, and `load_dataset` → `xarray`;
- configured pandas/custom tabular factories for sources that are not covered
  by a domain-specific recognizer.

The registry is centralized and extendable. Function and method names are
case-normalized for matching, while output tool names use the existing
canonical spelling.

### Analysis sinks

Use a comprehensive deterministic set based on the current tracer's analysis
methods, expanded for the repository's paleoclimate workflows. This registry is
an internal implementation detail: the user supplies only the target notebook
path and cannot configure analysis sinks.

Analysis sinks include scientific modeling and statistical operations such as
PCA/EOF, fitting and prediction, correlation, regression, coherence, spectral
analysis, wavelets, SSA, clustering, decomposition, ANOVA, t-tests, and related
configured operations. Display and inspection methods such as `head`, `info`,
`describe`, `plot`, and `screeplot` are not sinks by themselves.

The detector reports a source only when at least one recognized sink is
reachable from that source. Multiple sinks that resolve to the same pair are
deduplicated.

### Output selection

The graph is traversed backward from each analysis sink to find the nearest
analysis-boundary dataset value, while retaining the source tool attached to
that value's upstream source.

- For table-like sources such as LiPDGraph and pandas, return the final
  DataFrame that directly feeds the analyzed domain object. In the canonical
  example this is `filtered_df2`, not the original `df_res`.
- For xarray, return the final xarray Dataset/DataArray that feeds the
  analysis, such as `ds_geo` after selection/resampling.
- For object-based PyLiPD and PyleoTUPS sources, return the source object or
  terminal merged source object that supplied the analyzed values, rather than
  a metadata/search-result DataFrame returned by one of its methods.

Results are emitted as `list[list[str]]`, deduplicated in first-source-position
order. Every traversal over a set is sorted by recorded source position and
variable name, so repeated runs over unchanged code produce byte-for-byte
identical results.

### Public API and transition

The deterministic detector's public entry point accepts only the target
notebook path:

```python
detect_datasets_in_notebook(notebook_path)
```

It returns the existing `list[list[str]]` pair shape. There is no public
`mode`, `analysis_methods`, or alternate detector argument.

This task does not add interchangeable public pathways or integrate the new
detector into the active workflow. The existing LLM implementation, prompt,
workflow, agent, orchestrator, and notebooks remain unchanged while the new
detector is validated against fixtures. A future transition task can replace
the current detector call with `detect_datasets_in_notebook(notebook_path)`;
that switch is documented here rather than implemented now. The old pathway
is therefore preserved without adding a public mode selector.

### PyleoTUPS target policy

PyleoTUPS source objects may contain multiple studies, but the user-facing
request does not know the study names available inside a notebook variable. The
workflow therefore distinguishes all-data requests from targeted
dataset/study-identifier requests:

- an all-data request with no target (`targets=None`) cites every detected
  analysis-used PyleoTUPS source, using the studies already held by that source
  object;
- a non-empty target is always a requested dataset name or study ID; notebook
  source variables are internal detector output and cannot be selected by the
  user. If the notebook contains a relevant PyleoTUPS source, the workflow
  returns a warning explaining that a specific PyleoTUPS study cannot be
  selected because its available names are not known, and recommends citing
  everything;
- a request containing an unsupported PyleoTUPS study target is a no-op: it
  must not remove old generated cells, write the notebook, or inject a new
  retrieval cell. Mixed requests containing that unsupported target also abort
  without mutation.

The natural-language agent uses the existing warning envelope for this case.
The active workflow performs the same validation before notebook mutation and
returns no dataset pairs when it cannot honor the request. The deterministic
detector remains a separate transition task; this policy is implemented
independently so the active workflow is safe while that detector is validated.

### Conservative failure behavior

Malformed or unsupported cells are skipped consistently with the existing
notebook parsing behavior. A source with no provable path to an analysis sink
is omitted. Ambiguous source/tool matches are omitted rather than converted
into an unsupported or speculative citation.

## Testing strategy

Tests cover:

- LiPDGraph endpoint detection and the `df_res` → `filtered_df2` → PCA path;
- PyLiPD object detection only when its loaded data reaches an analysis sink;
- PyleoTUPS object detection, including `get_data()` results and unused search
  objects;
- xarray loading followed by transformations into an analyzed Dataset;
- the fixed internal analysis-sink registry and its excluded inspection methods;
- aliases, filtering, loops, list accumulation, and object merges;
- duplicate analysis calls and deterministic result ordering across repeated
  runs;
- the path-only deterministic detector entry point;
- the unchanged legacy LLM implementation and the absence of a public mode
  selector;
- PyleoTUPS source recognition when data reaches an analysis sink, while
  unused search objects are excluded.

The implementation test file is
`tests/test_deterministic_dataset_detection.py`. The detector task adds only
its standalone source and test files; existing detector, workflow, benchmark,
and notebook files remain unchanged by the detector transition. The PyleoTUPS
request policy is implemented in the active workflow separately; standalone
detector tests cover source recognition and analysis reachability, not request
routing.

Benchmark expectations will be revised where existing fixtures currently list
datasets that are loaded or inspected but do not reach a scientific analysis
operation. The canonical `paleoPCAlite` expectation remains
`[["filtered_df2", "LiPDGraph"]]`.

## Non-goals

- Executing arbitrary notebooks to discover runtime state;
- removing or silently changing the existing LLM pathway;
- guaranteeing analysis attribution for `eval`, dynamic imports, hidden kernel
  state, or opaque third-party helper functions;
- treating citation retrieval or display calls as scientific analysis by
  themselves.
