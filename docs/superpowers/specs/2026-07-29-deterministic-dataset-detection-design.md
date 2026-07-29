# Deterministic Dataset Detection Design

**Status:** Proposed

## Goal

Add a deterministic, source-aware dataset detector that returns the same
`[variable, tool]` pairs as the existing LLM pathway's public contract, while
preserving the LLM detector and allowing callers to select either pathway.
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

Implement a repository-native detector rather than copying the experimental
script wholesale. It may reuse the experiment's versioned dependency graph
ideas, but source identity is a first-class fact on each graph node.

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

This task does not add interchangeable public pathways. The existing LLM
workflow remains the active production path while the deterministic detector is
validated against the notebook fixtures. The LLM implementation and prompt are
not deleted; a commented integration point will show the future one-line switch
to `detect_datasets_in_notebook(notebook_path)` once the deterministic detector
is good enough. The deterministic implementation lives in a focused module and
is exercised directly by its tests and manual fixture checks.

### PyleoTUPS target policy

PyleoTUPS source objects may contain multiple studies, but the user-facing
request does not know the study names available inside a notebook variable. The
workflow therefore distinguishes source selection from study-name selection:

- an all-data request with no target (`targets=None`) cites every detected
  analysis-used PyleoTUPS source, using the studies already held by that source
  object;
- a target that exactly matches a detected notebook source variable may select
  that source object;
- an unmatched target is treated as a requested dataset name. If the notebook
  contains a PyleoTUPS source relevant to the request, the workflow returns a
  warning explaining that a specific PyleoTUPS study cannot be selected because
  its available names are not known, and recommends citing everything;
- a request containing an unsupported PyleoTUPS study target is a no-op: it
  must not remove old generated cells, write the notebook, or inject a new
  retrieval cell. Mixed requests containing that unsupported target also abort
  without mutation.

The natural-language agent uses the existing warning envelope for this case.
The direct workflow performs the same validation before notebook mutation and
returns no dataset pairs when it cannot honor the request.

### Conservative failure behavior

Malformed or unsupported cells are skipped consistently with the existing
notebook parsing behavior. A source with no provable path to an analysis sink
is omitted. Ambiguous source/tool matches are omitted rather than converted
into an unsupported or speculative citation.

## Testing strategy

Tests will be written before implementation and will cover:

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
- the unchanged legacy LLM implementation and its commented future switch;
- PyleoTUPS all-data requests, source-variable selection, and unsupported
  dataset-name warnings with zero notebook mutation.

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
