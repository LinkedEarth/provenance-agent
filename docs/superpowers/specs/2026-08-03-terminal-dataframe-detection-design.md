# Terminal dataframe detection design

This specification changes how `deterministic_dataset_detection.py` decides
which values to report for table-backed sources. It is an algorithm change and
is explicitly out of scope for
[`2026-07-31-package-and-notebook-migration-design.md`](./2026-07-31-package-and-notebook-migration-design.md);
that document's Phase 0 requires this work to be committed and re-baselined
before the migration begins.

The motivating case is `notebooks/testing/02a-query_lipd_graph.ipynb`. It runs
three LiPDGraph queries, filters one of them, and performs no analysis at all.
Every one of its result frames still deserves a citation, so reaching a
recognized analysis call cannot remain the only route to being reported.

## Overview principles

1. **Cite what the notebook ends up with.** The reportable value for a table
   source is the frame the notebook actually carries forward, not the raw
   query result it started from and not an intermediate step.
2. **Do not report ancestors.** If one reported frame is derived from another,
   only the derived one is cited. A citation list must not contain both a
   frame and the frame it was filtered from.
3. **Do not silently drop parallel unresolved results.** When a source has no
   resolved analysis sink, two frames that neither derives from the other are
   two results, and both are cited. If any frame for that source resolves an
   analysis sink, the analysis result is authoritative and the source-level
   fallback is not also applied to its sibling frames.
4. **Shortest correct change.** This replaces one selection function and the
   loop that calls it. Do not restructure the graph, the source recognizers, or
   the sink machinery to accommodate it.
5. **Keep one active detection path.** The deprecated LLM detector is untouched
   and remains an inactive rollback path.

## Decisions

The following design questions were resolved before writing this plan.

**Parallel leaves are all reported.** When one source branches into several
terminal frames, every leaf is cited rather than only the last one by position:

```python
raw  = pd.read_csv("f.csv")
warm = raw[raw["temp"] > 0]
cold = raw[raw["temp"] < 0]
```

This must report `warm` and `cold`. Current behavior reports only `cold`,
because `_terminal_table_for_source` returns `max(candidates, key=order)`.

**Object-family sources keep the existing rule.** A source that never becomes a
frame still requires an analysis sink to be reported. Loading a LiPD file and
abandoning it stays silent:

```python
from pylipd.lipd import LiPD
D = LiPD()
D.load("x.lpd")          # still reports []
```

`test_unused_pylipd_load_is_omitted_without_analysis` remains valid and must
keep passing unchanged.

**Analysis takes precedence over fallback.** The fallback is evaluated once
per active source only when that source has no resolved analysis sink. This
means an analyzed branch and an unused sibling branch from the same source do
not produce a second fallback result. Reporting those siblings would require a
separate per-leaf merge of analysis and fallback boundaries.

**The public pair contract remains deduplicated.** Results are keyed by
`(variable, tool)`, as they are today. Independent lineages with distinct
terminal variable names remain separate; indistinguishable duplicate pairs are
collapsed because the public contract exposes no source ID and downstream
target selection is variable-based.

## Goals

- Report, for each active source with no resolved analysis sink, every terminal
  table-valued frame derived from it.
- Define "terminal" as leaf-in-lineage, not latest-by-position.
- Leave the analysis-sink path, the source recognizers, and the diagnostics
  contract alone. The qualified-name tightening they needed is already done.
- Preserve the `[variable, tool]` output contract and its sort order.

## Non-goals and invariants

- Do not change the `[variable, tool]` output shape, pair deduplication, or the
  pair ordering rule.
- Do not change object-family or xarray-family reporting. The resulting
  asymmetry is deliberate and must be documented: a plot-only pandas frame is
  cited, a plot-only xarray dataset is not.
- Do not add a notebook-level "has any analysis" gate. The fallback stays
  per-source; see the consequence recorded below.
- Do not touch the deprecated LLM detector or its tests.
- Do not change `benchmark/ground_truth/*.yml`. Those files are retained as
  labeled data only; the benchmark runner has been deleted and this change does
  not revive it.

## Current implementation

Three pieces of `src/deterministic_dataset_detection.py` are in scope.

`_Value` (line 267) carries `value_id`, `name`, `order`, `deps`, `source_ids`,
`family`, and `kind`. `deps` holds the value ids this value was computed from,
which is the lineage edge this design relies on.

`_closure` (line 1299) walks `deps` transitively and returns the set of
reachable value ids. It already does the ancestor computation this change
needs; no new traversal is required.

`_terminal_table_for_source` (line 1356) is the function being replaced. It
collects every value whose `source_ids` contain the source and whose `kind` is
`"table"`, then returns the single latest by `order`. The fallback loop at
line 1449 calls it once per active source that has no resolved analysis sink.
The source's family is not the filter: object sources that produce a table
through `get_data()` or `get_timeseries()` are eligible, while object-only
sources and xarray values remain governed by their existing rules.

## Algorithm change

Replace `_terminal_table_for_source` with a function returning a list.

Collect candidates as today, with one added filter: values whose `source_ids`
contain the source, whose `kind` is `"table"`, whose `name` is not `None`, and
which are still bound to that name when the scan ends. Then keep only the
leaves. A candidate is a leaf when no other candidate has it inside
its dependency closure:

```python
def _terminal_tables_for_source(self, source: _Source) -> list[_Value]:
    candidates = [
        value
        for value in self.values.values()
        if source.source_id in value.source_ids
        and value.kind == "table"
        and value.name is not None
        and self.env.get(value.name) is value
    ]
    ancestors: set[int] = set()
    for value in candidates:
        ancestors |= self._closure(value.deps)
    return sorted(
        (value for value in candidates if value.value_id not in ancestors),
        key=lambda value: value.order,
    )
```

Two properties this must have. Name rebinding (`df = pd.read_csv(...)` followed
by `df = df.dropna()`) produces two values where the first is an ancestor of the
second, so only the second survives, and the reported name is unchanged.
Independent branches produce two values where neither appears in the other's
closure, so both survive.

The `self.env` check is what keeps dead frames out. `self.values` holds every
historical value, not only live bindings, so a frame whose name was later
rebound to something else stays a leaf forever: nothing derives from it, so the
ancestor filter cannot remove it.

```python
df = pd.read_csv("data.csv")
df = 42                      # reports [['df', 'pandas']] without this check
```

This is pre-existing behavior, not something the leaf change introduces; the
current `max(candidates, key=order)` reports the same dead frame. It matters
because the emitted pair drives an injected `{var}.{method}` retrieval cell, so
a name that no longer holds the dataset produces a cell that fails when the user
runs it.

Verify `self.env` is safe to read this way rather than assuming it. Branch
handling saves and restores `env` during traversal, which looks like it would
lose conditional and loop assignments, but the branch envs are merged afterward:
a frame assigned inside an `if` body or a `for` body is still bound at the end
of the scan, so the check does not drop it.

Update the fallback loop to iterate the returned list, emitting one pair per
leaf and keeping the existing dedup-by-`(name, tool)` and position rules. The
loop must still skip a source whose analysis sink already resolved, as stated
above. The position tuple stays `(source.order, candidate.order)` so ordering
among leaves follows their position in the notebook.

## Accepted consequence

The fallback remains per-source, so a source that is loaded and never analyzed
is cited even when the notebook analyzes something else:

```python
main    = xr.open_dataset("main.nc")
solver  = Eof(main)
scratch = pd.read_csv("lookup.csv")   # never used again
```

reports both `main` and `scratch`. For a citation tool this is arguably correct,
since the lookup table was still read. It does raise the number of predictions,
which would have shown up as lower precision and higher recall.

That tradeoff can no longer be measured: the benchmark runner has been deleted,
and `benchmark/ground_truth/*.yml` survives as labeled data with nothing to run
it. Accept the behavior on the reasoning above rather than on evidence. If it
later proves too noisy on real notebooks, the follow-up lever is gating the
fallback on the notebook having no resolved sink anywhere, and that remains a
separate decision.

## Test plan

Add to `tests/test_deterministic_dataset_detection.py`:

- parallel leaves: one source, two independent filtered frames, both reported;
- ancestor exclusion: a three-step chain reports only the final frame;
- rebinding: `df = pd.read_csv(...)` then `df = df.dropna()` reports `df` once;
- dead binding: `df = pd.read_csv(...)` then `df = 42` reports nothing, since no
  live name holds the frame. This currently reports `[['df', 'pandas']]`, so it
  is a behavior change and the one corpus-independent regression this plan
  fixes;
- live binding inside a branch: a frame assigned in an `if` body or a `for` body
  is still reported, confirming the `self.env` check does not drop conditional
  or loop assignments;
- a leaf that is also an analysis boundary is not reported twice;
- the already-tightened recognizers keep holding: bare non-pandas `DataFrame`,
  `merge`, `concat`, and `json_normalize` do not promote an unanalyzed PyLiPD
  object. Add this as a guard if it is not already covered;
- a source with one analyzed branch and one unused sibling does not receive a
  second fallback result.

Existing tests that must keep passing unchanged:

- `test_unused_pylipd_load_is_omitted_without_analysis`;
- `test_sparql_dataframe_helpers_produce_one_result_per_query`;
- `test_no_analysis_uses_latest_filtered_table_boundary`;
- `test_inspection_and_plotting_methods_report_terminal_table`;
- both `paleoPCA` notebook cases.

Add a regression test pinning `02a-query_lipd_graph.ipynb` to its three
terminal frames, `df_search`,
`sparql_results_1767729896_fbfadce4_unique_TSiD`, and
`sparql_results_1767730535_65f2d398`. Define one repository-root path constant
for the current `notebooks/testing/` location and update that constant in the
same migration change if the notebook moves to `notebooks/examples/`; do not
leave the test dependent on an untracked or duplicated fixture.

## Already completed

Two items from an earlier draft of this plan are done and committed. They are
recorded here so nobody implements them twice, and because the second one shapes
what the recognizers can and cannot do.

**Resolved-source bookkeeping.** `resolved_source_ids.add(source_id)` now runs
after the `pair[0] is None` check, so a sink candidate with no name can no
longer mark a source resolved and suppress its fallback.

**Qualified-name pandas recognizers.** `_is_pandas_dataframe_constructor` and
`_is_pandas_table_function` now require a `pd.` or `pandas.` prefix. Their old
`"." not in name` branch accepted any bare call named `merge`, `concat`,
`json_normalize`, or `DataFrame`, so an unrelated `from mylib import merge`
could promote an unanalyzed PyLiPD load into a citation. That mattered more once
table-backed values stopped needing an analysis call to be cited.

Two consequences to keep in mind while implementing the rest. Checking what was
actually imported is not an option: `visit_Import` and `visit_ImportFrom` are
deliberate no-ops because the recognizers work from call spelling by design. And
`from pandas import DataFrame; DataFrame(...)` is now intentionally unsupported,
which is the accepted cost of the prefix rule. `_is_pandas_reader` keeps its
bare-name branch, since `read_csv` and its siblings do not collide with ordinary
code.

## Steps

1. Replace `_terminal_table_for_source` with the leaf-returning, live-binding
   version and update its caller.
2. Update the module docstring and the numbered algorithm comment: step 6
   currently says "the latest source-backed table" and must say every terminal
   leaf. Document the pandas/xarray asymmetry in the design-decisions block.
3. Update `CLAUDE.md`'s detection bullet, which currently says "terminal tabular
   DataFrames" in the singular sense.
4. Re-run the corpus check below and diff it against the recorded table.

## Corpus check

With the benchmark runner deleted, this table replaces scoring. It is the
detector's output over every tracked notebook as of this plan, produced by
running `detect_datasets_in_notebook` on each path. Re-run it after the change
and diff against these values; every difference must be explained, and only the
rows marked as expected-to-change may change.

| Notebook | Expected pairs | Changes? |
|---|---|---|
| `notebooks/testing/02a-query_lipd_graph.ipynb` | `df_search`, `sparql_results_1767729896_fbfadce4_unique_TSiD`, `sparql_results_1767730535_65f2d398`, all `LiPDGraph` | no |
| `notebooks/Graph.ipynb` | `[['df_filt', 'LiPDGraph']]` | no |
| `notebooks/testing/LIPD.ipynb` [^ts] | `[['ts_list', 'PyLiPD'], ['df_cut', 'PyLiPD'], ['df_essential', 'PyLiPD'], ['df_merged', 'PyLiPD'], ['df_temp', 'PyLiPD'], ['df_filt', 'PyLiPD']]` | yes |
| `notebooks/testing/PyleoTUPS.ipynb` | `[['dfs', 'PyleoTUPS']]` | no |
| `notebooks/testing/paleoPCA.ipynb` | `[['filtered_df2', 'LiPDGraph'], ['ds_geo', 'xarray']]` | no |
| `notebooks/testing/paleoPCAlite.ipynb` | `[['filtered_df2', 'LiPDGraph']]` | no |
| `notebooks/testing/Instruction Notebooks/Notebook1/notebook1.ipynb` [^ts] | `[['df1', 'LiPDGraph'], ['df_data', 'PyleoTUPS'], ['ts_list', 'PyLiPD'], ['df3', 'PyLiPD']]` | yes |
| `notebooks/testing/Instruction Notebooks/Notebook2/notebook2.ipynb` | `[['df', 'PyleoTUPS'], ['df_graph', 'LiPDGraph'], ['df_l', 'PyLiPD']]` | no |
| `notebooks/testing/Instruction Notebooks/Notebook3/notebook3.ipynb` | `[['df_data', 'PyleoTUPS'], ['df', 'PyLiPD'], ['df_graph', 'LiPDGraph']]` | no |
| `notebooks/testing/Instruction Notebooks/Notebook4/notebook4.ipynb` | `[['D_ice', 'LiPDGraph'], ['dsp', 'PyleoTUPS'], ['D', 'PyLiPD']]` | no |

Every other tracked notebook returns `[]` and must continue to: the five
`comparing-simulated-reconstructed-climate/` notebooks,
`C02_b_DA_with_individual_seasonality.ipynb`, `overall_workflow.ipynb`,
`provenance_magic.ipynb`, `sample.ipynb`, `test_magic_commands.ipynb`,
`testing/data_workflow.ipynb`, `testing/dataset_pipeline.ipynb`,
`testing/test1.ipynb`, and `workflow.ipynb`.

[^ts]: The `ts_list` entry in these two rows is a recorded observation, not a
validated expectation. Both notebooks bind it through
`ts_list, df = D.get_timeseries(names, to_dataframe=True)`. With that flag
PyLiPD returns `(timeseries_dict, dataframe)`, so `df` holds the DataFrame and
`ts_list` holds a dict. Typing is applied at the call level, so both unpacked
names inherit the table kind and the dict is what gets cited. Because the pair
drives a `{var}.{method}` retrieval cell, these two rows would inject a cell
against a dict. Distinguishing tuple elements is a separate follow-up; until it
lands, treat these entries as pinning current behavior rather than correct
behavior, and do not cite them as evidence that the detector is right here.

Two rows intentionally change under the leaf rule. `LIPD.ipynb` and
`Instruction Notebooks/Notebook1/notebook1.ipynb` each contain several
independent live table values from the same PyLiPD source; the old
`max(candidates, key=order)` selection reported only the last one, while the
new algorithm reports every terminal leaf. The remaining 22 notebooks are
unchanged, and the live-binding filter removes no corpus result. Any other
diff means either a notebook changed since this table was recorded or the
implementation is wrong. If another notebook genuinely needs a new expected
value, update this table in the same commit and say why.

## Acceptance criteria

1. Parallel leaves are both reported; ancestors and dead bindings never are.
2. `02a-query_lipd_graph.ipynb` reports exactly its three terminal frames.
3. Object-family behavior is unchanged and its test passes untouched.
4. Bare non-pandas `DataFrame`, `merge`, `concat`, or `json_normalize` no longer
   creates a source-backed table.
5. The full test suite passes.
6. The corpus check reproduces the recorded table exactly, or each difference
   is explained and the table updated in the same commit.
7. The work is committed before the package migration's Phase 1 begins.
