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
3. **Do not silently drop parallel results.** Two frames that neither derives
   from the other are two results, and both are cited.
4. **Shortest correct change.** This replaces one selection function and the
   loop that calls it. Do not restructure the graph, the source recognizers, or
   the sink machinery to accommodate it.
5. **Keep one active detection path.** The deprecated LLM detector is untouched
   and remains an inactive rollback path.

## Decisions

Two design questions were resolved before writing this plan.

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

## Goals

- Report, for each active table-family source with no resolved analysis sink,
  every terminal frame derived from it.
- Define "terminal" as leaf-in-lineage, not latest-by-position.
- Leave the analysis-sink path, the source recognizers, and the diagnostics
  contract alone.
- Preserve the `[variable, tool]` output contract and its sort order.

## Non-goals and invariants

- Do not change the `[variable, tool]` output shape or the pair ordering rule.
- Do not change object-family or xarray-family reporting. The resulting
  asymmetry is deliberate and must be documented: a plot-only pandas frame is
  cited, a plot-only xarray dataset is not.
- Do not add a notebook-level "has any analysis" gate. The fallback stays
  per-source; see the consequence recorded below.
- Do not touch the deprecated LLM detector or its tests.
- Do not change benchmark scoring or ground-truth semantics. Score movement
  from this change is expected and is measured, not corrected for.

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
line 1449 calls it once per unresolved active source.

## Algorithm change

Replace `_terminal_table_for_source` with a function returning a list.

Collect candidates exactly as today: values whose `source_ids` contain the
source, whose `kind` is `"table"`, and whose `name` is not `None`. Then keep
only the leaves. A candidate is a leaf when no other candidate has it inside
its dependency closure:

```python
def _terminal_tables_for_source(self, source: _Source) -> list[_Value]:
    candidates = [
        value
        for value in self.values.values()
        if source.source_id in value.source_ids
        and value.kind == "table"
        and value.name is not None
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

Update the fallback loop to iterate the returned list, emitting one pair per
leaf and keeping the existing dedup-by-`(name, tool)` and position rules. The
position tuple stays `(source.order, candidate.order)` so ordering among leaves
follows their position in the notebook.

## Related fixes to make in the same change

**Reorder the resolved-source bookkeeping.** In the sink loop,
`resolved_source_ids.add(source_id)` at line 1443 runs before the
`if pair[0] is None: continue` check. A source whose sink candidate has no name
is marked resolved, emits nothing, and is then skipped by the fallback. I could
not construct a notebook that triggers this, so treat it as defensive: move the
`add` below the name check so an unnamed candidate cannot suppress the fallback.

**Tighten the bare-name pandas recognizers.** `_is_pandas_table_function`
(line 365) and `_is_pandas_dataframe_constructor` (line 358) accept any bare
call whose name is `merge`, `concat`, `json_normalize`, or `DataFrame`, because
of their `"." not in name` branch. An unrelated import promotes an unreported
source:

```python
from pylipd.lipd import LiPD
from mylib import merge      # nothing to do with pandas
D = LiPD()
D.load("x.lpd")
out = merge(D, 1)            # currently reports [['out', 'PyLiPD']]
```

This matters more after this change, because table-backed values no longer need
to reach an analysis call to be cited, so a spurious table constructor turns
directly into a spurious citation. Require the `pd.` or `pandas.` prefix for
`merge`, `concat`, and `json_normalize`, whose bare spellings collide with
ordinary user and third-party code.

Note that checking what was actually imported is not available: `visit_Import`
and `visit_ImportFrom` (lines 636 and 639) are deliberate no-ops, because the
recognizers work from call spelling by design. Requiring the prefix is the fix
that respects that decision. `_is_pandas_reader` keeps its bare-name branch;
`read_csv` and its siblings are distinctive enough not to collide.

## Consequence to measure, not to fix here

The fallback remains per-source, so a source that is loaded and never analyzed
is cited even when the notebook analyzes something else:

```python
main    = xr.open_dataset("main.nc")
solver  = Eof(main)
scratch = pd.read_csv("lookup.csv")   # never used again
```

reports both `main` and `scratch`. For a citation tool this is arguably correct,
since the lookup table was still read. It also raises the number of predictions,
so benchmark precision may fall while recall rises. Record the movement rather
than tuning it away; if precision drops unacceptably, gating the fallback on the
notebook having no resolved sink anywhere is the follow-up lever, and it is a
separate decision.

## Test plan

Add to `tests/test_deterministic_dataset_detection.py`:

- parallel leaves: one source, two independent filtered frames, both reported;
- ancestor exclusion: a three-step chain reports only the final frame;
- rebinding: `df = pd.read_csv(...)` then `df = df.dropna()` reports `df` once;
- a leaf that is also an analysis boundary is not reported twice;
- the tightened recognizers: a bare non-pandas `merge` no longer promotes an
  unanalyzed PyLiPD object.

Existing tests that must keep passing unchanged:

- `test_unused_pylipd_load_is_omitted_without_analysis`;
- `test_sparql_dataframe_helpers_produce_one_result_per_query`;
- `test_no_analysis_uses_latest_filtered_table_boundary`;
- `test_inspection_and_plotting_methods_report_terminal_table`;
- both `paleoPCA` notebook cases.

Add a regression test pinning `02a-query_lipd_graph.ipynb` to its three
terminal frames, `df_search`,
`sparql_results_1767729896_fbfadce4_unique_TSiD`, and
`sparql_results_1767730535_65f2d398`. That notebook moves to
`notebooks/examples/` during the migration, so reference it through a constant
the migration's path sweep will catch.

## Steps

1. Replace `_terminal_table_for_source` with the leaf-returning version and
   update its caller.
2. Reorder the `resolved_source_ids` bookkeeping.
3. Tighten the two bare-name recognizers.
4. Update the module docstring and the numbered algorithm comment: step 6
   currently says "the latest source-backed table" and must say every terminal
   leaf. Document the pandas/xarray asymmetry in the design-decisions block.
5. Update `CLAUDE.md`'s detection bullet, which currently says "terminal tabular
   DataFrames" in the singular sense.
6. Run the full suite, then re-run the benchmark and record the new baseline.

## Acceptance criteria

1. Parallel leaves are both reported; ancestors never are.
2. `02a-query_lipd_graph.ipynb` reports exactly its three terminal frames.
3. Object-family behavior is unchanged and its test passes untouched.
4. A bare non-pandas `merge`, `concat`, or `json_normalize` no longer creates a
   source-backed table.
5. The full test suite passes.
6. A new benchmark baseline is recorded, with the score delta against the
   previous baseline written down and explained rather than tuned away.
7. The work is committed before the package migration's Phase 1 begins.
