# In-place cleanup: preserve public paths, keep deterministic detection active

This revision narrows the earlier refactor plan after review. It is an
in-place cleanup task, not a packaging or public-API migration. The current
deterministic detector remains the active path, while the previous LLM detector
remains available as a deprecated fallback in case the project needs to switch
back later.

## Baseline

- Branch `provenance-magic-working` at `2270e97`.
- `/opt/anaconda3/envs/lang/bin/python -m pytest` reports **188 passed** at
  the time this plan was written.
- The working tree contains user notebook/editor changes. They are not part of
  this task and must not be staged with it.

## Scope boundaries

This task deliberately preserves the current project shape and public paths:

- Keep the flat `src/` layout. Do not create `src/provenance_agent/`.
- Do not add `pyproject.toml`, editable-install requirements, or package
  discovery configuration.
- Keep the existing module filenames and imports, including
  `notebook_parser.py`, `bibliography.py`, `data_workflow.py`,
  `software_workflow.py`, `orchestrator.py`, and `provenance.py`.
- Keep `%load_ext provenance` unchanged.
- Keep `cite_data`, `cite_software`, `cite_data_tool`, and
  `cite_software_tool` importable from their current paths.
- Keep the existing workflow signatures and compatibility parameters,
  including `tool`, `variable`, `targets`, `detected_pairs`, and `fmt`.
- Do not change benchmark code, benchmark tests, ground truth, or benchmark
  scoring in this task.
- Do not modify currently changed notebooks.

Packaging, module renames, notebook reorganization, and public import-path
migration are future tasks listed at the end of this document.

## Overview principles

These principles govern every implementation decision in this task:

1. **Shortest correct code.** A change is worthwhile only when it removes
   complexity or fixes a real contract problem. Do not introduce a new module,
   type, wrapper, or migration layer merely to make the architecture look
   different.
2. **Delete rather than preserve dead code.** Remove code that is genuinely
   unused and has no compatibility value. The retained LLM detector is an
   explicit exception: it is a documented rollback path, so it must remain
   deprecated and inactive rather than being treated as dead code.
3. **Stop when a file is fine.** Do not reopen or reorganize modules whose
   responsibilities and public behavior are already correct for this scope.
4. **Preserve behavior and public paths.** Existing imports, signatures, tool
   invocation, notebook mutation rules, and deterministic detector results
   remain stable. The intentional exceptions are deleting APA rendering and
   removing `fmt` validation while retaining `fmt` as an ignored parameter.
5. **Keep one active data-detection path.** Deterministic detection is the
   production path. The LLM detector is retained only as an explicitly marked
   fallback and must not be called by normal detection.
6. **Keep scope isolated.** Do not mix packaging, module renames, notebook
   reorganization, benchmark changes, or unrelated user-worktree cleanup into
   this task. Those belong to the tracked future-work items.

## Governing decisions

### Deterministic detection stays active

`dataset_detection.detect_datasets(notebook_path)` remains the active public
facade and continues to delegate to `deterministic_dataset_detection.py`.
The deterministic algorithm, module names, public aliases, diagnostics, and
`[variable, tool]` return contract are unchanged.

The legacy LLM detection helpers remain in `dataset_detection.py`:

- `DETECTION_PROMPT`;
- `build_detection_prompt`;
- `_strip_code_fences`;
- `parse_detection_response`; and
- the commented LLM fallback in `detect_datasets`.

They are explicitly deprecated in module and function documentation. The
deprecation is documentation-only: the active path emits no deprecation
warning, and the helper tests remain so the fallback can be restored later.

`message_text` remains in `llm.py` because the deprecated detector fallback
uses it. No lazy-client redesign is part of this task. The agent continues to
use its existing LLM construction behavior when the agent path is imported or
run.

### APA rendering is deleted; `fmt` remains compatibility-only

Delete the APA rendering implementation:

- `bibtex_to_apa`, its prompt, and its chain;
- `render_apa`;
- `render_bibtex_strings_to_apa`;
- `generate_bibliography` and `generate_bibliography_cell`; and
- `render_bibtex_strings_to_df` if it has no surviving caller.

Keep the lower-level citation-index and BibTeX parsing helpers used by
`collect_library_entries`.

Retain `fmt` everywhere it currently appears, including public function
signatures, `RouteDecision`, prompts, dispatch records, and tool schemas. Its
value is silently ignored. Remove format validation and do not branch on the
value; `"bibtex"`, `"apa"`, `"html"`, and other supplied values all follow the
existing output path. If `RouteDecision.fmt` currently uses a restrictive
`Literal`, widen it to `str` while keeping the field and default.

Update active source documentation and `CLAUDE.md` so they no longer claim that
APA rendering is available. Historical design documents may continue to
mention the retired path.

### Targeted PyleoTUPS requests remain a no-op

When a non-empty target is supplied and the detector finds a PyleoTUPS source,
including a target representing a study ID such as `830587`:

1. emit the existing `UserWarning` behavior;
2. return `[]`; and
3. leave the notebook unchanged.

The warning should explain that specific PyleoTUPS studies cannot be selected
before the notebook's in-memory object runs, and should refer to names or IDs.
Requests to cite everything remain supported.

## File-level work

### `src/llm.py`

- Remove only the APA prompt, chain, and conversion function.
- Keep the shared LLM client and `message_text` for the classifier and
  deprecated detector fallback.
- Preserve current model, temperature, and credential behavior.
- Do not make model or chain construction lazy in this task.

### `src/dataset_detection.py` and `src/deterministic_dataset_detection.py`

- Do not rename either file.
- Do not alter the deterministic algorithm or public return contracts.
- Keep the legacy LLM helpers and commented fallback.
- Add clear `Deprecated` documentation explaining that deterministic detection
  is the active path and identifying the preserved fallback helpers.
- Preserve warning emission and structured diagnostics behavior.

### `src/bibliography.py`

- Remove the APA/rendering and combined-bibliography functions listed above.
- Keep the module name and its surviving citation lookup/DataFrame helpers.
- Remove the obsolete `__main__` block if it calls a deleted bibliography
  function.
- Do not relocate notebook cell-lifecycle helpers in this task.
- Keep existing generated-cell marker constants and removal behavior unless a
  test demonstrates a directly dead branch that can be removed without a
  public or compatibility change.

### `src/orchestrator.py`

- Keep the module and all current direct/tool entry points.
- Keep `_cite_data_tool_entry` and the `StructuredTool` wrappers.
- Remove format validation, but retain `fmt` in signatures and descriptions as
  a silently ignored compatibility parameter.
- Preserve the current manual invocation contract:

  ```python
  from orchestrator import cite_data, cite_data_tool

  cite_data("analysis.ipynb", fmt="apa")
  cite_data_tool.invoke({"notebook_path": "analysis.ipynb", "fmt": "apa"})
  ```

### `src/data_workflow.py`, `src/software_workflow.py`, `src/agent.py`, and
`src/provenance.py`

- Keep filenames, imports, public signatures, and current routing behavior.
- Keep `fmt` threaded through the existing agent/data path, but make it have no
  effect and never reject its value.
- Keep `%load_ext provenance` and the existing magic module name.
- Do not remove `tool`, `variable`, `filter_datasets`, `split_targets`, or
  other compatibility surfaces in this task.

## Tests

Preserve the existing test organization and module names. Make the smallest
test changes needed for the cleanup:

- Keep the legacy LLM prompt/parser and `message_text` tests.
- Remove or rewrite only tests for deleted APA rendering functions.
- Replace format-rejection assertions with tests showing that supplied `fmt`
  values are accepted and produce the same behavior as the default.
- Add or retain a direct `cite_data`/`cite_data_tool` regression test proving
  the deterministic data path does not call the LLM detector.
- Test a targeted PyleoTUPS study ID/name for `UserWarning`, `[]`, and no
  notebook mutation.
- Preserve deterministic detector tests, including diagnostics and legacy
  aliases.
- Do not add benchmark tests or alter benchmark fixtures.

## Documentation and safe cleanup

Update active source documentation and `CLAUDE.md` to reflect:

- deterministic detection as the active path;
- the deprecated but retained LLM detector;
- APA rendering removal;
- `fmt` being accepted but ignored; and
- the unchanged flat-module and `%load_ext provenance` interfaces.

Do not edit currently modified notebooks as part of this task.

The following cleanup is explicitly approved, but must use exact paths rather
than a broad recursive deletion:

- `notebooks/testing/testjunk.ipynb`;
- `notebooks/testing/paleoPCAlite_with_citations.ipynb`;
- `notebooks/.DS_Store`; and
- `notebooks/testing/.DS_Store`.

Do not delete other untracked files or the LiPD fixture without a separate
explicit approval.

## Verification

The task is complete when:

1. `/opt/anaconda3/envs/lang/bin/python -m pytest` passes.
2. `cite_data` and `cite_data_tool.invoke` still work through their existing
   `orchestrator` imports.
3. `fmt="bibtex"`, `fmt="apa"`, `fmt="html"`, and an arbitrary string are
   accepted without changing the current output path.
4. The deterministic detector returns the same pairs and diagnostics as before.
5. The deprecated LLM detector helpers remain importable and their tests pass.
6. A targeted PyleoTUPS name or ID warns, returns `[]`, and does not rewrite
   the notebook.
7. No benchmark file, ground truth, or currently modified notebook is changed
   by the implementation.
8. Active documentation contains no claim that APA rendering is supported.

## Future work

These items are intentionally tracked here and are not part of this task:

1. **Package conversion:** introduce a proper installable package, choose the
   final package namespace, and remove `sys.path` mutations from tests and
   notebooks.
2. **Module renames:** consider `notebook_io.py`, `citations.py`, `data.py`,
   `software.py`, and any corresponding import migration.
3. **Public API migration:** decide whether and when to remove or replace
   `orchestrator.py`, and document canonical direct/tool imports.
4. **Magic packaging:** preserve or intentionally migrate `%load_ext
   provenance` when the package layout changes.
5. **Notebook reorganization:** move demos/fixtures/examples, regenerate stale
   cells, and update path references after the package migration.
6. **Benchmark work:** revisit benchmark code and ground truth separately,
   including the analysis-only citation criterion.

Each future item should receive its own design/implementation plan rather than
being folded into this cleanup.
