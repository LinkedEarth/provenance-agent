# Package and notebook migration design

This is a future migration specification. It expands the future-work items in
[`2026-07-31-refactor-plan.md`](./2026-07-31-refactor-plan.md) and is not part
of the current in-place cleanup.

The migration will turn the repository into an installable local Python
package for use by Jupyter kernels and future consumers, move the
implementation behind stable package-qualified imports, preserve the existing
`%load_ext provenance` notebook command, and eventually reorganize notebooks
without changing dataset-detection behavior or benchmark semantics. It is not
a public PyPI release or an application-integration task.

## Overview principles

These principles govern every implementation decision in this migration:

1. **Shortest correct code.** A change is worthwhile only when it removes
   complexity or fixes a real contract problem. Do not introduce a new module,
   type, wrapper, or migration layer merely to make the architecture look
   different.
2. **Delete rather than preserve dead code.** Remove code that is genuinely
   unused and has no compatibility value. The retained LLM detector is an
   explicit exception: it is a documented rollback path, so it must remain
   deprecated and inactive rather than being treated as dead code. The
   top-level `provenance.py` extension shim is another explicit exception
   because `%load_ext provenance` is a permanent public contract.
3. **Stop when a file is fine.** Do not reopen or reorganize modules whose
   responsibilities and public behavior are already correct for this scope.
4. **Preserve behavior and intentional contracts.** Dataset-detection results,
   workflow behavior, tool invocation, notebook mutation rules, and benchmark
   semantics remain stable. The intentional migration changes are package-
   qualified imports, responsibility-based module names, removal of
   `orchestrator.py`, and notebook path updates.
5. **Keep one active data-detection path.** Deterministic detection is the
   production path. The LLM detector is retained only as an explicitly marked
   fallback and must not be called by normal detection.
6. **Keep scope isolated.** Do not mix algorithm changes, benchmark-semantic
   changes, APA rendering changes, or unrelated user-worktree cleanup into this
   migration. Those belong to separate tasks.

## Goals

- Install the project and all supported workflow/test dependencies into the
  local `lang` environment with one command: `pip install -e .`. This is a
  local development installation requirement, not a publication requirement.
- Use `provenance_agent` as the Python import namespace and
  `provenance-agent` as the distribution name.
- Provide a stable Python package boundary for future consumers without
  committing to an external application integration.
- Remove `sys.path.insert(...)` and `sys.path.append(...)` from tests and
  notebook code cells.
- Give direct functions and LangChain tools one canonical package-qualified
  import path.
- Preserve `%load_ext provenance` as the notebook-facing extension name.
- Move the implementation into names that describe its responsibilities:
  `notebook_io`, `citations`, `data`, and `software`.
- Reorganize notebooks into demos, examples, fixtures, and exploration only
  after the package foundation is stable.
- Regenerate or remove stale generated provenance cells after the imports move.
- Keep deterministic dataset detection as the active path and preserve the
  deprecated LLM detector fallback.

## Non-goals and invariants

- Do not change the deterministic detection algorithm or its
  `[variable, tool]` output contract.
- Do not change PyleoTUPS targeted-request behavior: a specific name or ID
  produces a warning, returns `[]`, and leaves the notebook unchanged.
- Do not reintroduce APA rendering. `fmt` remains accepted and ignored.
- Do not make LLM construction lazy as part of this migration. Importing the
  direct data API must remain independent of the agent, but importing
  `provenance_agent.agent` retains the current agent/LLM behavior.
- Do not change benchmark scoring, expected datasets, or ground-truth
  semantics. Path-only edits to ground-truth `notebook:` fields are allowed
  when notebooks move.
- Do not publish the distribution to PyPI as part of this work.
- Do not integrate with any external application as part of this refactoring.
- Do not add a standalone CLI as part of this refactoring.
- Do not delete user-created or untracked files unless separately approved.
- Do not maintain multiple implementations of the same public operation.
  Temporary migration shims, where explicitly listed below, must contain only
  forwarding imports and have a removal condition.

## Chosen package layout

The final source layout is:

```text
src/
├── provenance.py                 # stable %load_ext provenance shim
└── provenance_agent/
    ├── __init__.py               # cite_data, cite_software
    ├── llm.py                    # existing eager shared LLM client
    ├── notebook_io.py            # renamed notebook_parser.py
    ├── citations.py              # renamed bibliography.py
    ├── software.py               # renamed software_workflow.py + tool
    ├── data.py                   # renamed data_workflow.py + tool
    ├── dataset_detection.py      # active facade + deprecated LLM helpers
    ├── deterministic_dataset_detection.py
    ├── agent.py                  # LCEL router and run()
    ├── magic.py                  # implementation of the IPython extension
    └── Citations/                # packaged citation data files
```

`provenance_agent` is the import namespace. `provenance` remains only as the
intentional IPython extension entry point. It forwards to
`provenance_agent.magic` and does not contain a second implementation.

The package root must not import `agent` or `magic`. This keeps the direct
functions independent from the agent's LLM import without introducing a new
lazy-client API:

```python
from provenance_agent import cite_data, cite_software
from provenance_agent.data import cite_data_tool
from provenance_agent.software import cite_software_tool
from provenance_agent.agent import run
```

`run` is intentionally imported from `provenance_agent.agent`, not reexported
from `provenance_agent`, so importing the direct APIs does not import the
agent. The package root exports only `cite_data` and `cite_software`.

The layout below is the final target. The package-foundation phase may first
preserve internal filenames while it establishes the package boundary; the
responsibility-based renames and `orchestrator.py` removal happen only after
package imports and direct APIs are verified. Direct `cite_data(...)` use
inside a local Jupyter kernel remains supported and continues to use the
package directly.

## Packaging configuration

Add a `pyproject.toml` with explicit setuptools configuration:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "provenance-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "nbformat",
  "bibtexparser",
  "PyYAML",
  "pandas",
  "requests",
  "python-dotenv",
  "langchain-core",
  "langchain-google-genai",
  "pydantic",
  "ipython",
  "ipynbname",
  "pylipd",
  "pyleotups",
  "pytest",
]

[tool.setuptools]
package-dir = {"" = "src"}
py-modules = ["provenance"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
provenance_agent = ["Citations/**/*"]
```

This configuration supports local editable installation and future application
environments; it does not require a public package release. `environment.yml`,
if retained for developer setup, may install the editable package as part of
environment creation, but it does not replace `pyproject.toml`.

The final project metadata must declare all dependencies needed by the package,
tests, generated retrieval cells, and notebook magic, using their distributable
names:

- `nbformat`;
- `bibtexparser`;
- `PyYAML`;
- `pandas`;
- `requests`;
- `python-dotenv`;
- `langchain-core`;
- `langchain-google-genai`; and
- `pydantic`;
- `ipython`;
- `ipynbname`;
- `pylipd`;
- `pyleotups`; and
- `pytest`.

`pylipd` and `pyleotups` are needed by generated cells in the target notebook
kernel, not by static package import, but they are included in the single
installation for a predictable supported environment.

The local `src/.env` file remains developer configuration and must not be
packaged or committed. Installed use relies on environment variables or a
user-provided dotenv file. Add a packaging test that confirms no secret file is
included in the build artifact.

## Citation data resources

The root `Citations/` directory cannot remain an uninstalled runtime
dependency. Move it under `src/provenance_agent/Citations/` and include its
files as package data.

Update `citations.py` to resolve the directory with
`importlib.resources.files("provenance_agent").joinpath("Citations")` rather
than walking from `__file__` to the repository root. The lookup and DataFrame
results must remain unchanged. Tests must exercise citation lookup after an
editable install and from a working directory outside the repository.

## Module migration

After the package foundation is working, perform these moves with imports
updated in the same migration change. The initial package installation does
not depend on completing every rename below.

| Current module | Canonical module | Responsibility |
|---|---|---|
| `notebook_parser.py` | `provenance_agent.notebook_io` | notebook reading, parsing, directives, generated-cell lifecycle |
| `bibliography.py` | `provenance_agent.citations` | citation index lookup and citation DataFrames |
| `software_workflow.py` | `provenance_agent.software` | software cell builder, direct function, LangChain tool |
| `data_workflow.py` | `provenance_agent.data` | data cell builder, direct function, LangChain tool |
| `dataset_detection.py` | `provenance_agent.dataset_detection` | deterministic facade, diagnostics, deprecated LLM helpers |
| `deterministic_dataset_detection.py` | `provenance_agent.deterministic_dataset_detection` | deterministic data-flow implementation |
| `agent.py` | `provenance_agent.agent` | classifier, dispatch, verification, `run` |
| `provenance.py` implementation | `provenance_agent.magic` | IPython magic implementation |
| `llm.py` | `provenance_agent.llm` | shared LLM client and response helper |

Use relative package imports inside `provenance_agent`. Generated notebook
cells must use package-qualified imports, for example:

```python
from provenance_agent.citations import collect_library_entries
```

The deprecated LLM detector helpers move with `dataset_detection.py`; they are
not deleted or silently replaced. The deterministic facade continues to be
the only normal caller path.

## Public API migration

The canonical public API after migration is:

```python
from provenance_agent import cite_data, cite_software
from provenance_agent.data import cite_data_tool
from provenance_agent.software import cite_software_tool
from provenance_agent.agent import run
```

The direct functions retain their current signatures, including `fmt` as a
silently ignored compatibility argument. The tools retain their current
LangChain invocation shape and hide internal `detected_pairs` state.

`orchestrator.py` is removed after all repository imports are migrated. Its
logic is moved to `data.py` and `software.py`; it is not retained as a second
implementation or a permanent compatibility module. The removal is a
deliberate public import migration:

```python
# old, removed after migration
from orchestrator import cite_data, cite_data_tool

# canonical
from provenance_agent import cite_data
from provenance_agent.data import cite_data_tool
```

Update source, tests, documentation, and tracked notebooks before deleting the
old module. Add a structural test that fails on old imports while allowing the
intentional `provenance` extension shim.

## Magic packaging

The user-facing command remains:

```python
%load_ext provenance
```

The installed top-level `provenance.py` shim must expose
`load_ipython_extension` and delegate to `provenance_agent.magic`. It may also
forward the existing public magic helpers needed by tests. It must not import
or duplicate the old implementation.

`provenance_agent.magic` owns the implementation and imports
`provenance_agent.agent` using package-qualified imports. The package
dependencies supply the IPython requirements.

Acceptance tests must verify both:

1. `import provenance` succeeds after editable installation; and
2. a fake IPython shell can register the magic through
   `provenance.load_ipython_extension(shell)`.

Do not change the notebook command to `%load_ext provenance_agent.magic` unless
a later, separately approved API decision chooses to migrate the extension
name.

## Removal of `sys.path` mutations

After editable installation, remove all repository-owned `sys.path.insert` and
`sys.path.append` setup from:

- tests;
- source modules;
- benchmark imports if they are touched solely to use the installed package;
- notebook code cells; and
- generated provenance cells.

The structural scan must parse Python source and notebook code-cell `source`,
not raw notebook JSON, markdown, or stored output. It must reject old flat
module imports such as `from bibliography import ...` and
`from data_workflow import ...`, while allowing historical prose and the
intentional top-level `provenance` shim.

Run package import tests from a temporary working directory so the repository
root cannot mask missing installation metadata.

## Notebook organization

Notebook reorganization is a later phase, not a prerequisite for installing
the package or verifying its direct Python APIs.

Apply the following repository-relative layout. The move table is part of the
migration and must be used to update path references:

```text
notebooks/
├── demos/
│   ├── software_workflow.ipynb       # notebooks/workflow.ipynb
│   ├── data_workflow.ipynb           # notebooks/testing/data_workflow.ipynb
│   ├── overall_workflow.ipynb        # notebooks/overall_workflow.ipynb
│   └── provenance_magic.ipynb        # notebooks/provenance_magic.ipynb
├── examples/
│   ├── paleoPCAlite.ipynb            # notebooks/testing/paleoPCAlite.ipynb
│   ├── paleoPCA.ipynb                # notebooks/testing/paleoPCA.ipynb
│   ├── C02_b_DA_with_individual_seasonality.ipynb  # notebooks/C02_b_DA_with_individual_seasonality.ipynb
│   └── comparing-simulated-reconstructed-climate/
├── fixtures/
│   ├── sample.ipynb                  # notebooks/sample.ipynb
│   ├── test_magic_commands.ipynb     # notebooks/test_magic_commands.ipynb
│   ├── Pages2k/*.lpd                 # notebooks/testing/Pages2k/*.lpd
│   └── mybiblio.bib                  # notebooks/testing/mybiblio.bib
└── exploration/
    ├── LIPD.ipynb                    # notebooks/testing/LIPD.ipynb
    ├── PyleoTUPS.ipynb               # notebooks/testing/PyleoTUPS.ipynb
    ├── Graph.ipynb                   # notebooks/Graph.ipynb
    ├── dataset_pipeline.ipynb        # notebooks/testing/dataset_pipeline.ipynb
    └── test1.ipynb                   # notebooks/testing/test1.ipynb
```

The existing `comparing-simulated-reconstructed-climate/` directory moves
under `notebooks/examples/` without changing its internal notebook names.

For each moved notebook:

- update package imports and all relative fixture paths;
- update `benchmark/ground_truth/*.yml` `notebook:` paths when applicable,
  without changing expected software or dataset entries;
- clear stale generated provenance cells whose imports name deleted modules;
- regenerate canonical generated cells with the new package imports where the
  notebook is intended to demonstrate the workflow;
- clear stored outputs that encode paths or obsolete imports when they are
  stale; and
- preserve markdown explanations unless they describe an API that was removed
  by the current cleanup, in which case rewrite the explanation against the
  surviving API.

Do not execute remote retrieval or analysis while rewriting notebooks. Use
`nbformat` and source-level transformations. Demo execution remains a manual
step because it requires API keys and remote services.

## Migration phases

### Phase 1: local package and resource foundation

- Add `pyproject.toml` and package discovery.
- Move `Citations/` into package resources.
- Create `provenance_agent/__init__.py` with only the direct function exports.
- Establish package-qualified internal imports while preserving current module
  filenames where that makes the foundation change smaller.
- Add the top-level `provenance.py` extension shim.
- Install with `pip install -e .` and keep the package/direct-API tests green in
  the `lang` environment.

### Phase 2: canonical API and module cleanup

- Move direct functions and tools into `data.py` and `software.py`.
- Rename the remaining modules to their responsibility-based names.
- Update agent, tests, documentation, and generated-cell templates.
- Update all tracked Python imports to canonical package paths.
- Delete `orchestrator.py` only after no active repository consumer imports it.
- Add structural import and direct/tool invocation tests.

### Phase 3: notebook migration

- Apply the notebook move table.
- Rewrite package imports and relative paths.
- Clear or regenerate stale generated cells.
- Update path-only ground-truth references.
- Run notebook JSON structural validation without executing remote cells.

### Phase 4: verification and handoff

- Install from a temporary working directory using the `lang` environment.
- Run the full test suite and package smoke tests.
- Run benchmark scoring tests without changing their semantics.
- Have Brian manually run the demo notebooks that require API keys or remote
  retrieval.

## Acceptance criteria

The migration is complete when:

1. `/opt/anaconda3/envs/lang/bin/pip install -e .` succeeds in the `lang`
   environment and `pip check` is clean; no public package publication is
   required.
2. Package imports work from outside the repository root.
3. `from provenance_agent import cite_data, cite_software` works without
   importing `provenance_agent.agent`.
4. The canonical direct and tool APIs work with the existing signatures and
   silently ignored `fmt`.
5. `%load_ext provenance` still registers the magic.
6. No active source, test, or notebook code cell contains a `sys.path` mutation.
7. No active source, test, or notebook code cell imports retired flat module
   names.
8. Citation lookup works using packaged `Citations/` resources from outside
   the repository.
9. Deterministic detector results, diagnostics, deprecated LLM helper tests,
   and PyleoTUPS no-op warning behavior are unchanged.
10. Notebook paths and path-only ground-truth references resolve.
11. The full test suite passes, with no benchmark scoring or ground-truth
    semantics changed.

## Risks and safeguards

- **Package data omission:** test citation lookup from outside the checkout and
  inspect the built artifact for `Citations/` files.
- **Import cycles:** keep `__init__.py` limited to direct workflow functions;
  test importing `provenance_agent` before importing `agent`.
- **Stale generated cells:** structurally scan and clear/regenerate cells before
  testing notebook execution.
- **Broken relative paths:** validate every moved notebook and every fixture
  path referenced in code cells.
- **Public API breakage:** delete `orchestrator.py` only after the canonical
  import scan is clean; document the migration in active documentation.
- **Magic regression:** install and test the top-level `provenance.py` shim
  separately from the package implementation.
- **User worktree damage:** operate only on tracked migration targets and
  preserve unrelated modified or untracked files.

This migration should be implemented under its own branch and reviewed as a
pull request before merging.
separate change from the in-place cleanup.
