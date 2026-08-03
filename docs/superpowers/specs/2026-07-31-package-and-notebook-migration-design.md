# Package and notebook migration design

This is a future migration specification. It expands the future-work items in
[`2026-07-31-refactor-plan.md`](./2026-07-31-refactor-plan.md) and is not part
of the current in-place cleanup.

The migration will turn the repository into an installable local Python
package for use by Jupyter kernels and future consumers, move the
implementation behind stable package-qualified imports, preserve the existing
`%load_ext provenance` notebook command, and eventually reorganize notebooks
without changing dataset-detection behavior or ground-truth data. It is not
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
   workflow behavior, tool invocation, notebook mutation rules, and
   ground-truth data remain stable. The intentional migration changes are package-
   qualified imports, responsibility-based module names, removal of
   `orchestrator.py`, and notebook path updates.
5. **Keep one active data-detection path.** Deterministic detection is the
   production path. The LLM detector is retained only as an explicitly marked
   fallback and must not be called by normal detection.
6. **Keep scope isolated.** Do not mix algorithm changes, ground-truth
   changes, APA rendering changes, or unrelated user-worktree cleanup into this
   migration. Those belong to separate tasks.

## Goals

- Install the project and all supported workflow/test dependencies into the
  local `lang` environment with one command: `pip install -e ".[dev]"`. This
  is a local development installation requirement, not a publication
  requirement.
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
- Do not change the expected software or dataset entries in
  `benchmark/ground_truth/*.yml`. Path-only edits to their `notebook:` fields
  are allowed when notebooks move.
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

During Phase 1, the package may temporarily contain the current internal names
`notebook_parser.py`, `bibliography.py`, `data_workflow.py`,
`software_workflow.py`, and `orchestrator.py`. These are package-internal names
only, not compatibility modules at the old top level. Phase 2 renames the first
four to the final names shown above and deletes `orchestrator.py`.

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
]

[project.optional-dependencies]
dev = [
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

This configuration supports local editable installation and future environments;
it does not require a public package release. The development installation
`pip install -e ".[dev]"` covers the provenance package, test tooling, notebook
magic, and dependencies required by generated citation cells. The base package
dependencies in `[project].dependencies` are the canonical list; `pytest` is
development-only and belongs in the `dev` extra.

The package installation does not attempt to install every scientific library
that an arbitrary user notebook might analyze. Notebook analysis libraries
beyond the generated retrieval-cell dependencies, such as `pyleoclim`,
`xarray`, and `eofs`, remain dependencies of the notebook's own kernel and are
out of scope for the package's one-command install. `pylipd` and `pyleotups`
remain package dependencies because generated retrieval cells require them in
the supported target-notebook environment.

The local `src/.env` file remains developer configuration and must not be
packaged or committed. Installed use relies on environment variables or a
user-provided dotenv file. Add a packaging test that confirms no secret file is
included in the build artifact.

The migrated `provenance_agent.llm` must not keep the current
`__file__`-relative dotenv lookup. Use a working-directory-first lookup and a
source-tree fallback so both installed use and the existing checkout work:

```python
dotenv_path = find_dotenv(usecwd=True) or find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
```

The working-directory lookup supports a user-provided `.env` when the package
is installed; the fallback continues to discover the existing `src/.env`
during the migration. Update `.gitignore` to ignore both `.env` and `src/.env`,
and keep environment variables higher priority than dotenv values. Do not move
credentials into the installed package.

## Citation data resources

The root `Citations/` directory cannot remain an uninstalled runtime
dependency. Move it under `src/provenance_agent/Citations/` and include its
files as package data.

Update the current bibliography implementation before its Phase 2 rename to
resolve the directory with
`importlib.resources.files("provenance_agent").joinpath("Citations")` rather
than walking from `__file__` to the repository root. Read resource files through
the `Traversable` API (`open()` or `as_file()`), not by passing the resource
object to `os.path.join`. The lookup and DataFrame results must remain
unchanged. Tests must exercise citation lookup after an editable install and
from a working directory outside the repository.

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
forward these existing public names so the magic API remains stable:
`cite`, `set_notebook_path`, `resolve_notebook_path`, and
`ProvenanceMagics`. It must not import or duplicate the old implementation.

`provenance_agent.magic` owns the implementation and imports
`provenance_agent.agent` using package-qualified imports. The package
dependencies supply the IPython requirements.

Acceptance tests must verify:

1. `import provenance` succeeds after editable installation;
2. a fake IPython shell can register the magic through
   `provenance.load_ipython_extension(shell)`; and
3. the forwarded helper names and `ProvenanceMagics` methods remain available.

Do not change the notebook command to `%load_ext provenance_agent.magic` unless
a later, separately approved API decision chooses to migrate the extension
name.

## Removal of `sys.path` mutations

After editable installation, remove all repository-owned `sys.path.insert` and
`sys.path.append` setup from:

- tests;
- source modules;
- notebook code cells; and
- generated provenance cells.

The structural scan must parse Python source and notebook code-cell `source`,
not raw notebook JSON, markdown, or stored output. It must reject old flat
module imports such as `from bibliography import ...` and
`from data_workflow import ...`, while allowing historical prose and the
intentional top-level `provenance` shim.

The scan must match on the fully resolved absolute module name from the AST,
never on a substring of the source text. Two of the retired names survive as
package-internal modules, so a naive scan produces false positives on correct
code:

- relative imports inside the package parse as
  `ImportFrom(module="dataset_detection", level=1)`, and the scan must skip any
  node with `level > 0` rather than matching `node.module` alone; and
- `from provenance_agent.dataset_detection import ...` contains a retired name
  as a substring but is the canonical form.

Flag a module only when it is imported as a top-level absolute name: `Import`
nodes whose first dotted segment is a retired name, and `ImportFrom` nodes with
`level == 0` whose first dotted segment is a retired name.

`benchmark/` contains no Python after the runner's removal, only
`ground_truth/*.yml`. It is data, not code, so it has no imports to migrate and
no `sys.path` handling to remove. Phase 3 still updates the `notebook:` paths
inside those files when notebooks move.

The scan does not inspect prose, so separately audit docstrings and markdown
for instructions that tell users to add `src/` to `sys.path`. Rewrite those
instructions to use the editable installation. This includes the former
`src/provenance.py` documentation, the software workflow documentation,
generated-cell comments, and notebook markdown.

Run package import tests from a temporary working directory so the repository
root cannot mask missing installation metadata.

## Notebook organization

Notebook reorganization is a later phase, not a prerequisite for installing
the package or verifying its direct Python APIs.

Apply the following repository-relative layout. The move table is part of the
migration and must be used to update path references:

The new files currently located under `notebooks/testing/` are organized by
purpose: `02a-query_lipd_graph.ipynb` is a curated LiPDGraph tutorial and moves
with the examples, while the four `Instruction Notebooks/NotebookN` bundles
become a separate instruction/evaluation collection. Each instruction bundle
keeps its local `.lpd` file beside its notebook so the exercise remains
self-contained; those files are not flattened into the shared fixtures.

The tutorial and instruction bundles are committed migration targets. Preserve
their committed content while moving them in Phase 3, and do not mix unrelated
notebook edits into the migration change.

Each bundle previously also contained a `notebookN.zip` archive holding a
duplicate copy of the notebook and its `.lpd`. Those archives have been deleted;
each bundle now contains one functional notebook and one `.lpd`; ignored
`.DS_Store` artifacts are excluded from that count. No zip handling is required
during the move.

```text
notebooks/
├── demos/
│   ├── software_workflow.ipynb       # notebooks/workflow.ipynb
│   ├── data_workflow.ipynb           # notebooks/testing/data_workflow.ipynb
│   ├── overall_workflow.ipynb        # notebooks/overall_workflow.ipynb
│   └── provenance_magic.ipynb        # notebooks/provenance_magic.ipynb
├── examples/
│   ├── 02a-query_lipd_graph.ipynb    # notebooks/testing/02a-query_lipd_graph.ipynb
│   ├── paleoPCAlite.ipynb            # notebooks/testing/paleoPCAlite.ipynb
│   ├── paleoPCA.ipynb                # notebooks/testing/paleoPCA.ipynb
│   ├── C02_b_DA_with_individual_seasonality.ipynb  # notebooks/C02_b_DA_with_individual_seasonality.ipynb
│   └── comparing-simulated-reconstructed-climate/
├── instructions/
│   ├── Notebook1/
│   │   ├── notebook1.ipynb           # notebooks/testing/Instruction Notebooks/Notebook1/notebook1.ipynb
│   │   └── MD98_2176.Stott.2007.lpd  # notebooks/testing/Instruction Notebooks/Notebook1/MD98_2176.Stott.2007.lpd
│   ├── Notebook2/
│   │   ├── notebook2.ipynb           # notebooks/testing/Instruction Notebooks/Notebook2/notebook2.ipynb
│   │   └── Vostok.Bazin.2013.lpd     # notebooks/testing/Instruction Notebooks/Notebook2/Vostok.Bazin.2013.lpd
│   ├── Notebook3/
│   │   ├── notebook3.ipynb           # notebooks/testing/Instruction Notebooks/Notebook3/notebook3.ipynb
│   │   └── Ng.EW9209-1JPC.2018.lpd   # notebooks/testing/Instruction Notebooks/Notebook3/Ng.EW9209-1JPC.2018.lpd
│   └── Notebook4/
│       ├── notebook4.ipynb           # notebooks/testing/Instruction Notebooks/Notebook4/notebook4.ipynb
│       └── Botuvera.Brazil.2005.lpd  # notebooks/testing/Instruction Notebooks/Notebook4/Botuvera.Brazil.2005.lpd
├── fixtures/
│   ├── sample.ipynb                  # notebooks/sample.ipynb
│   ├── test_magic_commands.ipynb     # notebooks/test_magic_commands.ipynb
│   ├── Pages2k/*.lpd                 # notebooks/testing/Pages2k/*.lpd
│   ├── Ocn-Palmyra.Nurhati.2011.lpd  # notebooks/testing/Ocn-Palmyra.Nurhati.2011.lpd
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
The `Instruction Notebooks/` parent directory is replaced by
`notebooks/instructions/`; preserve each `NotebookN/` bundle and its filenames.
After all listed files and directories have moved, including the newly added
tutorial and instruction bundles, remove the now-empty `notebooks/testing/`
directory; do not leave a second fixture or instruction location.

For each moved notebook:

- update package imports and all relative fixture paths;
- preserve the instruction bundles as self-contained units, and verify rather
  than rewrite their sibling `.lpd` references. Each bundle moves intact and the
  notebooks load their data by bare sibling name (`data_path =
  'MD98_2176.Stott.2007.lpd'`, `lipd.load('Vostok.Bazin.2013.lpd')`,
  `D.load('./Botuvera.Brazil.2005.lpd')`), so those paths stay correct after the
  move and editing them would break them;
- update `benchmark/ground_truth/*.yml` `notebook:` paths when applicable,
  without changing expected software or dataset entries;
- update every repository file that hardcodes the notebook's path, which
  includes test fixture constants as well as ground truth (see the table
  below);
- clear stale generated provenance cells whose imports name deleted modules;
- regenerate canonical generated cells with the new package imports where the
  notebook is intended to demonstrate the workflow;
- clear stored outputs that encode paths or obsolete imports when they are
  stale; and
- preserve markdown explanations unless they describe an API that was removed
  by the current cleanup, in which case rewrite the explanation against the
  surviving API.

The newly added notebooks, both the instruction bundles and
`02a-query_lipd_graph.ipynb`, do not have ground-truth files. Do not add any
as part of this organization change.

The test suite hardcodes notebook paths that this move table invalidates. These
references are known at planning time and must be updated in the same change as
the moves:

| Reference | Current path | New path |
|---|---|---|
| `tests/test_agent.py:23` (`SAMPLE`) | `notebooks/sample.ipynb` | `notebooks/fixtures/sample.ipynb` |
| `tests/test_notebook_parser.py:31` (`SAMPLE`) | `notebooks/sample.ipynb` | `notebooks/fixtures/sample.ipynb` |
| `tests/test_notebook_parser.py:32` (`MAGIC_NB`) | `notebooks/test_magic_commands.ipynb` | `notebooks/fixtures/test_magic_commands.ipynb` |
| `tests/test_deterministic_dataset_detection.py` (`paleoPCAlite` case) | `notebooks/testing/paleoPCAlite.ipynb` | `notebooks/examples/paleoPCAlite.ipynb` |
| `tests/test_deterministic_dataset_detection.py` (`paleoPCA` case) | `notebooks/testing/paleoPCA.ipynb` | `notebooks/examples/paleoPCA.ipynb` |

`tests/test_software_workflow.py` reads the same `sample.ipynb` fixture, and
`tests/test_notebook_parser.py` describes both fixtures in its module docstring;
update those with the constants. Re-scan for hardcoded `notebooks/` paths across
`tests/`, `src/`, and `benchmark/` after the moves rather than relying on this
table alone.

Do not execute remote retrieval or analysis while rewriting notebooks. Use
`nbformat` and source-level transformations. Demo execution remains a manual
step because it requires API keys and remote services.

## Migration phases

### Phase 0: preconditions

This migration runs after the terminal-dataframe detection work in
[`2026-08-03-terminal-dataframe-detection-design.md`](./2026-08-03-terminal-dataframe-detection-design.md),
not alongside it. One condition must hold before Phase 1 begins.

Deterministic-detection changes must be finished and committed. That work is an
algorithm change and is out of scope here by the non-goals above, but
`src/deterministic_dataset_detection.py` is a Phase 1 move target, so carrying
uncommitted edits into the move mixes an algorithm diff with a rename diff and
makes both harder to review or revert. `tests/test_deterministic_dataset_detection.py`
is edited by both tasks, the detection task changing cases and Phase 3
rewriting its two `paleoPCA` notebook paths, which is a second reason to let
detection settle first.

There is no benchmark precondition. The runner has been deleted, so detection
behavior is verified by the test suite alone and this migration has no scores
to hold steady.

### Phase 1: local package and resource foundation

- Add `pyproject.toml` and package discovery.
- Update `.gitignore` for `.env` and `src/.env`. Before installing into the
  existing `lang` environment, perform the required pip dry-run and
  dependency-gap check.
- Move `Citations/` into package resources.
- Create `provenance_agent/__init__.py` with only the direct function exports,
  temporarily importing those functions from `.orchestrator`.
- Move the current source modules, except the magic entry point, into
  `provenance_agent/` under their current internal filenames, remove the old
  top-level copies, and use package-qualified internal imports. Do not add
  compatibility shims for those old module names.
- Keep `orchestrator.py` temporarily as the internal
  `provenance_agent.orchestrator` implementation so the package root can
  re-export the direct functions; remove it after Phase 2 moves that logic into
  `data.py` and `software.py`.
- Update the current bibliography implementation to use packaged citation
  resources and add its outside-the-repository lookup test.
- Move the current LLM implementation's dotenv loading to the explicit
  `find_dotenv(usecwd=True)` plus fallback strategy. Test that the existing
  checkout credential location and an installed working-directory `.env` are
  both discoverable without packaging either secret file.
- Update generated-cell templates and their tests to use the Phase 1 package
  names, such as `from provenance_agent.bibliography import ...`.
- Before installing the shim, move the current implementation in
  `src/provenance.py` into `provenance_agent/magic.py` and update its imports;
  then replace `src/provenance.py` with the forwarding shim. Both the target
  module and the shim are Phase 1 deliverables, so the shim never points at a
  module that is deferred to Phase 2.
- Install with `pip install -e ".[dev]"` and keep the package/direct-API tests
  green in the `lang` environment.

### Phase 2: canonical API and module cleanup

- Move direct functions and tools into `data.py` and `software.py`.
- Rename the remaining modules to their responsibility-based names.
- Update agent, tests, documentation, and any remaining generated-cell
  references after the renames.
- Update all tracked Python imports to canonical package paths.
- Rewrite stale path/setup instructions in source docstrings and active
  documentation. Update `README.md` with the Python prerequisite,
  `pip install -e ".[dev]"`, direct API, and `%load_ext provenance` workflow.
  Update `CLAUDE.md` so it describes the package-qualified layout and imports
  while preserving the required `lang` interpreter instruction.
- Delete `orchestrator.py` only after no active repository consumer imports it.
- Add structural import and direct/tool invocation tests.

### Phase 3: notebook migration

- Apply the notebook move table.
- Rewrite package imports and relative paths.
- Clear or regenerate stale generated cells.
- Update path-only ground-truth references.
- Update the hardcoded notebook paths in `tests/` listed in the test-reference
  table, then re-scan `tests/`, `src/`, and `benchmark/` for any remaining
  hardcoded `notebooks/` path.
- Run notebook JSON structural validation without executing remote cells.

### Phase 4: verification and handoff

- Install from a temporary working directory using the `lang` environment and
  `pip install -e ".[dev]"`.
- Run the pip dry-run/dependency check before installation so Conda-managed
  paleoclimate packages are not unexpectedly altered.
- Run the full test suite and package smoke tests.
- Confirm `benchmark/ground_truth/*.yml` still resolves every `notebook:` path
  after the Phase 3 moves. There is nothing to score; the files are data only.
- Have Brian manually run the demo notebooks that require API keys or remote
  retrieval.

## Acceptance criteria

The migration is complete when:

1. `/opt/anaconda3/envs/lang/bin/pip install -e ".[dev]"` succeeds in the
   `lang` environment and `pip check` is clean; no public package publication
   is required.
2. Package imports work from outside the repository root.
3. `from provenance_agent import cite_data, cite_software` works without
   importing `provenance_agent.agent`.
4. The canonical direct and tool APIs work with the existing signatures and
   silently ignored `fmt`.
5. `%load_ext provenance` still registers the magic.
6. No active source, test, or notebook code cell contains a `sys.path`
   mutation.
7. No active source, test, or notebook code cell imports retired flat module
   names (`notebook_parser`, `bibliography`, `software_workflow`,
   `data_workflow`, `dataset_detection`, `deterministic_dataset_detection`,
   `orchestrator`, `agent`, or `llm`); only the intentional top-level
   `provenance` extension shim is allowed.
8. Citation lookup works using packaged `Citations/` resources from outside
   the repository.
9. Deterministic detector results, diagnostics, deprecated LLM helper tests,
   and PyleoTUPS no-op warning behavior are unchanged. "Unchanged" means
   identical to the Phase 0 state: run the detector over a fixed set of
   notebooks before and after the migration change and require the same output.
   This migration must not alter detection, and it must not roll back detection
   work that landed before Phase 0.
10. Notebook paths, test fixture path constants, and path-only ground-truth
    references all resolve; no repository file references a pre-move
    `notebooks/` path.
11. The editable install does not unexpectedly change Conda-managed
    paleoclimate package versions.
12. The full test suite passes, with no ground-truth entries changed beyond
    notebook paths; active documentation no longer instructs users to add
    `src/` to `sys.path`.

## Risks and safeguards

- **Package data omission:** test citation lookup from outside the checkout and
  inspect the built artifact for `Citations/` files.
- **Credential lookup regression:** test the cwd-first `find_dotenv` behavior
  with the existing `src/.env` checkout file and with an installed-package
  working-directory `.env`; ensure neither file is included in the artifact.
- **Conda/pip dependency conflict:** run `pip install --dry-run -e ".[dev]"`
  before installation. If pip proposes changing Conda-managed paleoclimate
  packages, manually verify the already-installed dependency set and use
  `pip install --no-deps -e ".[dev]"`, followed by `pip check`.
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
- **Stale setup documentation:** audit `README.md`, `CLAUDE.md`, source
  docstrings, generated-cell comments, and notebook
  markdown for obsolete flat-import or `src/` path instructions.
- **User worktree damage:** operate only on tracked migration targets and
  preserve unrelated modified or untracked files.

Implement this migration under its own branch. Review it as a pull request
before merging, and keep it separate from the in-place cleanup.
