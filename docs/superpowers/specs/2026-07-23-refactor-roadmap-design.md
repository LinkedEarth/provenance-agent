# Refactor roadmap: package, restructure, typed domain, notebooks cleanup

## Context

The provenance agent was built issue-by-issue and is well-factored at the
function level, but it has accumulated structural debt. This roadmap is
re-based on the current implementation rather than the older merge point.

As of 2026-07-28, the committed baseline is branch
`provenance-magic-working` at **`9e40140`**. The branch contains the workflow
and agent stack that was added after `main` remained at `68a6f05`:

- software and data citation workflows inject notebook cells rather than
  returning citation text;
- software injects one self-displaying `provenance_software` DataFrame cell;
- data injects one `provenance_datasets` cell containing one retrieval fragment
  per detected dataset and a final concatenation/display step. The cell's
  visible and returned runtime result is the source metadata DataFrame from
  the BibTeX retrieval (`get_bibtex()`/`get_publications()`); each fragment
  still leaves its raw `_bib_{variable}` value bound in the kernel;
- there is no new combined `provenance_bibliography` cell;
- each workflow removes its own prior generated cell before injecting a new one,
  and removes legacy combine cells for migration compatibility;
- generated cells are excluded from subsequent software parsing and dataset
  detection; and
- the LCEL agent verifies that expected workflow cells are present in the final
  notebook, so an idempotent rerun is still reported as successful even when it
  adds no new cell.

The remaining structural debt is:

- **No packaging.** A flat `src/` is reached through `sys.path` mutations in
  source entry points, tests, the benchmark, and tracked notebooks. `llm.py`
  builds the Gemini client at import time, which makes imports depend on API
  credentials and forces internal and LLM imports to be deferred.
- **Names that mislead.** `orchestrator.py` does not orchestrate - it holds the
  two tools; `agent.py` does the routing. `notebook_parser.py` also holds the
  shared raw-code reader used by dataset detection. `bibliography.py` mixes
  software-specific collection with shared rendering and assembly.
- **Stringly-typed domain.** Dataset detection and the data workflow still use
  `[variable, tool]` lists. The data workflow also has a legacy `variable`
  filter alongside the public `targets` filter. The direct workflow results
  are lists of software names versus dataset pairs, while the agent has its own
  typed routing envelope.
- **Duplication.** `generate_bibliography` re-implements the library
  collect/render and missing-item policy that the software path also needs.
- **Messy notebooks.** The repository mixes demos, canonical examples, test
  fixtures, exploratory notebooks, generated cells, and scratch/output files.

The goal is to pay this down without changing the current cell-injection
behavior where avoidable, in reviewable phases verified by the existing test
suite.

The current working tree also contains modified notebooks, untracked editor
configuration, and an untracked generated notebook. Those local artifacts are
not part of the refactor baseline and must not be staged accidentally. Their
source cells and generated outputs are inputs to the notebook-cleanup phase,
not a reason to preserve stale flat imports or the retired combine-cell layout.

## Decisions (from brainstorming)

- Scope: packaging and imports, typed domain model and API, deduplication and
  boundaries, file naming/one-file-one-thing, and `notebooks/` cleanup. A new
  README or CLI is out of scope; keep the per-module `__main__` blocks while
  removing their `sys.path` hacks.
- API churn: **open to renames and a typed API**; update all notebooks and tests
  to match. Keep `targets` as the user-facing selection name because it is now
  used consistently by the agent and `cite_data`; retire the legacy
  `variable` alias after callers migrate.
- Starting point: the refactor branches from the current committed baseline
  `9e40140`, not from `main` at `68a6f05`. Any local notebook or scratch changes
  are preserved and reviewed separately before staging.
- Output contract: preserve two independent workflow cells. There is no target
  combined bibliography frame. The data cell's visible/returned runtime result
  is the source metadata DataFrame returned alongside the BibTeX payload; raw
  BibTeX remains available through the per-variable `_bib_*` bindings but is
  not printed by the default path. BibTeX is the default at the user-facing
  `RouteDecision`, `cite_data`, and StructuredTool boundaries. The accepted
  `fmt` compatibility parameter is currently output-neutral in the
  DataFrame-only cell; its removal or replacement, including any future APA
  representation, is explicitly handled in Phase 2. A data retrieval failure
  stops the single data cell, which is the accepted tradeoff for one stable
  cell per workflow.
- LLM behavior: make the Gemini client lazy, but keep the model, temperature,
  and response normalization unchanged.
- Notebooks: keep every real notebook (including `LIPD.ipynb`,
  `PyleoTUPS.ipynb`, `Graph.ipynb`, `paleoPCA.ipynb`, `dataset_pipeline.ipynb`,
  and `test1.ipynb`); corral exploratory ones into `exploration/`. Keep the
  heavy fixtures (`C02_b`, `comparing-*/`) as examples. Generated notebook
  output is not a canonical fixture.
- Dataset API shapes: PyLiPD returns `(list[str], DataFrame)` while PyleoTUPS
  returns `(BibliographyData, DataFrame)`. Retrieval code preserves the source
  metadata DataFrame shape for `provenance_datasets` and keeps the citation
  payload bound for optional downstream use; the current BibTeX output path
  does not stringify it for printing or render it to APA.

## Target `src/provenance_agent/` package (was flat `src/`)

The package uses a `src` layout (`src/provenance_agent/`): the flat modules in
`src/` move into a package nested under `src/`. Code is split into two domain
subpackages so each side is obvious, with a parallel structure:
**identify → look up → cite**. Shared primitives and the two entry points
(`agent`, `magic`) stay at the top level.

```
src/provenance_agent/
├── __init__.py          # public API re-exports (cite_software, cite_data, run, ...)
├── llm.py               # SHARED: lazy get_llm() + message_text + bibtex_to_apa
├── notebook_io.py       # SHARED: notebook reading, directives, generated-cell lifecycle
├── bibliography.py      # SHARED output: rendering/assembly and compatibility helpers
├── result_type.py       # SHARED type: CitationResult                            (Phase 2)
│
├── software/            # everything touching SOFTWARE citations
│   ├── __init__.py
│   ├── libraries.py     #   identify: extract_libraries, parse_notebook, validate_libraries
│   ├── citations.py     #   look up:  collect_library_entries (Citations/ lookup)
│   └── tool.py          #   cite_software() + cite_software_tool (StructuredTool)
│
├── data/                # everything touching DATASET citations
│   ├── __init__.py
│   ├── dataset_type.py  #   Dataset(variable, tool)                             (Phase 2)
│   ├── detection.py     #   identify: LLM dataset detection
│   ├── retrieval.py     #   look up: endpoint lift and retrieval-cell builders
│   └── tool.py          #   cite_data() / generate_data_workflow() + cite_data_tool
│
├── agent.py             # NL router (imports cite_software_tool / cite_data_tool)
└── magic.py             # %provenance                                           (was provenance.py)
```

`notebook_io.py` owns `PROVENANCE_CELL_MARKER`, generated-cell detection,
`read_notebook_code`, `strip_ipython_directives`, and the helpers that remove a
workflow's old frame or a legacy combine cell. This keeps cell lifecycle and
dependency discovery out of bibliography rendering.

`bibliography.py` owns shared rendering/assembly. The standalone
`render_bibtex_strings_to_df` compatibility helper may remain during the
refactor, but it is not used to create a combined software/data frame. Software
citation collection belongs in `software/citations.py`.

Each `tool.py` owns its domain's cite function and its StructuredTool wrapper;
there is no separate top-level `tools.py`. The generated cells retain the
current contract:

- the software cell binds and displays `provenance_software`;
- the data cell contains all detected retrieval fragments and displays only the
  concatenated source metadata frames as `provenance_datasets`; and
- no workflow creates or depends on `provenance_bibliography`.

No import cycles:
`agent` → {`software.tool`, `data.tool`, `llm`}; `software.tool` →
{`software.libraries`, `software.citations`, `bibliography`, `notebook_io`};
`data.tool` → {`data.detection`, `data.retrieval`, `notebook_io`};
`data.detection` → {`llm`, `notebook_io`}; `data.retrieval` →
{`bibliography`, `notebook_io`}; and `magic` → `agent`.

`Citations/` stays at repo root. Because module depth now varies (src layout +
subpackages), resolve its path **once**, anchored to the package root - e.g. in
`__init__.py`: `_CITATIONS_DIR = Path(__file__).resolve().parents[2] / "Citations"`
(`src/provenance_agent` → `src` → repo root) - and import that constant where
needed, instead of each module recomputing `../..`. This intentionally makes
Phase 1 an **editable-checkout-only** package: the code runs from the
repository with `pip install -e`, while building/installing a standalone wheel
is not yet supported or verified. Moving `Citations/` into package data and
reading it via `importlib.resources` is deferred to a future packaging phase.

## Target `notebooks/`

```
notebooks/
├── demos/          software_workflow.ipynb (was workflow.ipynb), data_workflow.ipynb
│                   (was testing/data_workflow.ipynb), overall_workflow.ipynb, provenance_magic.ipynb
├── examples/       paleoPCAlite.ipynb, paleoPCA.ipynb, C02_b_DA_with_individual_seasonality.ipynb,
│                   comparing-simulated-reconstructed-climate/
├── fixtures/       sample.ipynb, test_magic_commands.ipynb, Pages2k/*.lpd, mybiblio.bib
└── exploration/    LIPD.ipynb, PyleoTUPS.ipynb, Graph.ipynb, dataset_pipeline.ipynb, test1.ipynb
```

Notebook moves happen only in Phase 3. Phase 1 migrates imports in every
tracked notebook while preserving physical paths. When checking for stale
imports or `sys.path` calls, parse each notebook and inspect code-cell
`source`; do not grep raw JSON, markdown, execution output, or tracebacks.

Delete only confirmed junk during Phase 3: the empty `testjunk.ipynb`, the
untracked generated `paleoPCAlite_with_citations.ipynb`, and `.DS_Store` files.
Do not stage or delete those local artifacts as part of Phase 1. Before
deleting the stray `testing/Ocn-Palmyra.Nurhati.2011.lpd`, verify that no
notebook references it; retain it if one does.

## Typed domain sketch (Phase 2)

```python
class Dataset(NamedTuple):      # replaces [variable, tool] list pairs
    variable: str
    tool: str

@dataclass
class CitationResult:           # unifies direct workflow return shapes
    kind: Literal["software", "data"]
    targets: tuple[str, ...] | tuple[Dataset, ...]
    frame: Literal["provenance_software", "provenance_datasets"]
    injected: bool
```

`Dataset` lives in `data/dataset_type.py` (a data-domain concept);
`CitationResult` lives in top-level `result_type.py` (it spans both workflows).
The result represents what was injected, not citation text: citations remain
the generated cell's runtime output. The existing `agent.TypedTarget` and
`RouteDecision` already type the natural-language routing decision and should
be reused rather than duplicated.

Phase 2 threads `Dataset` through detection, retrieval, and the data tool. It
keeps `targets` as the public selection parameter, removes the legacy
`variable` alias, and uses names such as `detected_datasets` internally so a
selection list is not confused with detected domain objects. `magic` and
`agent` render on `CitationResult.kind`/`frame`, not on tool-name strings.

## Phased roadmap

Each phase gets its own implementation plan and is verified with
`/opt/anaconda3/envs/lang/bin/python -m pytest` (all green) before the next.

### Baseline - current workflow stack (complete)

The refactor baseline is **`9e40140`** on `provenance-magic-working`, not
`main`'s older `68a6f05`. The current stack includes the LCEL router, separate
software/data workflow cells, generated-cell filtering, standard-library
filtering, legacy combine-cell removal, idempotent workflow replacement, and
final-cell presence verification. The committed suite reports 159 passing
tests.

The working tree contains the intended `data_workflow.py` direction: raw
BibTeX/APA printing is removed and `provenance_datasets` is the only visible
data-cell result. Stabilize this contract and add its tests before Phase 1; do
not silently mix an untested output-contract change into a packaging-only
commit. The `fmt` compatibility surface remains accepted but output-neutral
until Phase 2 makes its replacement/removal decision explicit. At review time,
the full suite reports 155 passing and 5 failing tests; those failures assert
the superseded BibTeX/APA-printing behavior and are the explicit stabilization
work for this direction.

Before starting Phase 1, preserve any local notebook/editor changes, finalize
the pending data-cell decision, and ensure the refactor branch is based on
`9e40140` or a later commit containing the stabilized behavior. Do not treat
generated notebook output as a new API requirement.

### Phase 1 - Package & restructure (behavior-preserving)

The foundation. Detailed below. The two-cell injection contract, idempotent
replacement, generated-cell filtering, and agent verification are part of the
behavior to preserve. The DataFrame-only data-cell output, BibTeX defaults, and
compatibility behavior must be stabilized and tested before this phase is
considered behavior-preserving.

### Phase 2 - Typed domain & unified result (intentional API change)

Introduce `result_type.py` (`CitationResult`) and `data/dataset_type.py`
(`Dataset`); thread `Dataset` through detection/retrieval/tool; remove the
legacy `variable` alias; and unify the direct workflow result shape without
inventing a combined bibliography frame. Update the agent envelope, magic,
tests, and demo notebooks.

### Phase 3 - Notebooks cleanup

Physically apply the `notebooks/` layout and delete confirmed junk. Update every
path reference affected by those moves: tests (fixtures), the `notebook:` fields
in `benchmark/ground_truth/`, and notebooks' internal paths. Import migrations
belong to Phase 1, not this phase. Normalize generated cells and stale outputs
as part of moving the notebooks so canonical demos contain only intentional
content.

---

## Phase 1 detail (implement first)

**Behavior is preserved**; only import paths, module boundaries, and package
metadata change. Tests are updated for the new imports and must stay green.
This phase supports use from an editable repository checkout only. Root-level
`Citations/` is deliberately retained; standalone wheel/package-data support is
deferred.

1. **Packaging (src layout).** Add `pyproject.toml` declaring package
   `provenance_agent` with `[tool.setuptools.packages.find] where = ["src"]`
   and declare the runtime dependencies actually imported by the project:
   nbformat, bibtexparser, pyyaml, pandas, requests, python-dotenv,
   langchain-core, langchain-google-genai, pydantic, pylipd, and pyleotups.
   Declare optional extras explicitly:

   ```toml
   [project.optional-dependencies]
   notebook = ["ipynbname", "ipython"]
   test = ["pytest"]
   ```

   Create `src/provenance_agent/`, move the modules in and reorganize into the
   `software/` and `data/` subpackages (step 4); add package `__init__.py`
   files re-exporting the public API. Editable install into the `lang` env:
   `/opt/anaconda3/envs/lang/bin/pip install -e ".[notebook,test]"`. Do not
   add wheel/package-data work to this phase.

2. **Lazy LLM client.** In `llm.py`, replace module-level dotenv loading,
   client construction, and APA chain construction with a cached `get_llm()`
   plus a function-local APA chain (or an equally lazy cached chain). Consumers
   call the lazy accessor. Preserve `gemini-flash-latest`, temperature `0`, and
   `message_text` normalization. This removes the import-time API-key
   requirement.

3. **Remove deferred internal/LLM imports.** With no import-time client
   construction and a real package, move internal `provenance_agent` imports
   and imports of `get_llm`, `message_text`, and `bibtex_to_apa` to module
   top-level where safe. Keep optional `ipynbname` and IPython imports guarded
   inside runtime fallback/extension paths; they must continue to degrade
   cleanly when the `notebook` extra is absent.

4. **Split and rename into the software/data subpackages** per the package map:

   - `notebook_parser.py` → `notebook_io.py` (shared notebook reader,
     directive handling, generated-cell lifecycle) + `software/libraries.py`
     (import extraction and validation).
   - `bibliography.py` → `software/citations.py` (Citations/ lookup and
     library collection) + slimmed top-level `bibliography.py`
     (rendering/assembly and standalone compatibility helpers).
   - `orchestrator.py` → `software/tool.py` (`cite_software` and
     `cite_software_tool`) and `data/tool.py` (`cite_data` and
     `cite_data_tool`); no standalone `tools.py`. The shared format validator
     moves to the appropriate shared output layer.
   - `dataset_detection.py` → `data/detection.py`.
   - `data_workflow.py` → `data/retrieval.py` (endpoint lift, retrieval
     fragments, and the single data-cell builder) + `data/tool.py`
     (filtering, generation, and `cite_data`).
   - `provenance.py` → `magic.py`.
   - Add `software/__init__.py` and `data/__init__.py`.

   Preserve the current injection invariants during the move: one
   `provenance_software` cell, one `provenance_datasets` cell for all detected
   pairs, no new combine cell, removal of a workflow's prior frame before
   injection, removal of legacy combine cells, and generated-cell exclusion
   during later parsing/detection. The data cell displays only the concatenated
   metadata DataFrame in the stabilized target; raw `_bib_*` values remain
   bound for callers that need BibTeX.

5. **Deduplicate library collection without changing messages.** Add a private
   helper in the shared bibliography/output layer that owns the library
   collect→render operation (APA or BibTeX). Both `software.tool.cite_software`
   and `bibliography.generate_bibliography` call it, but each caller retains its
   distinct missing-item policy. Requested libraries absent from the notebook
   must still produce exactly `[Not imported in notebook: X]`; imported
   non-stdlib libraries without a citation record in `generate_bibliography`
   must still produce exactly `[No citation found for: X]`. Do not collapse
   these cases or alter punctuation/capitalization.

6. **Kill all `sys.path` hacks and package the benchmark locally.** Remove
   every `sys.path.insert(...)` and `sys.path.append(...)` from source, tests,
   `benchmark/`, and all tracked notebook code-cell sources, then use absolute
   `provenance_agent...` imports. Add `benchmark/__init__.py` so it is a
   repository-local package: tests import `benchmark.run_benchmark`, and the
   workflow runs from the repository root as
   `python -m benchmark.run_benchmark`. It is not part of the installed
   `provenance_agent` distribution. Generated APA/retrieval cells must import
   from `provenance_agent`, for example
   `from provenance_agent.bibliography import render_bibtex_strings_to_apa`.

7. **Update every tracked notebook's import/setup cells.** Migrate stale flat
   imports and remove `sys.path` setup from demos, examples, fixtures, and
   exploratory/testing notebooks, not only the eventual four demos. In
   particular, migrate `notebooks/testing/dataset_pipeline.ipynb` in Phase 1
   and update the `software` expectations and explanatory note in
   `benchmark/ground_truth/dataset_pipeline.yml` to match its new imports.
   Keep that notebook physically at its current path and keep the current YAML
   `notebook:` value until Phase 3. Apart from import/setup text and generated
   cell cleanup needed to remove stale instructions, notebook content and
   physical paths remain unchanged in this phase.

### Phase 1 verification

- `/opt/anaconda3/envs/lang/bin/python -m pip install -e ".[notebook,test]"`
  succeeds from the repository checkout; no wheel-install claim is made.
- `/opt/anaconda3/envs/lang/bin/python -m pip check` reports no broken
  requirements.
- `/opt/anaconda3/envs/lang/bin/python -m pytest` is all green.
- Add a lazy-client regression test that patches `ChatGoogleGenerativeAI`,
  imports `provenance_agent.llm`, `provenance_agent.agent`, and
  `provenance_agent.data.detection`, and proves the constructor is not called
  until `get_llm()` is invoked; two `get_llm()` calls return the same cached
  client. Also smoke-test imports with blank API-key variables:
  `GOOGLE_API_KEY= GEMINI_API_KEY= /opt/anaconda3/envs/lang/bin/python -c
  "import provenance_agent; import provenance_agent.agent; import
  provenance_agent.data.detection; from provenance_agent import cite_software"`.
- Add offline idempotency tests: running each workflow twice leaves exactly one
  cell for its own frame, no combine cell, and stable generated-cell counts;
  running the agent twice reports `verification.present` and succeeds on the
  second run even when `verification.mutated` is false. The magic formatter
  must report static verification passed for that unchanged rerun.
- Pin the stabilized data-cell output contract in tests: the injected cell
  displays `provenance_datasets`, preserves each source metadata schema, keeps
  `_bib_*` values available, and does not create a combined bibliography frame.
  If `fmt` is still accepted in Phase 1, test its compatibility behavior; its
  final removal or replacement belongs to Phase 2.
- Add an automated structural check over source, tests, `benchmark/`, and every
  notebook returned by `git ls-files '*.ipynb'`. Parse notebook JSON and scan
  code-cell `source` only: there must be no `sys.path.insert`/
  `sys.path.append` and no imports from retired flat module names.
- Verify the repository-local benchmark import without an API key:
  `/opt/anaconda3/envs/lang/bin/python -c "from benchmark.run_benchmark import
  score_sets"`. Add an offline test that calls `benchmark_notebook()` for
  `notebooks/sample.ipynb` with module-level `detect_datasets` monkeypatched;
  it must exercise the final imports and scoring path without a network call.
- Spot-check one module `__main__` against the Phase 1 path:
  `/opt/anaconda3/envs/lang/bin/python -m
  provenance_agent.software.libraries notebooks/sample.ipynb`.
- Brian runs the demo notebooks (API-key paths) himself, per standing
  preference.

## Risks

- **Import churn is wide** (every test, benchmark, and tracked notebook).
  Mitigate it with one coherent migration, a parsed code-cell structural scan,
  the full test suite, and the benchmark smoke workflow.
- **Editable install must be in the `lang` env** for both pytest and notebook
  kernels; generated APA cells depend on it.
- **The one data cell is intentionally atomic.** A retrieval failure for one
  detected dataset stops the remaining retrieval fragments. This is the
  current accepted contract; changing it to per-dataset cells would be a
  separate behavior change, not an incidental packaging refactor.
- **Generated-cell migration can look like a dependency change.** The parser
  must continue excluding both the current marker and legacy signatures, and
  old combine cells must be removed rather than reintroduced as a target.
- **`Citations/` path** relies on `Citations/` staying at repo root, resolved
  via the single package-anchored constant (`parents[2]`). That is valid for
  the explicitly supported editable checkout, not a standalone wheel;
  package-data support remains future work.
- **Notebook artifacts can be large or untracked.** Do not stage generated
  outputs or editor files during Phase 1; Phase 3 handles confirmed cleanup
  and path updates.
- Phases 2-3 change public API and notebook paths; each gets its own spec.
