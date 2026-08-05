# Design Decisions

This document answers **why** the provenance agent is built the way it is. For
**what** it does and how to use it, see
[documentation-draft.md](documentation-draft.md). For how it would fold into
PaleoPAL, see [paleopal-integration.md](paleopal-integration.md).

Part 1 covers the decisions that shape the whole project and live between files.
Part 2 condenses the decisions recorded in each module's docstring, one section
per file. Where a summary here is not enough, the module docstring is the fuller
account and the source of truth.

---

## Contents

**Part 1 - Project-wide**

1. [Citations are a cell's output, not a return value](#1-citations-are-a-cells-output-not-a-return-value)
2. [Two symmetric workflows that never merge](#2-two-symmetric-workflows-that-never-merge)
3. [Detection is deterministic](#3-detection-is-deterministic)
4. [Exactly one model call exists](#4-exactly-one-model-call-exists)
5. [An installed package, with no `sys.path` anywhere](#5-an-installed-package-with-no-syspath-anywhere)
6. [Nothing constructs an LLM client at import](#6-nothing-constructs-an-llm-client-at-import)
7. [Every LLM provider is an extra](#7-every-llm-provider-is-an-extra)
8. [Standalone today, integration-ready by construction](#8-standalone-today-integration-ready-by-construction)
9. [One module per side, no orchestrator](#9-one-module-per-side-no-orchestrator)
10. [Compatibility surfaces are held open on purpose](#10-compatibility-surfaces-are-held-open-on-purpose)
11. [Report rather than drop; refuse rather than guess](#11-report-rather-than-drop-refuse-rather-than-guess)
12. [The test suite is offline and stays offline](#12-the-test-suite-is-offline-and-stays-offline)

**Part 2 - Per module**

[`__init__.py`](#__init__py) · [`provenance.py`](#provenancepy) ·
[`notebook_io.py`](#notebook_iopy) · [`citations.py`](#citationspy) ·
[`software.py`](#softwarepy) · [`data.py`](#datapy) ·
[`dataset_detection.py`](#dataset_detectionpy) ·
[`deterministic_dataset_detection.py`](#deterministic_dataset_detectionpy) ·
[`agent.py`](#agentpy) · [`magic.py`](#magicpy) · [`llm.py`](#llmpy) ·
[`Citations/`](#citations-package-data)

---

# Part 1 - Project-wide

## 1. Citations are a cell's output, not a return value

**The defining decision.** Neither workflow returns citations. Each injects one
cell into the notebook, and the citations are that cell's output when the user
runs it. `cite_software()` returns the library names it built a cell for;
`cite_data()` returns the `[variable, tool]` pairs it built a retrieval cell for.

The reason is the data side. Retrieving a dataset citation means calling
`get_bibtex()` on a `LiPD` object or `get_publications()` on a PyleoTUPS object,
and **those objects live in the user's kernel with their data already loaded.**
This agent runs outside that kernel and cannot reach in. So instead of
retrieving, it writes the retrieval code into the notebook and hands it back.

The software side has no such constraint - its citations come from packaged
files and could have been returned directly. It follows the same shape anyway,
so both workflows have one contract, one mental model, and one thing to explain.

Two consequences worth naming, because they surprise people:

- The notebook is modified in place unless `output_path` is given, and it must be
  reloaded before the new cell is visible.
- Nothing is verified at injection time. `agent.run()`'s `verify` stage diffs the
  notebook before and after; it does not execute anything.

Everything else in the project follows from this. In particular, it is why the
cell *builders* are public and separate from the *injectors*: a host that does
have a kernel can use the builders and skip the injection entirely. See
[paleopal-integration.md §3.2](paleopal-integration.md#32-there-is-a-kernel-which-changes-the-design).

## 2. Two symmetric workflows that never merge

Software and data are separate questions with separate answers, and they stay
separate all the way down.

| | Software | Data |
|---|---|---|
| Finds targets by | `ast` import scan | data-flow analysis |
| Citations come from | packaged `Citations/` | `get_bibtex()` / `get_publications()` at run time |
| Needs a live kernel | no | yes, to retrieve |
| Owns | one cell | one cell, covering every source |

Each workflow owns **exactly one** cell, and re-running replaces its own cell
rather than stacking a stale second one. There is deliberately no combined
software-plus-data frame: each cell can be re-run or deleted on its own, and one
side failing does not take the other with it.

The symmetry is the point. Both sides analyze the notebook, append one cell with
`nbformat`, write the notebook back, and let the user run the cell. A reader who
understands one understands the other.

## 3. Detection is deterministic

Dataset detection used to send the notebook's code to a model and parse a JSON
list of pairs out of the reply. It is now a static AST and data-flow analysis
that never calls a model and never executes notebook code.

What that buys: the same notebook always produces the same answer, detection
costs nothing, it works with no credentials, and its failures are inspectable
(`detect_datasets_with_diagnostics()` explains every analysis call it could not
trace to a source).

What it costs: a fixed vocabulary. `ANALYSIS_METHODS` lists what counts as an
analysis boundary, recognizers list what counts as a source, and extending
either is a code change rather than a better prompt.

The analyzer is deliberately **conservative**: syntax errors, opaque helper
functions, and dynamic imports produce a diagnostic warning rather than a
guessed source. A wrong citation is worse than a missing one.

The LLM detector is **retained and inactive** - see
[§10](#10-compatibility-surfaces-are-held-open-on-purpose).

## 4. Exactly one model call exists

The only model call in the entire tool classifies one natural-language request
into a typed decision. Detection, citation lookup, cell generation, and
verification are all deterministic code.

This is why the provider choice barely matters, why a local Ollama model is a
reasonable option, why the fast cheap tier is the right default, and why
PaleoPAL's heavyweight model defaults are deliberately not inherited.

It also means the direct functions - `cite_software()` and `cite_data()` - need
no model, no API key, and no provider installed. They are the majority of the
tool, and the natural-language layer is a convenience on top.

## 5. An installed package, with no `sys.path` anywhere

Everything lives in `src/provenance_agent/`, installed editable. Modules inside
the package import each other **relatively** (`from .citations import ...`);
everything outside it - tests, the `provenance` shim, generated notebook cells -
uses the absolute package name.

This exists because the generated software cell contains
`from provenance_agent.citations import collect_library_entries` and must run in
the user's kernel. A path-relative layout would have forced `sys.path`
manipulation into every notebook. `tests/test_import_hygiene.py` fails the suite
if a `sys.path` mutation or a retired flat import reappears, in Python **or** in
a notebook code cell.

`Citations/` is package *data*, resolved through `importlib.resources`, not a
directory found by walking up from `__file__`. The walk only worked from a source
checkout; the resource lookup works from an installed package and any working
directory.

## 6. Nothing constructs an LLM client at import

`llm.llm` and `agent.chain` are resolved through PEP 562 module-level
`__getattr__` functions (`get_llm()` / `get_chain()`) and cached in `llm._CLIENT`
and `agent._CHAIN`. `agent.py` does `from . import llm as _llm`, never
`from .llm import llm`, because binding the name would read it and read is what
builds it.

The problem this solved: the client used to be constructed at module scope, so
importing `provenance_agent.agent` - and therefore `%load_ext provenance` -
failed without credentials. `.env` is untracked, so a fresh clone running the
documented `pytest tests/ -q` got collection errors from the three test modules
that import the agent, despite the suite being genuinely offline.

**Substitute a client by assigning `_CLIENT` / `_CHAIN`, never by patching `llm`
or `chain`.** Reading those names is what builds them, so patching defeats the
laziness and raises without credentials. `tests/test_packaging.py` pins the
property.

The accepted tradeoff: a missing key surfaces at first model use, not at
`%load_ext`. That is intended behavior, not an oversight.

## 7. Every LLM provider is an extra

`llm.PROVIDERS` is a five-entry registry (google, openai, anthropic, ollama, xai)
of lazy import instructions. **No integration is a core dependency, including the
default one.** Google is the default *selection*, not a default installation.

Two reasons:

- `langchain-google-genai` pulls `google-genai` and `google-auth`, which pull
  `cryptography` and `aiohttp`. This package is routinely installed into conda
  environments holding scientific builds, where extra transitive installs are a
  real hazard.
- Nobody should install a provider they will never load.

The cost is that a core install has no model client at all. That failure is made
self-correcting: `build_llm()` raises a message naming the exact `pip install`
command rather than a bare `ModuleNotFoundError`, and it checks the API key
*before* importing the package, so someone with neither is told about the key
first - the requirement they would hit again immediately afterward.

This is deliberately **not** langchain's `init_chat_model`, which lives in the
`langchain` umbrella package. This project depends on `langchain-core` plus one
integration. A five-entry dict buys the same provider-agnosticism without that
dependency.

## 8. Standalone today, integration-ready by construction

Standing alone is the requirement now; folding into PaleoPAL is the expected
destination. Both are held at once by two rules: nothing may *require* PaleoPAL,
and nothing may make dropping this into PaleoPAL awkward.

What that produced, concretely:

- `run(..., model=...)` accepts any LangChain `Runnable`, so a host passes its own
  client and `llm.py`'s registry never runs. The registry is a standalone-mode
  default, not a competing abstraction that must be deleted on integration.
- API key variable names match PaleoPAL's exactly, and `DEFAULT_LLM_PROVIDER` is
  read as a fallback. Model variables are deliberately not read.
- Citation retrieval is hardcoded here and is **not** delegated to PaleoPAL's
  Code or SPARQL agents. Delegation would make standalone mode impossible.

The dependency arrow points one way: PaleoPAL may depend on this, never the
reverse. Full notes in [paleopal-integration.md](paleopal-integration.md).

## 9. One module per side, no orchestrator

The public function, the workflow, and the cell builders for each side live in
one module. `software.py` is the whole software side; `data.py` is the whole data
side.

They used to be split across an `orchestrator` module, which meant a reader
chasing "what does `cite_data` actually do" crossed a file boundary to reach a
one-line delegation. The orchestrator is gone.

`cite_*` and `generate_*_workflow` are **one implementation with two names**:
`cite_software` / `cite_data` are the public API that callers and the agent use;
`generate_software_workflow` / `generate_data_workflow` are the step names
notebook tooling regenerates cells with. Each `cite_*` is a one-line delegation.
Do not grow a second implementation behind either name.

## 10. Compatibility surfaces are held open on purpose

Four things look like dead code and are not. Each is documented where it lives.

- **`fmt` is accepted and ignored.** `cite_data`, `generate_data_workflow`, the
  tool schema, and `RouteDecision.fmt` all take it; no value changes the output
  and no value is rejected. APA rendering existed and was removed along with the
  LLM chain that produced it. The parameter is held open for a future non-LLM APA
  path that would render from the metadata DataFrame. `RouteDecision.fmt` is a
  plain `str` rather than a `Literal` precisely so an unexpected value cannot
  fail classification.
- **The LLM detector is retained but never called.** `DETECTION_PROMPT`,
  `build_detection_prompt()`, and `parse_detection_response()` in
  `dataset_detection.py` are a deliberate rollback path. Restoring it means
  uncommenting one call. They emit no deprecation warning and their tests are kept
  so the fallback stays known-good. **They are not dead code and should not be
  deleted.**
- **Legacy generated-cell signatures are still recognized.**
  `is_generated_cell()` matches the markers older versions wrote, so notebooks
  written by an earlier version have their stale cells found and replaced rather
  than duplicated.
- **`filter_datasets()`** retains the older variable-level filter behavior
  alongside the current dataset-name targeting.

## 11. Report rather than drop; refuse rather than guess

A consistent stance on partial and ambiguous results.

**Report what is missing.** An imported library with no citation entry produces a
row whose `note` reads "No citation found for imported library" rather than
vanishing, so the gap is visible. Standard-library imports are the exception:
they are dropped entirely, because nobody cites `os` and leaving them in produced
one noise row per stdlib module.

**Refuse rather than guess.** An unclear request, a requested library the
notebook does not import, or a data request with no detected sources returns a
warning and **does not mutate the notebook**. A specific PyleoTUPS study name
cannot be resolved before the live object runs, so it warns and changes nothing
rather than injecting a retrieval that cannot work.

**Fail loudly where a guess would be silent corruption.** An unsupported tool in
a detected pair raises `ValueError` rather than producing an empty bibliography.
A retired argument raises rather than being quietly ignored.

**Distinguish empty from unknown.** `detect_datasets()` emitting no pairs *and*
no warnings means it found no analysis at all. Emitting no pairs *with* warnings
means it found analysis it could not trace to a source. Those are different
answers and the diagnostics keep them distinguishable.

## 12. The test suite is offline and stays offline

No test calls a model, a SPARQL endpoint, or a remote dataset service, and none
needs an API key. That holds on a fresh clone with no `.env`.

Two mechanisms keep it true: the lazy client construction in
[§6](#6-nothing-constructs-an-llm-client-at-import), and `build_chain(model=...)`
accepting a fake `Runnable` so routing can be exercised without a provider.
`tests/test_packaging.py` pins the credential-free property;
`tests/test_import_hygiene.py` pins the import hygiene.

---

# Part 2 - Per module

## `__init__.py`

*Package root. Re-exports and nothing else.*

- Exports `cite_data` and `cite_software`, and **only** those.
- The LangChain tools and `run` are deliberately **not** re-exported. They stay at
  `provenance_agent.agent.run`, `provenance_agent.data.cite_data_tool`, and
  `provenance_agent.software.cite_software_tool`, so importing the package root
  never pulls in the routing layer, its LangChain dependencies, or its provider
  configuration.
- Each function is imported from the module that implements it, with no routing
  module in between.

## `provenance.py`

*The one intentional top-level module, outside the package.*

- Exists solely so `%load_ext provenance` keeps working. `%load_ext` takes a
  top-level module name, and `provenance` is the command this project has always
  documented. Installed via `py-modules = ["provenance"]`.
- **A forwarding shim with no implementation of its own.** Every name here is the
  same object as the one in `provenance_agent.magic`, so behavior cannot drift
  and there is only one place to change the magic.
- Uses the absolute package name, because it is outside the package.
- The import name and the magic name differ on purpose: you `import
  provenance_agent` in Python but `%load_ext provenance` in a notebook.

## `notebook_io.py`

*Notebook reading, import extraction, and the generated-cell marker.*

- Strips IPython magics (`%`, `%%`) and shell lines (`!`) before `ast.parse()`.
  Whole-cell magics whose body is not Python (`%%bash`) are dropped entirely.
- Cells with syntax errors fall back to **line-by-line import recovery**, because
  a broken cell's imports are still real dependencies. (Note the asymmetry: the
  data-flow analyzer cannot do this and skips broken cells whole.)
- **Injected cells are excluded from both scans.** They carry imports of their own
   - the software cell imports from `provenance_agent`, a LiPDGraph retrieval cell
  imports `pylipd` - which are the tool's machinery, not the notebook's
  dependencies. Since `pylipd` has a citation on file, scanning them made a second
  run cite a library the notebook never used.
- Dataset detection is deliberately **not** here. This module is the software
  (import) side, plus `read_notebook_code()` for the data side's endpoint
  extraction.

## `citations.py`

*Software citation lookup, plus the generated-cell lifecycle helpers both sides share.*

- `Citations/` is read through `importlib.resources.files("provenance_agent")`,
  never by walking up from `__file__` to a repository root. Files are read through
  the returned `Traversable`'s `open()`, never handed to `os.path`.
- `collect_library_entries()` merges matching BibTeX into one DataFrame, deduped
  by DOI. This is what the injected software cell calls, so **that DataFrame is
  the citation output the user sees.**
- **There is no rendering layer.** APA rendering was removed with the LLM chain
  that produced it. No module renders citation text in any format.
- The cell-removal helpers live here rather than in `notebook_io` because they are
  written against the frame names this module's output is bound to. `notebook_io`
  knows how to read a notebook; only this module knows what `provenance_software`
  means.
- Collection here is **software-specific**. Dataset citations are collected by
  `data.py`, in the user's kernel.

## `software.py`

*The whole software side.*

- The injected cell **imports** `collect_library_entries` rather than baking the
  collected BibTeX inline, so it stays short and always reflects current
  `Citations/` data. The import is package-qualified, so the kernel only needs the
  package installed.
- **One cell for all libraries**, because all software citations resolve to a
  single DataFrame. There is no per-library live object to reuse.
- Citations are the cell's **output**, not this module's return value, so the
  contract matches `cite_data`.
- **No combined software-plus-data frame.** Each workflow's cell stands alone.
- Standard-library imports are dropped before the cell is built, and
  `collect_library_entries` drops them too, so a direct caller passing a stdlib
  name gets the same answer.
- When nothing matches, **no cell is injected and the notebook is untouched**,
  mirroring the data side's "nothing detected" path.

## `data.py`

*The whole data side.*

- **The LiPDGraph endpoint is lifted from the notebook by AST**, not hardcoded, so
  a notebook pointed at a different repository is handled correctly.
  `_LIPDVERSE_ENDPOINT` is only a fallback when no URL is found.
- **Untargeted retrieval reuses the notebook's already-loaded objects**
  (`{var}.{method}`); a targeted PyLiPD or LiPDGraph request builds a fresh `LiPD`
  object and loads only the requested names, so unrelated datasets are not
  fetched.
- LiPDGraph is the special case: its terminal variable is a DataFrame, so the cell
  lifts the `dataSetName` column, loads those into a fresh `LiPD` object from the
  endpoint, then calls `get_bibtex()`.
- **A specific PyleoTUPS study cannot be targeted.** Its available names exist only
  inside the live provider object, so there is nothing to match against
  statically. It warns and leaves the notebook unchanged.
- **One cell for every source, not one per dataset** - accepting that a retrieval
  failure in one source stops the whole cell.
- The tool wraps `_cite_data_tool_entry`, not `cite_data`, to keep
  `detected_pairs` out of the public tool schema. It is an internal reuse hook the
  LCEL pipeline passes so detection does not run twice, and **a model must never
  be able to supply it.**
- Results stay bound in the kernel as `_bib_{variable}` / `_meta_{variable}`, and
  each metadata frame is displayed. Both PyLiPD's `get_bibtex()` and PyleoTUPS'
  `get_publications()` return `(citations, metadata DataFrame)`.
- An unsupported tool raises `ValueError` rather than silently producing an empty
  bibliography.

## `dataset_detection.py`

*The detection facade, plus the retained LLM fallback.*

- A thin facade: `detect_datasets()` and `detect_datasets_with_diagnostics()`
  delegate to the deterministic analyzer.
- The active detector takes a **notebook path**, so the analyzer can preserve cell
  boundaries and skip generated cells. The deprecated LLM path took concatenated
  code instead, which is why the commented-out call reads differently.
- `detect_datasets()` keeps its list return contract and emits unresolved-lineage
  problems as `UserWarning`, so they surface in a notebook without being asked
  for. Callers needing structure use the diagnostics variant.
- The deprecated prompt is spliced with `str.replace("{code}", ...)` rather than
  `str.format()`, so notebook code containing braces cannot break templating.
- See [§10](#10-compatibility-surfaces-are-held-open-on-purpose) on why the
  deprecated path stays.

## `deterministic_dataset_detection.py`

*The analyzer. By far the largest module.*

- Builds a **versioned data-flow graph** over the notebook's cells: every assigned
  value records its dependencies, source groups, object family, and source
  position. Recognizers attach sources to LiPD, PyleoTUPS, LiPDGraph, xarray, and
  pandas loaders. Analysis calls are sinks. Results come from walking each sink's
  dependency closure to the nearest source-family boundary.
- **Never calls a model and never executes notebook code.**
- **A source is not reported merely because it was loaded.** Its lineage must
  reach a configured analysis sink, or produce live source-backed terminal table
  leaves. This is what makes the tool cite what was *used*, and it is the single
  most surprising behavior for new users.
- The terminal fallback is **deliberately tabular** - source-backed pandas
  DataFrames - so an xarray source that only ever produces arrays still requires a
  real analysis boundary.
- The fallback is evaluated **independently per active source**, so an unrelated
  source-backed table can be reported alongside an analyzed one. This preserves
  one result per independent query lineage.
- Unresolvable analysis calls are retained as **diagnostics**, so an empty result
  can be told apart from an unsupported loader.
- Notebook variables are **internal result labels**. User-facing targeting is
  dataset-name based, in the workflow layer.
- The analysis and inspection registries are centralized, so extending the
  scientific vocabulary does not require touching graph logic.

## `agent.py`

*The LCEL router.*

- Five **named** `Runnable` stages composed with `|`:
  `prepare_context | classify | resolve_targets | dispatch | verify`. This
  replaced hidden one-shot model tool-calling, so the pipeline is inspectable
  (`chain.get_graph()`) rather than opaque.
- Classification returns a **Pydantic-validated JSON decision**. Targets are typed
  as `software` or `data` at classification time, so dispatch never has to guess a
  target's category afterward.
- **An unclear or unsatisfiable request warns and does not mutate the notebook.**
  See [§11](#11-report-rather-than-drop-refuse-rather-than-guess).
- `verify` diffs the notebook before and after **without executing** the injected
  cells, and sets `runtime_unverified` whenever specific dataset names were
  requested, because those cannot be checked statically.
- The `StructuredTool` wrappers stay public for direct callers but are **not bound
  to the classification model.**
- `build_chain(model=...)` accepts a fake `Runnable`, which is how routing is
  tested offline.
- `run(..., model=...)` exposes the same injection point to callers, so an
  embedding application passes its own client. See
  [§8](#8-standalone-today-integration-ready-by-construction).
- Nothing is constructed at import; `chain` resolves through a module
  `__getattr__`, and `llm` is imported as a module rather than by name. See
  [§6](#6-nothing-constructs-an-llm-client-at-import).

## `magic.py`

*The IPython extension implementation.*

- **No citation or routing logic lives here.** It resolves the notebook path,
  calls `agent.run()`, and renders the envelope. It imports `agent` and nothing
  else from the project, so the routing contract stays in one place.
- Logic is module-level functions rather than methods on the `Magics` class, so
  tests exercise it without constructing an IPython shell.
- Raises `UsageError` rather than `RuntimeError` for user mistakes, because
  IPython renders it as one clean line instead of a traceback.
- The `%provenance_notebook` session override exists because `ipynbname` matches
  the running kernel against the Jupyter server's session list, which routinely
  fails in VSCode - **the primary environment for this project.** Auto-detection
  stays the default; the override is the recovery path, and in VSCode it is the
  normal one.
- Summaries report **what was injected** and tell the user to run the cells,
  rather than implying citations were produced. Empty results say so instead of
  being reported as the answer.

## `llm.py`

*Provider registry, credential discovery, and the shared client.*

- `PROVIDERS` is a **lazy import table**, not langchain's `init_chat_model` -
  which lives in the `langchain` umbrella package this project deliberately does
  not depend on.
- **No integration is a hard dependency, not even the default provider's.** See
  [§7](#7-every-llm-provider-is-an-extra).
- **Credentials are validated before the chat class is constructed.**
  Pydantic-based chat classes raise a `ValidationError` about a field name when a
  key is absent, which reads as a bug in this project rather than a missing key.
  The explicit check names the variable to set. A provider with no key variables
  (Ollama, which is local) skips the check.
- **Credentials are discovered, not packaged.** Lookup is working-directory first
  (`find_dotenv(usecwd=True)`) so an installed copy picks up the `.env` beside
  whatever project uses it, with a source-tree fallback so a plain `pytest` in
  this checkout still finds one. No `.env` is ever included in the built artifact.
- `load_dotenv` keeps its default `override=False`, so an exported variable beats
  any dotenv value. CI and shell exports win over a stale local file.
- **Notebook caveat:** python-dotenv treats a Jupyter kernel as interactive and
  resolves *both* lookups from the kernel's working directory, so a `.env` buried
  in `src/` is invisible there. Keep it at the repository root.
- `gemini-flash-latest` is used as the Google default because it is an alias that
  tracks the current Flash model, so a specific version being retired does not
  404 (as happened with `gemini-2.5-flash`).
- PaleoPAL compatibility covers credentials and vendor, **never the model.** See
  [§8](#8-standalone-today-integration-ready-by-construction).
- The client is built lazily. See
  [§6](#6-nothing-constructs-an-llm-client-at-import).

## `Citations/` (package data)

*The software citation index.*

- A YAML index (`library_citations.yml`) mapping a library to its inline `paper`
  BibTeX and/or a `software` `.bib` file beside it.
- **Curated by hand, with no automatic lookup** against GitHub, Zenodo, or
  Crossref. Adding a library is a data edit, not a code change.
- Shipped as package data so citation lookup works from an installed package
  regardless of working directory. See
  [§5](#5-an-installed-package-with-no-syspath-anywhere).
- Coverage is finite by design, and gaps are **reported rather than hidden**. See
  [§11](#11-report-rather-than-drop-refuse-rather-than-guess) and the coverage
  list in [documentation-draft.md](documentation-draft.md#citation-coverage).
