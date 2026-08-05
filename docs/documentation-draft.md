# Provenance Agent - Documentation (draft)

> Draft. The intent is that most of this eventually replaces or expands
> `README.md`; it is kept separate for now so the README stays short while this
> is being edited. Everything in it describes the tool as it currently behaves.

This is the manual: what the agent does and how to use it. Two companion
documents cover the rest:

- [design-decisions.md](design-decisions.md) - why it is built this way,
  project-wide and per module.
- [paleopal-integration.md](paleopal-integration.md) - notes on folding it into
  PaleoPAL.

---

## Contents

1. [What the agent is](#1-what-the-agent-is)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Debugging](#4-debugging)
5. [Technical details](#5-technical-details)
6. [Known limitations](#6-known-limitations)

---

## 1. What the agent is

The provenance agent takes a Jupyter notebook and works out what should be
cited in it: the Python libraries it imports, and the paleoclimate datasets it
actually uses. It is a standalone tool in the PaleoPAL ecosystem and does not
call the PaleoPAL agents.

It answers two separate questions, and each has its own workflow:

- **Software.** Which libraries does this notebook import, and what is the
  citation for each? Answered by reading the notebook's source with `ast`. It
  works on a notebook that has never been run.
- **Data.** Which variables in this notebook hold datasets that were actually
  used, as opposed to loaded and abandoned? Answered by a static data-flow
  analysis over the notebook's cells, which traces each external source forward
  and reports the variable that carries it into an analysis.

### The one thing to understand before using it

**Neither workflow returns citations. Each one injects a cell into your
notebook, and the citations are that cell's output when you run it.**

This is not an implementation detail; it shapes everything else. The reason is
the data side. Retrieving a dataset citation means calling `get_bibtex()` on a
LiPD object or `get_publications()` on a PyleoTUPS object, and those objects
live in *your* kernel with your data already loaded. The agent cannot reach
into your kernel from outside, so instead it writes the retrieval code into
your notebook and hands it back to you to run. The software side follows the
same shape so the two contracts match.

So `cite_software()` returns the list of library names it built a cell for, and
`cite_data()` returns the `[variable, tool]` pairs it built a retrieval cell
for. To see citations, reload the notebook and run the new cell at the bottom.

### What a "dataset source" means here

Three kinds, and the detector recognizes each by the calls that create it:

| Source | Recognized by | Cited with |
|---|---|---|
| **PyLiPD** | `LiPD()` then `load()`, `load_from_dir()`, `load_remote_datasets()` | `get_bibtex(remote=True)` |
| **PyleoTUPS** | `PangaeaDataset()` / `NOAADataset()` then `search_studies()` | `get_publications()` |
| **LiPDGraph** | `requests.post()` to `linkedearth.graphdb.mint.isi.edu/repositories/...`, parsed into a DataFrame | dataset names lifted from the result frame, loaded into a fresh `LiPD` object, then `get_bibtex(remote=True)` |

`xarray` and `pandas` loaders are also tracked, so an analysis fed by
`xr.open_dataset(...)` or `pd.read_csv(...)` is reported too.

---

## 2. Installation

Requires **Python 3.10+**. The package is not on PyPI, so install it from a
clone:

```bash
git clone https://github.com/LinkedEarth/provenance-agent.git
cd provenance-agent
pip install -e ".[dev,google]"
```

That one command installs the package, its runtime dependencies, `pytest`, and
one LLM provider. Nothing needs to go on `sys.path`.

**Swap `google` for the provider you actually want** - `openai`, `anthropic`,
`ollama`, or `xai`. No provider is installed by default, so nobody carries an integration
they will never load; see [Choosing an LLM
provider](#choosing-an-llm-provider). If you only plan to use the direct Python
functions, which need no model at all, `pip install -e ".[dev]"` is enough.

### Which environment to install into

**Install it into the same environment as the Jupyter kernel you will analyze
notebooks from.** The cell the software workflow injects contains
`from provenance_agent.citations import collect_library_entries`, so that kernel
has to be able to import the package. If your notebooks run in a conda
environment, activate that environment first and install into it; if they run in
a virtualenv, install into that.

The agent depends on `pylipd` and `pyleotups` because the retrieval cells it
generates import them. It deliberately does **not** depend on the analysis
libraries your notebook uses - `pyleoclim`, `xarray`, `eofs`, `cfr`, and the
rest belong to the notebook's own environment. Installing the agent into the
kernel you already use for that science is therefore the whole setup.

### Before installing into an existing scientific environment

pip can silently replace conda-managed scientific packages with PyPI builds.
If you are installing into an environment you care about, check first:

```bash
pip install --dry-run -e ".[dev,google]"
```

The output should end with `Would install provenance-agent-0.1.0` and nothing
else. If it proposes changing `numpy`, `pandas`, `pylipd`, `pyleotups`, or
`pyleoclim`, install with `--no-deps` instead and then run `pip check`.

### Credentials

Only the natural-language layers (`%provenance` and `agent.run`) call a model.
The direct Python functions, `cite_software()` and `cite_data()`, use no model
at all and need no key of any kind. If you never want to configure an API key,
[skip to the direct functions](#from-python-the-direct-functions) - they are the
majority of the tool.

Put the key in a `.env` file **at the root of the project you are working in**,
or export it. An exported environment variable wins over the file.

```bash
# .env
GOOGLE_API_KEY=...
```

> **The root location matters.** `provenance_agent.llm` looks in the working
> directory first and falls back to a search from the package's own directory.
> Inside a Jupyter kernel, python-dotenv treats the session as interactive and
> resolves *both* lookups from the kernel's working directory, so the fallback
> never runs there. A `.env` that only exists under `src/` is invisible to
> notebooks. Keep one at the repository root.

### Choosing an LLM provider

The model is only used to classify one short request into a typed decision, so
any competent chat model does the job, and you are not tied to Google. Two
environment variables choose one:

| Variable | Meaning | Default |
|---|---|---|
| `PROVENANCE_LLM_PROVIDER` | `google`, `openai`, `anthropic`, `ollama`, or `xai` | `google` |
| `PROVENANCE_LLM_MODEL` | the model identifier to pass to that provider | the provider's default, below |

| Provider | Install | Key variable | Default model |
|---|---|---|---|
| `google` | `pip install -e ".[google]"` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini-flash-latest` |
| `openai` | `pip install -e ".[openai]"` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `pip install -e ".[anthropic]"` | `ANTHROPIC_API_KEY` | `claude-sonnet-5` |
| `ollama` | `pip install -e ".[ollama]"` | none - runs locally | `llama3.1` |
| `xai` | `pip install -e ".[xai]"` | `XAI_API_KEY` | `grok-4` |

`grok` is accepted as an alias for `xai`, which is the name PaleoPAL uses.

Google is only the *default selection*, not a default installation. So an
OpenAI user writes:

```bash
pip install -e ".[dev,openai]"
```

```bash
# .env
PROVENANCE_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

and optionally `PROVENANCE_LLM_MODEL=gpt-4o` to override the default model. No
Google package is installed and no Google key is ever consulted.

**Every provider is an extra, including the default one.** The integrations are
imported lazily, so you install exactly the one you selected and nothing else.
That matters because `langchain-google-genai` pulls `google-genai` and
`google-auth`, which in turn pull `cryptography` and `aiohttp` - precisely the
kind of transitive install the section above warns about for conda
environments. The trade-off is that a core install has no provider at all, so
the first `%load_ext provenance` after forgetting to pick one fails with an
error naming the exact install command:

```
RuntimeError: LLM provider 'google' needs the langchain_google_genai package,
which is not installed. Install it with: pip install "provenance-agent[google]"
```

The API key is checked before the package is imported, so someone with neither
a key nor an integration is told about the key first. That ordering is
deliberate: the key is the requirement they would hit again immediately after
installing the package.

### If you already use PaleoPAL

This agent is standalone, but it is meant to sit comfortably alongside PaleoPAL
and is expected to be integrated into it eventually. Its LLM configuration is
therefore readable from a PaleoPAL setup without editing anything:

- **Keys carry over unchanged.** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GOOGLE_API_KEY`, and `XAI_API_KEY` are the same names PaleoPAL uses, so
  whichever you already have works here.
- **Vendor selection carries over.** If `PROVENANCE_LLM_PROVIDER` is unset, the
  agent reads PaleoPAL's `DEFAULT_LLM_PROVIDER`, including its `grok` spelling
  for xAI. Setting `PROVENANCE_LLM_PROVIDER` always overrides it, so you can run
  PaleoPAL on one vendor and this agent on another.
- **Model selection deliberately does not carry over.** PaleoPAL's
  `OPENAI_MODEL`, `CLAUDE_MODEL`, `GOOGLE_MODEL`, `GROK_MODEL`, and
  `OLLAMA_MODEL` are ignored. Its defaults are heavyweight reasoning models
  chosen for multi-agent work over paleoclimate literature, whereas this agent
  makes one short classification call that the cheap fast tier answers just as
  well. Use `PROVENANCE_LLM_MODEL` if you want a specific model here.

One thing name compatibility does not solve: PaleoPAL keeps its keys in
`backend/.env`, while this agent searches upward from the working directory. A
notebook outside `paleopal/backend/` will not find that file. Copy the keys into
a `.env` at the root of your notebook project.

If you are embedding this agent rather than configuring it, skip the
environment entirely and pass your own client to `run(..., model=...)`. That is
the intended path for eventual integration: PaleoPAL builds its chat clients
through `LLMProviderFactory` and wraps them in a `LangChainWrapper`, which is a
LangChain `Runnable` and can be handed straight to `run`. See [Supplying your
own model](#supplying-your-own-model).

To construct a client yourself rather than through the environment:

```python
from provenance_agent.llm import build_llm

model = build_llm(provider="anthropic", model="claude-opus-5")
```

Adding a fifth provider means one entry in `PROVIDERS` in
`src/provenance_agent/llm.py` - the integration package, the chat class, a
default model, and the environment variables it accepts as a key.

### Verifying the install

```bash
pytest tests/ -q
```

The suite is fully offline. No test calls a model, SPARQL, or a remote dataset
service, and it needs no API key. That holds on a fresh clone with no `.env`:
the chat client is built on first use rather than at import, so importing the
agent costs no credentials, and `tests/test_packaging.py` pins that property.

---

## 3. Usage

### From a notebook: the `%provenance` magic

This is the intended experience.

> **The extension name is `provenance`, not `provenance_agent`.** The import
> namespace and the magic name deliberately differ: you `import
> provenance_agent` in Python, but you `%load_ext provenance` in a notebook.
> `%load_ext provenance_agent` fails with
> `The provenance_agent module is not an IPython extension.`, and every
> `%provenance` line after it then fails with
> `UsageError: Line magic function '%provenance' not found.`
>
> `provenance` is a real top-level module of the installed package,
> `src/provenance.py`, kept as a thin shim precisely so this command stays
> stable.

```python
%load_ext provenance

%provenance cite the software
%provenance cite the datasets
%provenance cite Pyleoclim
%provenance cite everything
```

Each request appends one cell to the notebook **on disk** and prints a summary
of what it injected. Reload the notebook and run the new cell.

If notebook auto-detection fails, which it usually does in VSCode, set the path
once per session:

```python
%provenance_notebook path/to/notebook.ipynb
```

Auto-detection uses `ipynbname`, which matches the running kernel against the
Jupyter server's session list. VSCode does not expose that list the same way,
so detection fails there and the override is the normal path, not a fallback.

### From Python: the direct functions

```python
from provenance_agent import cite_data, cite_software

cite_software("notebook.ipynb")
cite_software("notebook.ipynb", libraries="pyleoclim")
cite_software("notebook.ipynb", libraries=["pyleoclim", "pandas"],
              citation_types=["software"])

cite_data("notebook.ipynb")
cite_data("notebook.ipynb", targets="Ocn-RedSea.Felis.2000")
```

Both take an `output_path` to write to a copy instead of modifying the notebook
in place. Both are static and offline: they read the notebook, decide what to
cite, and write one cell back.

| Argument | Applies to | Meaning |
|---|---|---|
| `libraries` | `cite_software` | a name or list; names the notebook does not import are dropped |
| `citation_types` | `cite_software` | `"paper"` and/or `"software"`; omit for both |
| `targets` | `cite_data` | exact `dataSetName` values or study IDs. **Never notebook variable names.** |
| `output_path` | both | write here instead of in place |
| `fmt` | `cite_data` | accepted and ignored, see [Known limitations](#6-known-limitations) |

### From Python: the LangChain tools and the router

```python
from provenance_agent.data import cite_data_tool
from provenance_agent.software import cite_software_tool
from provenance_agent.agent import run

result = run("cite the software", "notebook.ipynb")
for call in result["dispatch"]:
    print(call["name"], "->", call["result"])
```

`run` returns an envelope, not a list:

```python
{
  "status": "ok" | "warning",
  "decision": {...},          # the classifier's typed decision
  "dispatch": [ {"name": ..., "args": ..., "result": ...}, ... ],
  "verification": {"cells": [...], "present": [...], "mutated": bool,
                   "runtime_unverified": bool},
  "warning": "..."            # only when status is "warning"
}
```

`run` is imported from `provenance_agent.agent`, not from the package root.
That keeps the routing layer, its LangChain dependencies, and its provider
configuration off the import path of callers who only want the two direct
functions.

#### Supplying your own model

`run` takes an optional third argument, `model`, which is any LangChain
`Runnable` that returns a message. Passing one classifies with that model
instead of the one `llm.py` configured:

```python
from provenance_agent.agent import run

result = run("cite the software", "notebook.ipynb", model=my_chat_client)
```

Omitting it behaves exactly as before. This exists so an application embedding
this agent can hand over the chat client it already built, rather than
configuring a second one through `PROVENANCE_LLM_PROVIDER`. The provider
registry then serves standalone use and gets out of the way when it is not
wanted.

The injected model is used only for classification; nothing else in the
pipeline calls a model, so the rest of the run is unaffected.

Injecting also costs nothing when you do not use the configured provider. The
default client is built on first use rather than at import, so a caller that
always passes `model=` never constructs one, and never needs
`PROVENANCE_LLM_PROVIDER` or a key at all.

### What the injected cells look like

Software, one cell for all libraries:

```python
# provenance-agent-generated
from provenance_agent.citations import collect_library_entries
provenance_software = collect_library_entries(['numpy', 'pandas', 'pyleoclim'], None)
display(provenance_software)
```

Data, one cell covering every detected source:

```python
# provenance-agent-generated
from pylipd.lipd import LiPD
_names_filtered_df2 = filtered_df2["dataSetName"].unique().tolist()
_lipd_filtered_df2 = LiPD()
_lipd_filtered_df2.set_endpoint("https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic")
_lipd_filtered_df2.load_remote_datasets(_names_filtered_df2)
_bib_filtered_df2, _meta_filtered_df2 = _lipd_filtered_df2.get_bibtex(remote=True)
display(_meta_filtered_df2)
```

After running it, `_bib_filtered_df2` holds the BibTeX and
`_meta_filtered_df2` holds the metadata frame, both still bound in your kernel.

Each workflow owns exactly one cell. Running a workflow again replaces its own
cell rather than stacking a stale second one, so re-running is safe.

### The software citation DataFrame

`provenance_software` has one row per citation entry:

| Column | Contents |
|---|---|
| `library` | lowercased library name |
| `citation_type` | `"paper"` or `"software"` |
| `key` | BibTeX key |
| `title`, `author`, `year`, `doi` | parsed BibTeX fields |
| `bibtex` | the full BibTeX entry as text |
| `note` | `"No citation found for imported library"` when nothing matched, otherwise empty |

A library with both a paper and a software citation produces two rows. Rows are
deduplicated by DOI.

---

## 4. Debugging

### Start here: the module CLIs

Each analysis module runs standalone against a notebook. This is the fastest
way to see what the agent thinks, with no LLM and no notebook mutation.

```bash
# which libraries does it see?
python -m provenance_agent.notebook_io notebooks/demos/paleoPCAlite.ipynb
# Libraries: ['matplotlib', 'numpy', 'pandas', 'pyleoclim', ...]

# which datasets does it detect?
python -m provenance_agent.dataset_detection notebooks/demos/paleoPCAlite.ipynb
# filtered_df2    LiPDGraph

# same, straight from the analyzer
python -m provenance_agent.deterministic_dataset_detection notebooks/examples/02a-query_lipd_graph.ipynb
```

### "It found no datasets" - ask the detector why

`detect_datasets_with_diagnostics` returns the same pairs plus an explanation
for every analysis call whose data it could not trace back to a source:

```python
from provenance_agent.dataset_detection import detect_datasets_with_diagnostics

d = detect_datasets_with_diagnostics("notebook.ipynb")
print(d["pairs"])
for w in d["warnings"]:
    print(w)
```

For a notebook that loads data through something the detector does not know:

```
Analysis operation 'pca' at cell 1, line 3 has no recognized dataset source
lineage; the data may come from an unsupported loader or local computation.
```

`detect_datasets` emits the same messages as `UserWarning`, so they surface in
a notebook without asking for them.

**An empty result with no warnings is different from an empty result with
warnings.** No warnings means the detector found no analysis calls at all -
the notebook loads data and stops. Warnings mean it found analysis but could
not connect it to a recognized source.

### Symptom table

| Symptom | Likely cause | What to do |
|---|---|---|
| `The provenance_agent module is not an IPython extension.` | `%load_ext provenance_agent` | the extension is `%load_ext provenance`. `provenance_agent` is the *import* name, not the magic |
| `UsageError: Line magic function '%provenance' not found.` | the extension never loaded, usually the row above | fix the `%load_ext` line and re-run it before any `%provenance` line |
| A `%provenance` request raises `RuntimeError: No API key found for LLM provider ...` | no key for the selected provider. The client is built on first use, so this surfaces at the first request rather than at `%load_ext` | set the variable the message names, in a `.env` at the project root or exported. Use the direct functions if you do not need routing |
| `RuntimeError: LLM provider 'openai' needs the langchain_openai package` | `PROVENANCE_LLM_PROVIDER` selected a provider whose integration is not installed | run the install command in the message, e.g. `pip install -e ".[openai]"` |
| `ValueError: Unknown LLM provider ...` | a typo in `PROVENANCE_LLM_PROVIDER` | the message lists the registered names; see [Choosing an LLM provider](#choosing-an-llm-provider) |
| `UsageError: Could not auto-detect the current notebook path` | `ipynbname` cannot match the kernel, normal in VSCode | `%provenance_notebook path/to/notebook.ipynb` |
| The result is missing a cell you just wrote | every workflow reads the `.ipynb` **from disk** | save the notebook and re-run |
| `ModuleNotFoundError: provenance_agent` in an injected cell | the kernel's environment does not have the package | `pip install -e ".[dev]"` into *that* environment, then restart the kernel |
| `NameError: filtered_df2` when running the injected data cell | the untargeted retrieval path reuses your already-loaded objects | run the notebook's own cells first, then the injected cell |
| A library shows a `note` row instead of a citation | no entry in the packaged citation index | see [Known limitations](#6-known-limitations) |
| `cite_data` returned `[]` and changed nothing, with a warning about PyleoTUPS | a specific PyleoTUPS study was requested by name or ID | ask for all datasets instead |
| `ValueError: source-variable targeting is unsupported` | `generate_data_workflow` was called with the retired `variable=` argument | pass dataset names through `targets=` instead. `cite_data` does not expose `variable` at all |
| A specific `targets=` name silently matched nothing | `targets` values are `dataSetName`s. A notebook variable name is not rejected, it just never matches | use the name as it appears in the data, not the variable holding it |
| Two provenance cells at the bottom | one software, one data. That is correct | each workflow owns one cell |
| An older notebook has a `provenance_bibliography` combine cell | written by a version that no longer exists | any workflow run strips it automatically |

### Checking what got injected without running anything

```python
import nbformat
from provenance_agent.notebook_io import is_generated_cell

nb = nbformat.read("notebook.ipynb", as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and is_generated_cell(cell.source):
        print(f"--- cell {i} ---\n{cell.source}")
```

`is_generated_cell` also recognizes the signatures older versions wrote, so it
finds stale cells too.

### Inspecting the router without spending a call

`build_chain` accepts a fake model, which is how the tests exercise routing
offline:

```python
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from provenance_agent.agent import build_chain

decision = '{"action":"cite","scope":"all","kinds":["software"],"targets":[],"fmt":"bibtex"}'
chain = build_chain(RunnableLambda(lambda _: AIMessage(decision)))
print(chain.invoke({"request": "cite the software", "notebook_path": "notebook.ipynb"}))
```

The chain's stages are named, so `chain.get_graph()` shows
`prepare_context | classify | resolve_targets | dispatch | verify`.

### Working safely

Every workflow modifies the notebook **in place** unless given `output_path`.
When experimenting, copy first:

```python
import shutil
shutil.copy("notebook.ipynb", "notebook_demo.ipynb")
```

`notebooks/demos/workflow.ipynb` does this already.

---

## 5. Technical details

```text
provenance-agent/
├── pyproject.toml                  setuptools src-layout config; deps, the
│                                   `dev` extra, one extra per optional LLM
│                                   provider, and Citations/ as package data
├── src/
│   ├── provenance.py               the top-level module `%load_ext provenance`
│   │                               resolves. A forwarding shim over
│   │                               provenance_agent.magic with no logic of its
│   │                               own; the one intentional top-level module
│   └── provenance_agent/
│       ├── __init__.py             exports cite_data and cite_software, and
│       │                           nothing else. The tools and `run` stay on
│       │                           their own modules so importing the package
│       │                           root never constructs the LLM client
│       ├── notebook_io.py          reads .ipynb files. Strips IPython magics
│       │                           and shell lines so cells parse, walks the
│       │                           AST for imports, recovers imports line by
│       │                           line from cells with syntax errors, and
│       │                           owns is_generated_cell()
│       ├── citations.py            software citation lookup. Reads the
│       │                           packaged Citations/ index and .bib files
│       │                           through importlib.resources and merges them
│       │                           into one DataFrame deduped by DOI. Also
│       │                           holds the generated-cell removal helpers
│       │                           both workflows share
│       ├── software.py             the whole software side: the injected
│       │                           cell's source, injection, the workflow,
│       │                           cite_software, and cite_software_tool
│       ├── data.py                 the whole data side: per-source retrieval
│       │                           blocks, the single injected cell, target
│       │                           handling, the workflow, cite_data, and
│       │                           cite_data_tool. Lifts the LiPDGraph
│       │                           endpoint out of the notebook by AST so a
│       │                           notebook pointed at a different repository
│       │                           is handled correctly
│       ├── dataset_detection.py    the detection facade. detect_datasets()
│       │                           and detect_datasets_with_diagnostics()
│       │                           delegate to the analyzer below. Also holds
│       │                           the DEPRECATED LLM detector - prompt,
│       │                           builder, response parser - kept inactive as
│       │                           a documented rollback path. Not dead code
│       ├── deterministic_dataset_detection.py
│       │                           the analyzer, and by far the largest module
│       │                           (~1600 lines). Builds a versioned data-flow
│       │                           graph over the notebook's cells: every
│       │                           assignment records its dependencies, source
│       │                           groups, and object family; recognizers
│       │                           attach sources to LiPD/PyleoTUPS/LiPDGraph/
│       │                           xarray/pandas loaders; analysis calls are
│       │                           sinks. Results come from walking each sink's
│       │                           dependency closure to the nearest source
│       │                           boundary, plus a per-source fallback to live
│       │                           terminal tables. Never executes anything
│       ├── agent.py                the LCEL router. prepare_context | classify
│       │                           | resolve_targets | dispatch | verify, each
│       │                           a named Runnable. Classification is a
│       │                           Pydantic-validated JSON decision; verify
│       │                           diffs the notebook before and after without
│       │                           running the injected cells
│       ├── magic.py                the IPython extension implementation.
│       │                           Resolves the notebook path, calls
│       │                           agent.run(), renders the envelope. No
│       │                           citation or routing logic
│       ├── llm.py                  the PROVIDERS registry, build_llm(), the
│       │                           single shared chat client, and the helper
│       │                           that normalizes a response to text. Owns
│       │                           provider selection and credential discovery.
│       │                           Integrations are imported lazily, so only
│       │                           the selected provider must be installed, and
│       │                           the client itself is built on first access
│       │                           via a module __getattr__, so importing
│       │                           costs no credentials
│       └── Citations/              packaged citation data: library_citations.yml
│                                   (the index) plus one .bib per library
├── notebooks/
│   ├── demos/                      workflow.ipynb, the single demo/dev
│   │                               notebook: software building blocks, data
│   │                               building blocks, then the agent layer.
│   │                               paleoPCAlite.ipynb sits beside it as the
│   │                               notebook it runs against, edited in place
│   ├── examples/                   worked science notebooks. Also the
│   │                               detection corpus the test suite pins
│   └── instructions/               NotebookN bundles, each self-contained with
│                                   its own .lpd sibling
├── benchmark/ground_truth/         expected software and dataset entries per
│                                   notebook, as YAML. Data only; the scoring
│                                   runner was removed
└── tests/                          pytest, fully offline
```

### How the two workflows differ

|  | Software | Data |
|---|---|---|
| How targets are found | `ast` import scan | data-flow analysis |
| Needs the notebook to have run | no | no, to *detect* |
| Needs a live kernel | no | yes, to *retrieve* |
| Injected cells | 1 | 1, covering every source |
| Binds | `provenance_software` | `_bib_{var}`, `_meta_{var}` per source |
| Where citations come from | packaged `Citations/` | `get_bibtex()` / `get_publications()` at run time |

### Test modules

| Module | Covers |
|---|---|
| `test_notebook_io.py`, `test_citations.py` | parsing and citation lookup |
| `test_llm.py` | response normalization and the provider registry: name resolution, both environment variables, and the missing-package / missing-key messages |
| `test_software.py`, `test_data.py` | each workflow's cell building and injection |
| `test_dataset_detection.py` | the facade, including the deprecated LLM helpers |
| `test_deterministic_dataset_detection.py` | the analyzer, including the tracked corpus |
| `test_agent.py`, `test_magic.py`, `test_provenance_shim.py` | routing, rendering, and the `%load_ext` shim |
| `test_public_api.py` | the four canonical import paths |
| `test_packaging.py` | properties only an install shows: imports and citation lookup from outside the checkout, credential discovery, that importing the agent and loading the magic need no credentials at all, and a built wheel with no `.env` in it |
| `test_import_hygiene.py` | an AST scan that fails on any `sys.path` mutation or retired flat import, in Python **or** in a notebook code cell |
| `test_notebooks.py` | every notebook validates, round-trips, and resolves its local `.lpd`/`.bib` paths |

---

## 6. Known limitations

### Output format

- **Citations are a DataFrame, not styled text.** There is no APA, MLA, or
  Chicago rendering. APA rendering existed and was removed along with the LLM
  chain that produced it. A future APA implementation would render from the
  metadata DataFrame without a model.
- **`fmt` is accepted and ignored.** `cite_data(..., fmt="apa")`,
  `fmt="bibtex"`, and `fmt="anything"` all produce byte-identical output. No
  value is rejected either. The parameter is held open for that future APA
  path, and it is carried through the agent's routing decision for the same
  reason.
- **No `.bib` file is written.** BibTeX is available, but as data rather than a
  file: the `bibtex` column of the software DataFrame, and the `_bib_{var}`
  bindings the data cell leaves in the kernel. Writing a `.bib` is up to you.

### Citation coverage

- **The index covers 15 libraries.** `ammonyte`, `cartopy`, `cfr`, `cftime`,
  `eofs`, `matplotlib`, `numpy`, `pandas`, `pyleoclim`, `pyleotups`, `pylipd`,
  `requests`, `scipy`, `seaborn`, `xarray`.
- **Anything else gets a "not found" row, not a citation.** An imported library
  with no entry produces a row whose `note` reads
  `"No citation found for imported library"` and whose citation fields are all
  null. It is reported rather than dropped, on purpose, so you can see what is
  missing. Among the paleoclimate libraries the agent is otherwise built for,
  **`pens` and `climlab` have no entry**.
- **Three libraries have a paper but no software citation.** `eofs`, `scipy`,
  and `seaborn` have an inline paper entry and no `.bib` file, so asking for
  `citation_types=["software"]` gives them a "not found" row.
- Adding a library means editing `src/provenance_agent/Citations/` - an entry
  in `library_citations.yml` and/or a `<library>.bib` beside it. There is no
  automatic lookup against GitHub, Zenodo, or Crossref.
- Standard-library imports are dropped entirely rather than reported as
  missing.

### What dataset detection will and will not see

- **A source that is loaded but never used is not reported.** Detection
  requires each source's lineage to reach a recognized analysis call, or to
  produce a live terminal tabular DataFrame. A notebook that loads a LiPD file
  and stops reports nothing. This is the intended behavior - it is what makes
  the tool cite what was *used* - but it surprises people.
- **The analysis vocabulary is a fixed registry.** `ANALYSIS_METHODS` in
  `deterministic_dataset_detection.py` lists what counts: `pca`, `corr`,
  `regress`, `wavelet`, `ssa`, `fit`, `spectral`, and about 40 more. A domain
  method outside that list is not an analysis boundary. Extending it is a code
  change.
- **xarray sources need a real analysis boundary.** The terminal-table fallback
  is deliberately tabular, so an xarray source that only ever produces arrays
  is not reported unless it reaches an analysis call.
- **Lineage does not cross user-defined helper functions.** If your loading is
  wrapped in your own function, or the import is dynamic, the detector will not
  guess. It records a diagnostic warning instead of inventing a source.
- **Cells with syntax errors are skipped whole by the detector.** The *import*
  scanner recovers imports line by line from a broken cell, but the data-flow
  analyzer cannot, so anything in that cell is invisible to detection.
- Cells the tool previously injected are skipped by both scanners, so a second
  run does not cite the tool's own machinery.

### Targeting

- **A specific PyleoTUPS study cannot be requested.** Asking for one by name or
  ID emits a warning, returns `[]`, and leaves the notebook completely
  unchanged. The available studies only exist inside the in-memory provider
  object, so there is nothing to match against before the notebook runs. Ask
  for all datasets instead. PyLiPD and LiPDGraph targets work.
- **Targets are dataset names, never notebook variable names.** Variable names
  are internal detector output. Passing one raises `ValueError`.
- **Requested dataset names are never checked statically.** Whenever a request
  names specific datasets, `agent.run`'s envelope sets
  `verification.runtime_unverified` and the magic adds "Some requested dataset
  names will be verified when the cells run." A name that does not exist is not
  an error until the injected cell runs. A name given via `targets=` that
  matches nothing simply produces an empty retrieval.

### Workflow and I/O

- **Notebooks are read from disk.** Unsaved cells are invisible.
- **Writes are in place** unless `output_path` is given. The `%provenance`
  magic and `agent.run` always write in place.
- **Dataset retrieval needs your kernel.** Untargeted retrieval reuses the
  objects already loaded in the notebook, so the injected cell fails with a
  `NameError` if the notebook's own cells have not run.
- The notebook must be reloaded after injection before the new cell is visible.

### The agent layer

- **`%provenance` and `agent.run` need an API key, but importing does not.**
  The client is built on first model use, so `%load_ext provenance`, `import
  provenance_agent.agent`, and the whole test suite all work with no
  credentials configured. The missing key is reported when you actually ask the
  agent to route something. Which key depends on `PROVENANCE_LLM_PROVIDER`; the
  default is Google. The direct functions never need one.
- **That means a missing key is reported late rather than early.** You will not
  find out at `%load_ext` time that your `.env` is wrong; you find out on your
  first `%provenance` request. This is the deliberate tradeoff for letting a
  fresh clone run the test suite and for letting an embedding application
  supply its own model without configuring this one.
- **Substitute a client by assigning the cache, not by patching the name.**
  `provenance_agent.llm._CLIENT` and `provenance_agent.agent._CHAIN` are the
  supported injection points. Reading `llm.llm` or `agent.chain` is what
  triggers construction, so patching those names defeats the laziness and,
  without credentials, raises instead.
- **Only one model call is ever made, and only by the agent layer.** It
  classifies the request. Nothing else in the tool - detection, citation lookup,
  cell generation - consults a model, which is why the provider choice barely
  matters and why a local Ollama model is a reasonable option.
- **Provider support is a fixed registry.** `PROVIDERS` in `llm.py` covers
  Google, OpenAI, Anthropic, Ollama, and xAI - the same set PaleoPAL supports.
  Anything else (Bedrock, Azure OpenAI, Vertex, Mistral) needs a registry entry,
  which is a four-line code change but still a code change. There is no generic
  `provider:model` string parsing.
- **PaleoPAL compatibility is a convenience, not a contract.** Reading
  `DEFAULT_LLM_PROVIDER` is a courtesy to users who already configured PaleoPAL.
  Nothing verifies that PaleoPAL still uses that name, and if it renames the
  variable this agent silently falls back to its own default rather than
  failing. Set `PROVENANCE_LLM_PROVIDER` explicitly if you need certainty.
- **Alternative credential schemes are not recognized.** The key check looks for
  the environment variables in the registry, so Google application-default
  credentials, an AWS profile, or an Azure deployment configuration will fail
  the check even where the underlying client could have authenticated. Call
  `build_llm(...)` directly if you need that.
- **Routing is a model call, so it is not deterministic.** The classifier runs
  at `temperature=0`, but an unusual phrasing can still route differently or
  come back as a warning. Weaker models route worse; the failure mode is a
  warning rather than a wrong citation, because an unclear decision does not
  mutate the notebook. The direct functions are the deterministic path.
- **Notebook auto-detection fails in VSCode.** It works under JupyterLab and
  the classic notebook server, where `ipynbname` can match the kernel against
  the server's session list. In VSCode `%provenance_notebook` is the normal way
  to work, not a fallback.

### Scope and packaging

- **Not published.** Local editable install only; there is no PyPI release.
- **Analysis libraries are not dependencies.** `pyleoclim`, `xarray`, `eofs`,
  `cfr`, and similar belong to the analyzed notebook's own environment. Only
  `pylipd` and `pyleotups` are package dependencies, because the generated
  retrieval cells import them.
- **No LLM integration is a core dependency.** All five providers are extras,
  including the default one, so a core install has no model client and the
  agent layer raises until you pick a provider. This is deliberate: the
  alternative was shipping `langchain-google-genai` to every user regardless of
  which provider they use. The `langchain` umbrella package is also not a
  dependency, which is why provider selection is a small registry here rather
  than a call to langchain's `init_chat_model`.
- **No SPARQL-direct pathway.** LiPDGraph always converts to a LiPD object
  first. A hardcoded bibliography query exists only as a fallback if that
  conversion ever proves too slow.
- **The LLM detector is retained but never called.** `DETECTION_PROMPT`,
  `build_detection_prompt`, and `parse_detection_response` in
  `dataset_detection.py` are a deliberate rollback path with tests kept green.
  They are not dead code and should not be deleted.

### Ground truth

`benchmark/ground_truth/` holds the expected software and dataset entries for
each corpus notebook, as manually reasoned YAML. The repeatable evaluator reads
those records and compares them with the current software-import and
deterministic dataset detectors:

```text
/opt/anaconda3/envs/lang/bin/python benchmark/run_ground_truth.py
```

The command evaluates every record, prints per-notebook missing and unexpected
values with precision/recall/F1, and writes the detailed report to
`benchmark/results/ground_truth_results.json`. To evaluate one notebook, filter
by a substring of its repository-relative path:

```text
/opt/anaconda3/envs/lang/bin/python benchmark/run_ground_truth.py \
  --notebook Notebook1
```

Use `--ground-truth PATH` or `--output PATH` to override the input directory or
JSON report location. The runner does not execute notebooks, alter the YAML
labels, inject citation cells, call an LLM, or create `.bib` files.
