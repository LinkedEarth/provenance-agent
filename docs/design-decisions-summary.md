# High-level Design Decisions

## Project-wide decisions

**Citations are produced by notebook cells.** Dataset citation information comes
from data objects that are already loaded in the notebook's live Python kernel.
The tool runs outside that kernel, so it cannot use those objects directly. It
adds code to the notebook instead, and you run that code. Software citations
could be returned directly, but using the same approach for software and data
makes the project easier to use and cohesive.

**Software and data have separate workflows.** The software workflow looks at
the libraries imported by the notebook and uses citation files included with
this project. The data workflow follows how datasets move through the notebook
and uses the data objects that are already loaded. Each workflow creates its own
cell, so you can run or remove them separately.

**Dataset detection uses fixed rules.** The project looks for known data loaders
and known analysis calls. This makes the result repeatable, free, and usable
without an API key. The tradeoff is that it cannot understand every custom
loader or helper function. When it cannot follow a dataset, it reports the
problem instead of guessing. Theoretically, the retired LLM detection would not necessarily work as a fallback, because the deterministic detection would always return something, and would have no way to flag if the result is incorrect and therefore fallback. The best way to improve it would be to add edge cases and support xarray/pandas data loading.

**The agent layer is technically optional.** `agent.run()` lets someone make a request in
plain English, such as asking for dataset citations. The direct
`cite_software()` and `cite_data()` also work. They do not need a model or an API key.

**Only one model call is used.** A API call to an LLM only used for the natural-language request. The tools: normal Python code handles dataset detection, citation
lookup, notebook changes, and verification. This keeps the project cheaper and
easier to troubleshoot compared to a probabilistic LLM.

**The project is installed as a normal Python package.** The code lives under
`src/`, and notebooks import the installed package as `provenance_agent`. The
project does not change `sys.path` by hand. This means generated notebook cells
use the same kind of import as any other installed Python package.

**The model client is created only when needed.** Importing the package, using
the direct functions, loading the notebook extension, and running offline tests
should work without credentials. A model client is created only when the agent
actually needs to send a request to a model. So lazy LLM loading.

**The project works by itself and can also be integrated later.** PaleoPAL is
not required. At the same time, the project uses compatible credential names
and provider settings so a larger application can use it later. Citation
retrieval stays in this project instead of depending on PaleoPAL agents, as writing code only requires following a specific format for this project.

**Each workflow owns its code.** `software.py` contains the software workflow,
and `data.py` contains the data workflow. Keeping each workflow modular makes testing and expansion easier.

**Some older interfaces are kept for compatibility and future development.** The `fmt` argument, markers from older generated cells, the older language-model detector, and
older filtering behavior remain so existing callers and notebooks do not break.
Newer implementations are used by default.

**The agent shows missing information instead of hiding it.** If a citation is
not available, the project reports that fact. If a request is unclear or not
supported, it warns the user and leaves the notebook unchanged. A visible
missing result is safer than an incorrect citation.

**Tests run without outside services.** The test suite should work on a fresh
checkout without an API key, model call, SPARQL request, or remote dataset
service. Fake model objects and local test data are used instead.

## Why these files exist

The project is split by responsibility and by the environment in which each
piece runs. The notebook-facing code (`src/provenance.py`, `magic.py`, and
`notebook_io.py`) handles IPython and notebook mechanics, while the package
root exposes only the two direct workflows. `software.py` and `data.py` are
separate because they answer different citation questions: software citations
come from imports and curated metadata, while data citations depend on tracing
dataset variables and running generated code in the notebook's live kernel.

`dataset_detection.py` keeps the public detection contract and diagnostics
separate from `deterministic_dataset_detection.py`, where the fixed analysis
rules can evolve and be tested independently. The agent layer is separate from
`llm.py` for the same reason: `agent.py` orchestrates plain-English requests,
target selection, workflow execution, and verification, while `llm.py` only
handles provider settings, credentials, lazy client creation, and response
normalization. This lets direct software and data workflows remain usable
without an API key or model dependency, and lets the provider implementation
change without entangling the routing logic.
The remaining directories support those boundaries. `Citations/` is packaged
data rather than executable code, so the curated index and BibTeX files ship
with the install and can be read without a network connection or a particular
working directory; `pyproject.toml` declares that packaging boundary and the
optional dependencies, while `.env.example` documents credentials without
shipping secrets. `notebooks/` contains the examples and manual-development
inputs, `benchmark/ground_truth/` contains expected evaluation records rather
than runtime data, and `tests/` checks the public boundaries with offline test
inputs. Keeping these concerns separate makes the common paths smaller, keeps
side effects visible, and allows a notebook workflow, the optional agent, or
the citation index to be tested and changed without requiring the rest of the
system.

## Main tradeoffs

**Detection trades coverage for precision and predictability.** The detector
recognizes a fixed set of loaders, transformations, and analysis calls instead
of executing arbitrary notebook code or guessing about hidden behavior. This
makes results repeatable, usable offline, and less likely to attach a citation
to the wrong dataset. The cost is false negatives: custom loaders, dynamic
imports, hidden helper functions, cells with syntax errors, and analysis methods
outside the known list may produce a warning or no result. A missing result
means that the lineage was not recognized, not that the notebook used no data.

**The agent trades a small amount of cost and nondeterminism for flexible natural-language requests.** An LLM fits this layer because users can ask for
all software citations, a particular dataset, or a combination without
following a fixed command grammar. A conventional NLP pipeline based on
keywords, rules, or a trained intent classifier could handle a narrow
vocabulary, but it would become brittle as phrasing, filters, and workflows
expand and would require maintaining a grammar or labeled examples. The LLM is
limited to interpreting the request and selecting the workflow; deterministic
Python performs dataset detection, citation lookup, notebook edits, and
verification, so the model does not invent the results. If the supported
requests become a stable command language, replacing this layer with a
conventional parser would be a reasonable simplification.

**Citation coverage trades automatic discovery for stable, offline results.**
The project ships a curated library-to-citation index instead of querying
GitHub, Zenodo, or Crossref while it runs. This avoids network failures, rate
limits, changing metadata, and ambiguous search results. The cost is that the
index can be incomplete or need manual updates: if a library has no entry, the
project reports it as missing even if a citation exists elsewhere.

**Dataset citation trades full automation for fidelity to the notebook's live state.** The tool can inspect the saved notebook and generate code, but it
cannot directly access the Python objects in a running Jupyter kernel. The
generated cell therefore reuses the objects the notebook actually loaded,
which preserves the notebook's current data and filters and avoids executing
the notebook in a separate context. The cost is a manual state-dependent step:
save the notebook, run the data-loading cells, and then run the generated
citation cell in that same kernel. A restarted kernel or unsaved change can
leave the citation cell without the expected objects or with stale inputs.

**The output trades ready-to-publish formatting for structured, inspectable metadata.** The workflows show DataFrames and BibTeX fields rather than
turning the result into APA or MLA prose or automatically writing a `.bib`
file. This keeps the output precise and reusable by other tools, but users must
format or export it themselves for a manuscript or bibliography file.

**The editable checkout trades turnkey installation for development and environment control.** Installing the project with `pip install -e` makes the
local source importable and lets source changes take effect without rebuilding
the package. It also leaves scientific dependencies in the notebook's own
environment: libraries such as `pyleoclim` and `xarray` are not installed
automatically. This avoids forcing a large or conflicting scientific stack on
every user, but it means users must prepare a compatible environment and keep
the notebook's libraries separate from the provenance package itself.
