# High-level Design Decisions

## Project-wide decisions

**Citations are produced by notebook cells.** Dataset citation information comes
from data objects that are already loaded in the notebook's live Python kernel.
The tool runs outside that kernel, so it cannot use those objects directly. It
adds code to the notebook instead, and you run that code. Software citations
could be returned directly, but using the same approach for software and data
makes the project easier to use.

**Software and data have separate workflows.** The software workflow looks at
the libraries imported by the notebook and uses citation files included with
this project. The data workflow follows how datasets move through the notebook
and uses the data objects that are already loaded. Each workflow creates its own
cell, so you can run or remove them separately.

**Dataset detection uses fixed rules.** The project looks for known data loaders
and known analysis calls. This makes the result repeatable, free, and usable
without an API key. The tradeoff is that it cannot understand every custom
loader or helper function. When it cannot follow a dataset, it reports the
problem instead of guessing.

**The agent layer is optional.** `agent.run()` lets someone make a request in
plain English, such as asking for dataset citations. The direct
`cite_software()` and `cite_data()` functions are the main dependable way to use
the project. They do not need a model or an API key.

**Only one model call is used.** The model only decides which workflow matches a
plain-English request. Normal Python code handles dataset detection, citation
lookup, notebook changes, and verification. This keeps the project cheaper and
easier to troubleshoot.

**The project is installed as a normal Python package.** The code lives under
`src/`, and notebooks import the installed package as `provenance_agent`. The
project does not change `sys.path` by hand. This means generated notebook cells
use the same kind of import as any other installed Python package.

**The model client is created only when needed.** Importing the package, using
the direct functions, loading the notebook extension, and running offline tests
should work without credentials. A model client is created only when the agent
actually needs to send a request to a model.

**Model providers are optional.** Google, OpenAI, Anthropic, Ollama, and xAI are
separate optional installations. Users only need to install the provider they
plan to use. This avoids extra packages and lowers the chance that pip will
replace scientific packages managed by Conda.

**The project works by itself and can also be integrated later.** PaleoPAL is
not required. At the same time, the project uses compatible credential names
and provider settings so a larger application can use it later. Citation
retrieval stays in this project instead of depending on PaleoPAL agents.

**Each workflow owns its code.** `software.py` contains the software workflow,
and `data.py` contains the data workflow. Keeping each workflow in one place
makes it easier to understand and change.

**Some older interfaces are kept for compatibility.** The `fmt` argument,
markers from older generated cells, the older language-model detector, and
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

**`pyproject.toml`** Defines the installable package, its main dependencies,
development tools, optional model-provider installations, and packaged citation
data.

**`.env.example`** Shows the names of supported credential variables without
including a real secret. Real `.env` files stay on the user's computer.

**`src/provenance.py`** Keeps `%load_ext provenance` working while the main code
lives inside the `provenance_agent` package.

**`src/provenance_agent/__init__.py`** Exposes the two direct public functions.
Keeping the package root small means importing it stays fast and does not need
credentials.

**`notebook_io.py`** Reads notebooks, cleans cells enough to analyze them,
finds imports, and recognizes cells created by this project. It contains shared
notebook code, not dataset-specific logic.

**`citations.py`** Reads the software citation index and BibTeX files included
with the project. It builds the citation table and removes old generated cells
before a new cell is added.

**`software.py`** Finds imported libraries, builds one software-citation cell,
and writes that cell into the notebook.

**`data.py`** Builds one data-citation cell. When that cell runs, it uses the
dataset objects that are already in the notebook's kernel.

**`dataset_detection.py`** Provides the public dataset-detection functions and
diagnostics. It also keeps the inactive language-model detector as a possible
fallback.

**`deterministic_dataset_detection.py`** Performs the fixed-rule analysis. It
recognizes data sources, follows their use through notebook cells, and connects
them to analysis calls.

**`agent.py`** Provides the optional plain-English router. It prepares the
request, classifies it, chooses targets, runs a workflow, and checks what
changed in the notebook.

**`magic.py`** Implements the IPython extension. It finds the notebook, calls
the router, and prints the message shown after a `%provenance` request.

**`llm.py`** Holds model-provider settings, credential lookup, lazy model-client
creation, and response handling.

**`Citations/`** Contains the curated library index and BibTeX files shipped
with the package. Citation lookup does not need the network or a particular
working directory.

**`notebooks/`** Contains demos, scientific examples, self-contained
instructions, fixtures, and exploratory notebooks used during development and
manual testing.

**`benchmark/ground_truth/`** Contains the expected citation records used to
evaluate the tracked notebooks.

**`tests/`** Contains offline checks for imports, notebook parsing, dataset
detection, workflows, packaging, provider settings, and the public API.

## Main tradeoffs

**Detection is conservative.** Custom loaders, dynamic imports, hidden helper
functions, cells with syntax errors, and analysis methods outside the known list
may produce a warning or no result.

**Citation coverage is curated, not automatic.** If a library has no citation
entry, the project reports it as missing. The project does not query GitHub,
Zenodo, or Crossref while it runs.

**Dataset citation needs the notebook's current state.** Save the notebook,
run the cells that load the data, and then run the generated citation cell in
that same kernel.

**The output is citation metadata, not formatted writing.** The workflows show
DataFrames and BibTeX fields. They do not turn the result into APA or MLA text,
and they do not automatically write a `.bib` file.

**The package is installed from a checkout.** It is an editable local package.
Libraries such as `pyleoclim` and `xarray` belong to the notebook's environment,
so this project does not install them automatically.
