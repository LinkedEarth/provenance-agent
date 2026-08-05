[![license](https://img.shields.io/github/license/linkedearth/provenance-agent.svg)]()
[![NSF-2425885](https://img.shields.io/badge/NSF-2425885-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2425885)


# PaleoPAL: Provenance Agent
This repository contains an AI agent prototype regarding capturing provenance in scientific Notebooks for better transparency. 

## Goal

This repository contains code to build a standalone AI agent that takes a Jupyter Notebook and automatically generates a bibliography for the data and software used in it. 

It should extract content and identify imported libraries, imported data, and what data is actually being used after filtering. Data and software is clearly distinguised to help scientists credit every aspect of research.

This AI agent serves as a prototype for a larger agent that in addition to data/library identification will be able to: prompt the user if a citation cannot be found and update context appropriately, and help user deposit their own data when used in a notebook.

Although this agent serves as a prototype for future integration into PaleoPAL, it is separate from the three main PaleoPAL agents, and will automate the tedious task of manual citation.

## Documentation

| Document | Covers |
|---|---|
| [`docs/documentation-draft.md`](docs/documentation-draft.md) | the full manual: installation, usage, debugging, and known limitations |
| [`docs/design-decisions.md`](docs/design-decisions.md) | why the project is built this way, project-wide and per module |
| [`docs/paleopal-integration.md`](docs/paleopal-integration.md) | notes on folding this agent into PaleoPAL |

This README is the short version of the first of those.

## Installation

Requires Python 3.10 or newer. Clone the repository. 

```bash
git clone https://github.com/LinkedEarth/provenance-agent.git
cd provenance-agent
```

A development environment is recommended; from the repository root run:

```bash
pip install -e ".[dev]"
```

For a runtime environment without test tools, run:

```bash
pip install -e .
```

To install in a new conda environment, run:

```bash
conda create -n provenance-agent python=3.12 pip
conda activate provenance-agent

python -m pip install -e .
```

### Installing an LLM provider

The commands above install no LLM integration. Only the natural-language layers
(`%provenance` and `agent.run`) use a model, and rather than committing every
user to one vendor's client, each provider is a separate extra. Add the one you
intend to use:

| Provider | Install | Default model |
|---|---|---|
| Google | `pip install -e ".[google]"` | `gemini-flash-latest` |
| OpenAI | `pip install -e ".[openai]"` | `gpt-4o-mini` |
| Anthropic | `pip install -e ".[anthropic]"` | `claude-sonnet-5` |
| Ollama (local) | `pip install -e ".[ollama]"` | `llama3.1` |
| xAI | `pip install -e ".[xai]"` | `grok-4` |

Extras combine, so a development install with OpenAI is one command:

```bash
pip install -e ".[dev,openai]"
```

Integrations are imported lazily, so you install exactly the provider you
selected and nothing else. If you skip this step, the first `%load_ext
provenance` fails with a `RuntimeError` that names the missing package and the
exact command to install it. The direct Python functions described under Usage
need no provider at all.

### If you already use PaleoPAL

The five providers above are the same set PaleoPAL supports, and the key
variables are the same names, so an existing PaleoPAL key works here untouched:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `XAI_API_KEY`.

If `PROVENANCE_LLM_PROVIDER` is unset, the agent also reads PaleoPAL's
`DEFAULT_LLM_PROVIDER`, accepting its `grok` spelling for xAI. Setting
`PROVENANCE_LLM_PROVIDER` overrides it, so the two can run on different vendors.

PaleoPAL's per-provider model variables (`OPENAI_MODEL`, `CLAUDE_MODEL`,
`GOOGLE_MODEL`, `GROK_MODEL`, `OLLAMA_MODEL`) are intentionally ignored. Its
defaults are heavyweight reasoning models chosen for multi-agent work, while
this agent makes a single short classification call. Use
`PROVENANCE_LLM_MODEL` to pick a model here.

Note that PaleoPAL keeps its keys in `backend/.env`, while this agent searches
upward from the working directory, so copy the keys into a `.env` at the root of
your notebook project.

### Before installing into an existing scientific environment

pip can silently replace conda-managed scientific packages with PyPI builds.
If you are installing into an environment you care about, check first:

```bash
pip install --dry-run -e ".[dev]"
```

The output should end with `Would install provenance-agent-0.1.0` and nothing
else. If it proposes changing `numpy`, `pandas`, `pylipd`, `pyleotups`, or
`pyleoclim`, install with `--no-deps` instead and then run `pip check`.

Example notebooks may additionally require notebook-specific scientific packages such as `pyleoclim`, `xarray`, etc. that require further installation.

### Credentials

Only the natural-language layers (`%provenance` and `agent.run`) call a model.
The direct Python functions described under Usage need no key of any kind.

Put the key for the provider you selected in a `.env` file at the root of the
project you are working in, or export it. An exported environment variable wins
over the file. `.env.example` lists the recognized names; copy it to `.env` and
fill in the one you need.

```bash
# .env
GOOGLE_API_KEY=...
```

The key variable depends on the provider: `GOOGLE_API_KEY` (or `GEMINI_API_KEY`),
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `XAI_API_KEY`. Ollama runs locally and
needs none.

Keep the `.env` at the repository root. Inside a Jupyter kernel, python-dotenv
resolves the lookup from the kernel's working directory, so a `.env` that only
exists under `src/` is invisible to notebooks.


## Usage

### From a Jupyter notebook

```python
%load_ext provenance

%provenance cite the software
%provenance cite the datasets
%provenance cite Pyleoclim
```

```python
%load_ext provenance
``` 
Must be done first to import the `provenance` module.

```python
%provenance_notebook path/to/notebook.ipynb
```
Is an optional command to set the the notebook path to cite; by default it is the current notebook.

```python
%provenance cite everything
``` 
Ask the agent in natural language what to cite. Both broad requests, like "cite the software" or "cite the data" or "cite everything," as well as specific targets such as "cite pandas and numpy."



Each request appends one cell to the notebook on disk and reports what it
injected. Reload the notebook and run the new cell to see the citations: the
software cell displays a DataFrame of citation metadata, and the dataset cell
performs the retrievals and displays each source's metadata frame. The notebook
is read from disk, so save it before asking.

If notebook auto-detection fails - it usually does in VSCode - set the path once
per session:

```python
%provenance_notebook path/to/notebook.ipynb
```



### From Python

```python
from provenance_agent import cite_data, cite_software

cite_software("notebook.ipynb")                      # every imported library
cite_software("notebook.ipynb", libraries="pyleoclim")
cite_data("notebook.ipynb")                          # every detected dataset
cite_data("notebook.ipynb", targets="Ocn-RedSea.Felis.2000")
```

Both functions modify the notebook in place unless given an `output_path`, and
return what they injected a cell for rather than the citations themselves.
`cite_software` returns the library names; `cite_data` returns `[variable,
tool]` pairs. Dataset detection is static, but dataset *retrieval* runs in the
notebook's own kernel, which is why the citations are the injected cell's
output.

The LangChain tools and the natural-language router live in their own modules,
so importing the two functions above does not construct the model client:

```python
from provenance_agent.data import cite_data_tool
from provenance_agent.software import cite_software_tool
from provenance_agent.agent import run
```

## Technical details

The repository uses a `src`-layout Python package. The top-level shim keeps
`%load_ext provenance` stable while the package modules separate notebook I/O,
dataset detection, citation lookup, workflow generation, and LLM routing:

```text
provenance-agent/
├── pyproject.toml                         # setuptools src-layout config; runtime deps,
│                                          # the `dev` extra, one extra per optional LLM
│                                          # provider, and Citations/ as package data
├── .gitignore                             # local secrets, build products, and generated files
├── LICENSE                                # project license
├── README.md                              # installation, usage, debugging, and limitations
├── benchmark/ground_truth/                # expected citation records per tracked notebook;
│                                         # data only, since the scoring runner was removed
├── notebooks/                            # demos, examples, fixtures, and exploration;
│                                         # see the reference tree below
├── src/
│   ├── provenance.py                # the top-level module `%load_ext provenance`
│   │                               # resolves. A forwarding shim over
│   │                               # `provenance_agent.magic` with no logic of
│   │                               # its own; the one intentional top-level module
│   └── provenance_agent/
│       ├── __init__.py              # exports `cite_data` and `cite_software`, and
│       │                             # nothing else. The tools and `run` stay on
│       │                             # their own modules so importing the package
│       │                             # root never constructs the LLM client
│       ├── notebook_io.py           # reads `.ipynb` files. Strips IPython magics
│       │                             # and shell lines so cells parse, walks the
│       │                             # AST for imports, recovers imports line by
│       │                             # line from cells with syntax errors, and
│       │                             # owns `is_generated_cell()`
│       ├── citations.py             # software citation lookup. Reads the packaged
│       │                             # Citations/ index and `.bib` files through
│       │                             # `importlib.resources` and merges them into
│       │                             # one DataFrame deduped by DOI. It also holds
│       │                             # the generated-cell removal helpers shared
│       │                             # by both workflows
│       ├── software.py              # the software workflow: imported-library
│       │                             # detection, citation-cell source and
│       │                             # injection, `cite_software`, and its tool
│       ├── data.py                  # the data workflow: per-source retrieval
│       │                             # blocks, the single injected cell, target
│       │                             # handling, `cite_data`, and its tool. It
│       │                             # lifts the LiPDGraph endpoint out of the
│       │                             # notebook by AST so a notebook pointed at
│       │                             # a different repository is handled correctly
│       ├── dataset_detection.py     # the detection facade. `detect_datasets()`
│       │                             # and `detect_datasets_with_diagnostics()`
│       │                             # delegate to the analyzer below. It also
│       │                             # holds the deprecated LLM detector as a
│       │                             # documented, inactive rollback path
│       ├── deterministic_dataset_detection.py
│       │                             # the analyzer (~1,600 lines). Builds a
│       │                             # versioned data-flow graph over notebook
│       │                             # cells: assignments record dependencies,
│       │                             # source groups, and object families;
│       │                             # recognizers attach sources to
│       │                             # LiPD/PyleoTUPS/LiPDGraph/xarray/pandas
│       │                             # loaders; analysis calls are sinks. Results
│       │                             # walk each sink's dependency closure to the
│       │                             # nearest source boundary, with a per-source
│       │                             # fallback to live terminal tables. It never
│       │                             # executes notebook code
│       ├── agent.py                 # the LCEL router: `prepare_context`,
│       │                             # `classify`, `resolve_targets`, `dispatch`,
│       │                             # and `verify` are named Runnable stages.
│       │                             # Classification is Pydantic-validated JSON;
│       │                             # verification diffs the notebook before and
│       │                             # after without running injected cells
│       ├── magic.py                 # the IPython extension implementation. It
│       │                             # resolves the notebook path, calls
│       │                             # `agent.run()`, and renders the envelope;
│       │                             # it contains no citation or routing logic
│       ├── llm.py                   # the `PROVIDERS` registry, `build_llm()`, the
│       │                             # shared chat client, dotenv credential
│       │                             # discovery, and the response-to-text helper.
│       │                             # Integrations are imported lazily, so only
│       │                             # the selected provider must be installed, and
│       │                             # the client itself is built on first access
│       │                             # via a module `__getattr__`, so importing
│       │                             # costs no credentials
│       └── Citations/               # packaged citation data: `library_citations.yml`
│                                   # (the index) plus one `.bib` file per library
└── tests/                           # pytest suite; fully offline; see the
                                    # reference tree below
```

The main tree intentionally keeps `notebooks/` and `tests/` at directory level.
The following reference trees are available when you need to locate a specific
example or test file.

### Notebook reference tree

```text
notebooks/
├── demos/                            # the four workflow demos
│   ├── data_workflow.ipynb
│   ├── overall_workflow.ipynb
│   ├── provenance_magic.ipynb
│   └── software_workflow.ipynb
├── examples/                         # worked scientific notebooks and the
│   ├── 02a-query_lipd_graph.ipynb    # deterministic-detection corpus
│   ├── C02_b_DA_with_individual_seasonality.ipynb
│   ├── paleoPCA.ipynb
│   ├── paleoPCAlite.ipynb
│   └── comparing-simulated-reconstructed-climate/
│       ├── CMIP6_LMR.ipynb
│       ├── VICS_dashboard.ipynb
│       ├── data_from_esm_cloudcat.ipynb
│       ├── spatial_snapshots_xarray_bonuses.ipynb
│       └── widget_primer.ipynb
├── instructions/                     # self-contained NotebookN bundles
│   ├── Notebook1/                    # each includes an .ipynb and .lpd
│   ├── Notebook2/
│   ├── Notebook3/
│   └── Notebook4/
├── fixtures/                         # test notebooks, bibliography files,
│   ├── sample.ipynb                  # Pages2k/, and .lpd datasets
│   ├── test_magic_commands.ipynb
│   ├── mybiblio.bib
│   ├── Ocn-Palmyra.Nurhati.2011.lpd
│   └── Pages2k/*.lpd
└── exploration/                      # scratch notebooks and single-library studies
```

### Test reference tree

```text
tests/
├── test_agent.py                     # LCEL routing and dispatch behavior
├── test_citations.py                 # packaged citation lookup and DataFrames
├── test_data.py                      # data workflow and retrieval-cell behavior
├── test_dataset_detection.py         # public detection facade and diagnostics
├── test_deterministic_dataset_detection.py # AST/data-flow detector behavior
├── test_import_hygiene.py            # package import and path hygiene
├── test_llm.py                       # provider registry, credential discovery,
│                                     # and response helpers
├── test_magic.py                     # IPython extension behavior
├── test_notebook_io.py               # notebook parsing and generated-cell lifecycle
├── test_notebooks.py                 # notebook structure and path validation
├── test_packaging.py                 # editable-install and package-data checks
├── test_provenance_shim.py           # top-level extension shim
├── test_public_api.py                # package-level public imports
└── test_software.py                  # software workflow and citation-cell behavior
```

The main runtime path is:

1. `notebook_io` reads notebook code cells while preserving their order.
2. `dataset_detection` delegates dataset detection to the deterministic
   data-flow analyzer and keeps the deprecated LLM helpers as a fallback.
3. `data` or `software` builds one citation cell and appends it to the
   notebook on disk.
4. The user runs the generated cell in the notebook's own kernel to retrieve
   and display citation metadata.

The `src/provenance.py` file is intentionally kept at the top level because
`%load_ext provenance` is the public notebook command. It forwards to
`provenance_agent.magic`; the package contains the implementation. The
`Citations/` directory is packaged with the Python distribution so citation
lookup works after installation and does not depend on the repository's
working directory.

The test suite is designed to run offline. It exercises package imports,
notebook parsing, deterministic detection, citation-cell generation, the
IPython shim, and packaged resources without calling an LLM or a remote
dataset service.

## Citation

## Acknowledgement

The research presented here is supported by NSF #2425885 and the NSF REU program. 
