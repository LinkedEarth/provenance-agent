[![license](https://img.shields.io/github/license/linkedearth/provenance-agent.svg)]()
[![NSF-2425885](https://img.shields.io/badge/NSF-2425885-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2425885)


# PaleoPAL: Provenance Agent

This repository contains a standalone AI agent prototype for capturing provenance
in scientific Jupyter notebooks.

## Goal

This repository contains code to build a standalone AI agent that takes a Jupyter
notebook and surfaces citation metadata for the data and software used in it.
BibTeX is returned in the workflow output; the agent does not currently write a
`.bib` file.

It extracts notebook content, identifies imported libraries and datasets, and
tracks which data reaches recognized analysis calls after filtering. Data and
software are clearly distinguished so scientists can credit every aspect of
their research.

This AI agent serves as a prototype for a larger agent that in addition to data/library identification will be able to: prompt the user if a citation cannot be found and update context appropriately, and help user deposit their own data when used in a notebook.

Although this agent serves as a prototype for future integration into PaleoPAL, it is separate from the three main PaleoPAL agents, and will automate the tedious task of manual citation.

## Other Documentation

| Document | Covers |
|---|---|
| [`docs/design-decisions-summary.md`](docs/design-decisions-summary.md) | design decisions regarding the entire project and each module |
| [`docs/paleopal-integration.md`](docs/paleopal-integration.md) | notes on integration into PaleoPAL |

## Installation

The package supports Python 3.10 and newer; the recommended environment below
uses Python 3.12. An API key is required only for the natural-language routing
layer (`%provenance` and `agent.run`). The direct workflows, tests, and
deterministic dataset detector do not require a key. Clone the repository:

```bash
git clone https://github.com/LinkedEarth/provenance-agent.git
cd provenance-agent
```


### Recommended: new Conda environment

Create and activate an environment:

```bash
conda create -n provenance-agent python=3.12 pip
conda activate provenance-agent
```

Then install the package with the extras you need:

- `dev` adds pytest for running the test suite.
- `data` adds `pylipd` and `pyleotups`, which are needed only when the notebook
  kernel executes generated PyLiPD, LiPDGraph, or PyleoTUPS retrieval cells.
- One provider extra (`google`, `openai`, `anthropic`, `ollama`, or `xai`) adds
  the corresponding LLM integration. Install only the provider you intend to
  use.

For notebook use with Google and data retrieval:

```bash
python -m pip install -e ".[google,data]"
```

To include the test tools as well:

```bash
python -m pip install -e ".[dev,google,data]"
```

The agent selects one configured provider at a time. `google` is the default;
swap it for `openai`, `anthropic`, `ollama`, or `xai` as needed.

| Provider | Install | Default model |
|---|---|---|
| Google | `python -m pip install -e ".[google]"` | `gemini-flash-latest` |
| OpenAI | `python -m pip install -e ".[openai]"` | `gpt-4o-mini` |
| Anthropic | `python -m pip install -e ".[anthropic]"` | `claude-sonnet-5` |
| Ollama (local) | `python -m pip install -e ".[ollama]"` | `llama3.1` |
| xAI | `python -m pip install -e ".[xai]"` | `grok-4.3` |

For compatibility, an exact `PROVENANCE_LLM_MODEL=grok-4` setting is rerouted
to `grok-4.3`; other model identifiers are preserved.

> If you select a provider without installing its extra, the agent reports the
> missing integration and the required install command, for example:
> `python -m pip install -e ".[xai]"`.

A Conda environment is optional. If you use an existing environment, skip the
`conda create` command and install the package there.

### Before installing into an existing scientific environment

`pip` can silently replace Conda-managed scientific packages with PyPI builds.
If you are installing into an environment you care about, check first:

```bash
python -m pip install --dry-run -e ".[dev,google,data]"
```

The output should end with `Would install provenance-agent-0.1.0`. If it
proposes changing Conda-managed packages such as `numpy`, `pandas`, `pylipd`, or
`pyleotups`, install with `--no-deps` only after confirming the environment
already provides the dependencies you need, then run:

```bash
python -m pip install --no-deps -e ".[dev,google,data]"
python -m pip check
```




Regardless of installation option, **install into the same environment as the
Jupyter kernel that runs the notebooks you analyze.** Generated cells import
`provenance_agent`, so that kernel has to be able to import it. If you create a
new Conda environment as above, register it as the kernel for those notebooks
and install your analysis libraries there too.


### Credentials

The committed `.env.example` contains placeholders only. Copy it to a local
`.env` file and fill in the key for the provider you selected:

```bash
cp .env.example .env
```

Keep real credentials in `.env`; it is ignored by Git. Do not replace or commit
`.env.example` with a real key. Ollama runs locally and does not need an API key.
Only the natural-language layers (`%provenance` and `agent.run`) call a model.

The `.env` file starts with a dot, so it may be hidden by your file browser. To
show it:

- **macOS Finder:** press `Command` + `Shift` + `.` to toggle hidden files.
  In Terminal, use `ls -la` from the repository root.
- **Windows File Explorer:** select **View** → **Show** → **Hidden items**.
  In Command Prompt, use `dir /a`; in PowerShell, use `Get-ChildItem -Force`.


## Usage

Start with `notebooks/demos/workflow.ipynb`, which demonstrates the software,
data, and agent layers. The demo uses
`notebooks/demos/paleoPCAlite.ipynb` as its target and edits that notebook in
place, so review its diff before committing after a demo run.

### From a Jupyter notebook

These commands are to be run in new code cells you insert.

```python
%load_ext provenance
```

Run this once per kernel before using the other magics.

```python
%provenance_notebook path/to/notebook.ipynb
```

This optional command sets the notebook path to analyze. By default, the agent
uses the current notebook when it can detect it.

```python
%provenance cite everything
```

Ask the agent in natural language what to cite. Broad requests such as "cite
the software," "cite the data," or "cite everything" are supported, as are
specific targets such as "cite pandas and numpy." The command appends generated
cell(s) to the notebook on disk; it does not execute them. Save and reload the
notebook, then run the generated cells to display citation metadata.

In VSCode:
- macOS: Cmd+Shift+P
- Windows: Ctrl+Shift+P
Choose File:Revert File. Make sure to save the file before if there are unsaved changes.
For dataset citations, rerun the notebook's data-loading and filtering cells
before running the generated citation cells.


**Examples**
```python
%provenance cite the software
```
You may cite just the software.

```python
%provenance cite the datasets
```
You may cite just the datasets.

```python
%provenance cite Pyleoclim
```
You may specify one or more imported software libraries.

```python
%provenance cite TR04EVLI
```
You may cite a specific dataset name loaded through PyLiPD or LiPDGraph.




### Troubleshooting
**Kernel not visible** If the environment does not appear in VS Code's notebook
kernel picker, register it explicitly. Run these commands in the environment
you want to use:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name provenance-agent --display-name "Python (provenance-agent)"
```

Replace `provenance-agent` with your environment name if you are using a
different environment.

To find the exact interpreter path:

```bash
conda activate provenance-agent  # or the environment you are using
python -c "import sys; print(sys.executable)"
```

This should print something like:
`/opt/anaconda3/envs/provenance-agent/bin/python`

Copy the printed path. In VS Code, run **Python: Select Interpreter** from the
Command Palette with Cmd+Shift+P, choose **Enter interpreter path...**, and paste
that path. Then select the same interpreter for the notebook with **Select
Kernel** → **Select Another Kernel...** → **Python Environments**. The Python
interpreter and notebook kernel are separate selections, so set both when
needed. If you are using a remote VS Code window, run the command in that same
remote environment and use its path.

**Module not found** Example notebooks may require additional scientific
packages such as `pyleoclim` or `xarray`. Install those packages in the same
environment as the notebook kernel.

**API key not found** Confirm that `.env` is at the repository root, contains
the key for the selected provider, and is loaded by the same environment as the
notebook kernel. If you just created or changed `.env`, restart the Jupyter
kernel.

**Changes to code don't seem to register** Make sure you restart the kernel.

**Can't see injected cell** Make sure you refresh the current notebook.

**Undefined variable after running injected code cell** Make sure you have rerun the data loading and filtering cells of the notebook.

**VS Code may not detect the current notebook automatically.** Set the path
explicitly with `%provenance_notebook path/to/notebook.ipynb` when needed.


## Testing and benchmarking

Run the test suite from the repository root in the development environment:

```bash
python -m pytest tests/ -q
```

The benchmark evaluates the deterministic software and dataset detectors
against the curated records in `benchmark/ground_truth/`. 

```bash
python benchmark/run_ground_truth.py
```

The command prints precision, recall, and F1 scores and writes the detailed
JSON report to `benchmark/results/ground_truth_results.json`. To evaluate one
notebook or choose a different report path:

```bash
python benchmark/run_ground_truth.py --notebook paleoPCAlite
python benchmark/run_ground_truth.py --output /tmp/provenance-results.json
```


## Technical details


```text
provenance-agent/
├── pyproject.toml                    # setuptools src-layout config; runtime dependencies,
│                                     # the `dev` and `data` extras, provider extras,
│                                     # and Citations/ as package data
├── .gitignore                        # local secrets, build products, and generated files
├── LICENSE                           # project license
├── README.md                         # installation, usage, debugging, and limitations
├── benchmark/ground_truth/           # curated records used by the benchmark
├── notebooks/                        # demo and example notebooks; see the reference tree below
├── src/
│   ├── provenance.py                # the top-level module `%load_ext provenance`
│   │                                 # resolves. A forwarding shim over
│   │                                 # `provenance_agent.magic`
│   └── provenance_agent/
│       ├── __init__.py              # exports `cite_data` and `cite_software`.
│       │                             # The tools and `run` stay on their own
│       │                             # modules so importing the package root never
│       │                             # constructs the LLM client
│       ├── notebook_io.py           # reads `.ipynb` files, strips IPython magics
│       │                             # and shell lines so cells parse, extracts
│       │                             # imports with AST parsing, and manages
│       │                             # generated-cell markers
│       ├── citations.py             # loads the packaged Citations/ index,
│       │                             # builds citation DataFrames, and provides
│       │                             # generated-cell removal helpers
│       ├── software.py              # the software workflow: imported-library
│       │                             # detection, citation-cell source and injection,
│       │                             # `cite_software`, and its tool
│       ├── data.py                  # the data workflow: per-source retrieval
│       │                             # blocks, the single injected cell, target
│       │                             # handling, `cite_data`, and its tool.
│       ├── dataset_detection.py     # the data detection interface. `detect_datasets()`
│       │                             # and `detect_datasets_with_diagnostics()` call
│       │                             # the deterministic analyzer below. It also holds
│       │                             # the deprecated LLM detector as a documented,
│       │                             # inactive rollback path
│       ├── deterministic_dataset_detection.py
│       │                             # the dataset tracer. Builds a versioned data-flow
│       │                             # graph over notebook cells: assignments record
│       │                             # dependencies, source groups, and object families;
│       │                             # recognizers attach sources to
│       │                             # LiPD/PyleoTUPS/LiPDGraph/xarray/pandas loaders;
│       │                             # analysis calls are sinks. Results walk each
│       │                             # sink's dependency to the nearest source, with a
│       │                             # fallback to all unique dataframes if no analysis
│       │                             # is found.
│       ├── agent.py                 # the LCEL router: `prepare_context`, `classify`,
│       │                             # `resolve_targets`, `dispatch`, and `verify` are
│       │                             # named Runnable stages.
│       ├── magic.py                 # the IPython extension implementation. It resolves
│       │                             # the notebook path, calls `agent.run()`
│       ├── llm.py                   # sets up the AI model used by the agent. Creates
│       │                             # the shared chat client, and converts model
│       │                             # responses to text.
│       └── Citations/               # packaged citation data: `library_citations.yml`
│                                     # (the index) plus one `.bib` file per library
└── tests/                           # pytest suite; see the reference tree below
```

The main tree intentionally keeps `notebooks/` and `tests/` at directory level.
The following reference trees are available when you need to locate a specific
example or test file.

### Notebook reference tree

```text
notebooks/
├── demos/                            # workflow.ipynb, the single demo/dev notebook,
│                                     # plus paleoPCAlite.ipynb, the notebook it edits
├── examples/                         # worked scientific notebooks and the deterministic-detection corpus
└── instructions/                     # self-contained NotebookN bundles, each
                                      # with its own .lpd sibling
```

### Test reference tree

```text
tests/
├── test_agent.py                     # LCEL routing and dispatch behavior
├── test_citations.py                 # packaged citation lookup and DataFrames
├── test_data.py                      # data workflow and retrieval-cell behavior
├── test_dataset_detection.py         # public data detection and diagnostics
├── test_deterministic_dataset_detection.py # AST/data-flow detector behavior
├── test_import_hygiene.py            # package import and path hygiene
├── test_llm.py                       # provider registry, credential discovery,
│                                     # and response helpers
├── test_magic.py                     # IPython extension behavior
├── test_notebook_io.py               # notebook parsing and generated-cell lifecycle
├── test_notebooks.py                 # notebook structure and path validation
├── test_packaging.py                 # editable-install and package-data checks
├── test_provenance_shim.py           # extension shim
├── test_public_api.py                # package-level public imports
└── test_software.py                  # software workflow and citation-cell behavior
```

The main runtime path is:

1. `notebook_io` reads notebook code cells.
2. `dataset_detection` performs dataset detection deterministically by default and keeps the deprecated LLM helpers as a fallback.
3. `data` or `software` builds one citation cell respectively and appends it to the
   notebook on disk.
4. The user runs the generated cell in the notebook's own kernel to retrieve
   and display citation metadata.


## Known limitations

- **Notebook changes happen on disk.** The notebook must be saved and reloaded to see the generated citation cell. Dataset retrieval also requires the notebook's own data loading/filtering cells to have run first.
- **Dataset detection has limited coverage.** It recognizes supported data
  loaders and analysis patterns; custom loaders, dynamic imports, and unused
  datasets may not be detected, and will report warnings.
- **Software citation coverage is limited to the static index
  ([Citations/](src/provenance_agent/Citations/)).** Libraries without an
  entry are reported as missing rather than receiving an automatically found
  citation. Additionally, trying to cite software libraries not imported in a notebook will return a warning.
- **Citations are returned as data.** The workflows display citation metadata
  in DataFrames; they do not format APA text.
- **PyleoTUPS datasets cannot be specified by name.** PyleoTUPS datasets are
  loaded by study ID inside the live provider object, so a specific PyleoTUPS
  name is not available to the router. PyLiPD and LiPDGraph datasets can be
  selected by dataset name.


## Citation

## Acknowledgement

The research presented here is supported by NSF #2425885 and the NSF REU program. 
