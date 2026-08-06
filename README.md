[![license](https://img.shields.io/github/license/linkedearth/provenance-agent.svg)]()
[![NSF-2425885](https://img.shields.io/badge/NSF-2425885-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2425885)


# PaleoPAL: Provenance Agent
This repository contains an AI agent prototype regarding capturing provenance in scientific Notebooks for better transparency. 

## Goal

This repository contains code to build a standalone AI agent that takes a Jupyter Notebook and automatically generates a bibliography for the data and software used in it. 

It should extract content and identify imported libraries, imported data, and what data is actually being used after filtering. Data and software is clearly distinguised to help scientists credit every aspect of research.

This AI agent serves as a prototype for a larger agent that in addition to data/library identification will be able to: prompt the user if a citation cannot be found and update context appropriately, and help user deposit their own data when used in a notebook.

Although this agent serves as a prototype for future integration into PaleoPAL, it is separate from the three main PaleoPAL agents, and will automate the tedious task of manual citation.

## Other Documentation

| Document | Covers |
|---|---|
| [`docs/design-decisions-summary.md`](docs/design-decisions-summary.md) | why the project is built this way, and what each file is for |
| [`docs/paleopal-integration.md`](docs/paleopal-integration.md) | notes on folding this agent into PaleoPAL |

## Installation

Requires Python 3.12 or newer. Clone the repository. 

```bash
git clone https://github.com/LinkedEarth/provenance-agent.git
cd provenance-agent
```


### Recommended (new conda environment)

To install in a new conda environment, run:

```bash
conda create -n provenance-agent python=3.12 pip
conda activate provenance-agent

python -m pip install -e ".[dev,google,data]"
```

[dev] adds pytest testing tools
To install without:
```bash
pip install -e ".[google,data]"
```
[google] specifies the llm provider
[data] also installs `pylipd` and `pyleotups`, which are needed only when
the notebook kernel executes generated PyLiPD, LiPDGraph, or PyleoTUPS
retrieval cells. 
To install without:
```bash
pip install -e ".[dev, google,data]"
```
or
```bash
pip install -e ".[google]"
```

You may specify the LLM provider alongside the package, since the
agent routing needs one. `google` is the default example; swap it for `openai`,
`anthropic`, `ollama`, or `xai`.

| Provider | Install | Default model |
|---|---|---|
| Google | `pip install -e ".[google]"` | `gemini-flash-latest` |
| OpenAI | `pip install -e ".[openai]"` | `gpt-4o-mini` |
| Anthropic | `pip install -e ".[anthropic]"` | `claude-sonnet-5` |
| Ollama (local) | `pip install -e ".[ollama]"` | `llama3.1` |
| xAI | `pip install -e ".[xai]"` | `grok-4.3` |

For compatibility, an exact `PROVENANCE_LLM_MODEL=grok-4` setting is rerouted
to `grok-4.3`; other model identifiers are preserved.

```
Note on RuntimeError: LLM provider 'xai' needs the langchain_xai package, which is not
installed. Install it with: pip install "provenance-agent[xai]"
```

You can also install without a conda environment, just don't create the conda environment.

### Before installing into an existing scientific environment

pip can silently replace conda-managed scientific packages with PyPI builds.
If you are installing into an environment you care about, check first:

```bash
pip install --dry-run -e ".[dev,google,data]"
```

The output should end with `Would install provenance-agent-0.1.0`. If it
proposes changing conda-managed packages such as `numpy`, `pandas`,
`pylipd`, or `pyleotups`, install with `--no-deps` instead and then run
`pip check`.




Regardless of installation option, **install into the same environment as the Jupyter kernel
you analyze notebooks from.** The cell the software workflow injects imports
`provenance_agent`, so that kernel has to be able to import it. If you create a
new conda environment as above, register it as the kernel you run those notebooks
in, and install your analysis libraries there too.

**Note** If the environment does not appear in VS Code's notebook kernel picker, register
it explicitly. Run these commands in the environment you want to use:

To find the exact interpreter path, activate the environment and print the
executable that will run the notebook:

```bash
conda activate provenance-agent
python -c "import sys; print(sys.executable)"
```

This should print:
`/opt/anaconda3/envs/provenance-agent/bin/python`

Copy the printed path. In VS Code, run **Python: Select Interpreter** from the
Command Palette, choose **Enter interpreter path...**, and paste or browse to
that path. Then select the same interpreter for the notebook with **Select
Kernel** → **Select Another Kernel...** → **Python Environments**. The Python
interpreter and notebook kernel are separate selections, so set both when
needed. If you are using a remote VS Code window, run the command in that same
remote environment and use its path.

Example notebooks may additionally require notebook-specific scientific packages such as `pyleoclim`, `xarray`, etc. that require further installation.





### Credentials

Add an API Key to `.env.example` copy or rename it to `.env` and
fill in the one you need. Only the natural-language layers (`%provenance` and `agent.run`) call a model.


## Usage

We recommend starting in `notebooks/demos/.` Open `workflow.ipynb` to test functions explicity. Or, directly in `paleoPCAlite.ipynb` to start testing directly.

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

 **VS Code may not detect the current notebook automatically.** Set the path
  explicitly with `%provenance_notebook path/to/notebook.ipynb` when needed.

```python
%provenance cite everything
``` 
Ask the agent in natural language what to cite. Both broad requests, like "cite the software" or "cite the data" or "cite everything," as well as specific targets such as "cite pandas and numpy." This will append code cell(s) to the notebook on disk. To see the appended cell(s) reload the file.

In VSCode:
- macOS: Cmd+Shift+P
- Windows: Ctrl+Shift+P
Choose File:Revert File. Make sure to save the file before if there are unsaved changes.

Running the code cells will display DataFrames of citation metadata. For datasets, make sure the data loading and filtering cells in the notebook have been rerun.



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
├── benchmark/ground_truth/           # expected citation records per tracked notebook;
├── notebooks/                        # demo and example notebooks; see the reference tree below
├── src/
│   ├── provenance.py                # the top-level module `%load_ext provenance`
│   │                                 # resolves. A forwarding shim over
│   │                                 # `provenance_agent.magic`
│   └── provenance_agent/
│       ├── __init__.py              # exports `cite_data` and `cite_software`
│       │                             # . The tools and `run` stay on their own
│       │                             # modules so importing the package root never
│       │                             # constructs the LLM client
│       ├── notebook_io.py           # reads `.ipynb` files. Strips IPython magics
│       │                             # and shell lines so cells parse, uses AST
│       │                             # parsing for imports, line from cells with
│       │                             # syntax errors, and
│       ├── citations.py             # software citation lookup. Reads the packaged
│       │                             # Citations/ index and merges them into one
│       │                             # DataFrame deduped by DOI. It also contains
│       │                             # the generated-cell removal helpers
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

## Citation

## Acknowledgement

The research presented here is supported by NSF #2425885 and the NSF REU program. 
