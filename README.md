[![license](https://img.shields.io/github/license/linkedearth/provenance-agent.svg)]()
[![NSF-2425885](https://img.shields.io/badge/NSF-2425885-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2425885)

# PaleoPAL: Provenance Agent
This repository contains an AI agent prototype regarding capturing provenance in scientific Notebooks for better transparency. 

## Goal

This repository contains code to build a standalone AI agent that takes a Jupyter Notebook and automatically generates a bibliography for the data and software used in it. 

It should extract content and identify imported libraries, imported data, and what data is actually being used after filtering. Data and software is clearly distinguised to help scientists credit every aspect of research.

This AI agent serves as a prototype for a larger agent that in addition to data/library identification will be able to: prompt the user if a citation cannot be found and update context appropriately, and help user deposit their own data when used in a notebook.

Although this agent serves as a prototype for future integration into PaleoPAL, it is separate from the three main PaleoPAL agents, and will automate the tedious task of manual citation.

## Installation

Requires Python 3.10 or newer. From the repository root:

```bash
pip install -e ".[dev]"
```

That one command installs the `provenance_agent` package, its runtime
dependencies, and the test tooling. Install it into the same environment as the
Jupyter kernel you want to analyze notebooks from - the cells this tool injects
import the package, so the kernel has to be able to see it. Nothing needs to be
added to `sys.path`.

The agent's natural-language routing calls Gemini and needs a `GOOGLE_API_KEY`.
Export it, or put it in a `.env` file at the root of the project you are working
in. An exported environment variable takes precedence over the file. The direct
Python functions below do not use the model and need no key.

## Usage

### From a notebook

```python
%load_ext provenance

%provenance cite the software
%provenance cite the datasets
%provenance cite Pyleoclim
```

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

## How is this repositoy organized

- `src/provenance_agent/` - the package. `notebook_io` reads and parses
  notebooks, `citations` looks up software citations in the packaged
  `Citations/` data, `software` and `data` own one injected cell each along with
  their public function and LangChain tool, `dataset_detection` and
  `deterministic_dataset_detection` find which variables hold cited datasets,
  `agent` is the LCEL router, and `magic` implements the IPython extension.
- `src/provenance.py` - the top-level module `%load_ext provenance` resolves. It
  forwards to `provenance_agent.magic` and holds no implementation.
- `notebooks/` - demos, worked examples, instruction bundles, test fixtures, and
  exploratory work.
- `tests/` - run with `pytest`. Fully offline; no test calls a model or a remote
  dataset service.
- `benchmark/ground_truth/` - expected software and dataset entries per
  notebook, kept as data.

## Citation

## Acknowledgement

The research presented here is supported by NSF #2425885 and the NSF REU program. 
