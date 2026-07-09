# Orchestrator + APA Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the software and data workflows as two callable tools (`cite_software`, `cite_data`) with LangChain `StructuredTool` wrappers, with APA as the default output format, so a future NL agent can route to them.

**Architecture:** A new `src/orchestrator.py` holds two thin wrappers over existing functions - no new citation logic. `cite_software` runs in-process (`parse_notebook` -> `collect_library_entries` -> `render_apa`). `cite_data` reuses `generate_data_workflow` to inject retrieval cells whose output is BibTeX or (default) APA. Each tool also has a LangChain `StructuredTool` wrapper whose description is the routing "system prompt".

**Tech Stack:** Python 3.12, LangChain (`langchain_core.tools.StructuredTool`), Gemini via `src/llm.py`, pybtex, nbformat, pytest.

## Global Constraints

- Run everything in the `lang` conda env: `/opt/anaconda3/envs/lang/bin/python`. It is the only env with pylipd/pyleotups/pyleoclim.
- Python 3.12.
- Never use the em dash; use a plain dash `-`.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- Commit at each task boundary. Do not open a PR unless asked.
- Detection is LLM-based and retrieval is hardcoded/live-kernel; no SPARQL-direct pathway; no PaleoPAL agent delegation.
- APA is the default output format (`fmt="apa"`); `fmt="bibtex"` skips the Gemini call.
- Do NOT use the ENSO dataset for development (it is the final eval set).
- Work happens on the `orchestrator` branch (already cut from `data-workflow`).

---

## File Structure

- **Modify `src/data_workflow.py`**: `filter_datasets` gains `str | list[str]` variable support; `build_retrieval_cell`, `inject_retrieval_cells`, and `generate_data_workflow` gain an `fmt` parameter that makes the injected cell render APA in-kernel.
- **Create `src/orchestrator.py`**: `cite_software`, `cite_data`, and the two `StructuredTool` wrappers plus a shared `_check_fmt` helper.
- **Modify `tests/test_data_workflow.py`**: add `filter_datasets` list tests and `fmt` cell-content tests.
- **Create `tests/test_orchestrator.py`**: unit tests for both tools (offline / monkeypatched) and the tool wrappers.
- **Modify `CLAUDE.md`** (local, gitignored): add `orchestrator.py` to the repo structure.

---

## Task 1: filter_datasets accepts a list of variables

**Files:**
- Modify: `src/data_workflow.py` (the `filter_datasets` function, ~lines 85-106)
- Test: `tests/test_data_workflow.py`

**Interfaces:**
- Produces: `filter_datasets(pairs, tool=None, variable=None)` where `variable` may now be `str | list[str] | None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_data_workflow.py` (after the existing `filter_datasets` tests):

```python
def test_filter_by_variable_list():
    assert filter_datasets(PAIRS, variable=["D", "filtered_df2"]) == [
        ["D", "PyLiPD"], ["filtered_df2", "LiPDGraph"]
    ]


def test_filter_by_variable_str_still_works():
    assert filter_datasets(PAIRS, variable="ds") == [["ds", "PyleoTUPS"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_data_workflow.py::test_filter_by_variable_list -v`
Expected: FAIL (a list `variable` matches nothing, so the result is `[]`, not the expected two pairs).

- [ ] **Step 3: Write minimal implementation**

In `src/data_workflow.py`, replace the `variable` branch of `filter_datasets`:

```python
    if variable is not None:
        wanted = {variable} if isinstance(variable, str) else set(variable)
        result = [p for p in result if p[0] in wanted]
```

Update the docstring `Args` line for `variable` to:

```python
        variable: if given, keep only pairs for this variable name or, if a
            list, any variable in it
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_data_workflow.py -v`
Expected: PASS (all data-workflow tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/data_workflow.py tests/test_data_workflow.py
git commit -m "filter_datasets: accept a list of variable targets"
```

---

## Task 2: fmt parameter threads APA rendering into the injected cell

**Files:**
- Modify: `src/data_workflow.py` (`build_retrieval_cell`, `inject_retrieval_cells`, `generate_data_workflow`)
- Test: `tests/test_data_workflow.py`

**Interfaces:**
- Consumes: `_LIPDVERSE_ENDPOINT`, `extract_lipdgraph_endpoint` (existing).
- Produces:
  - `build_retrieval_cell(variable, tool, endpoint=None, fmt="bibtex")`
  - `inject_retrieval_cells(nb, pairs, endpoint=None, fmt="bibtex")`
  - `generate_data_workflow(notebook_path, tool=None, variable=None, output_path=None, fmt="bibtex")`
  - When `fmt="apa"`, the cell collects BibTeX into `_bib_{variable}` and prints `render_bibtex_strings_to_apa(_bib_{variable})`; when `fmt="bibtex"` it prints the BibTeX (unchanged behavior).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data_workflow.py`:

```python
def test_pylipd_cell_apa_renders_via_bibliography():
    cell = build_retrieval_cell("D", "PyLiPD", fmt="apa")
    assert "_bib_D, _ = D.get_bibtex(remote=True)" in cell
    assert "from bibliography import render_bibtex_strings_to_apa" in cell
    assert "print(render_bibtex_strings_to_apa(_bib_D))" in cell


def test_pyleotups_cell_apa_wraps_publications():
    cell = build_retrieval_cell("ds", "PyleoTUPS", fmt="apa")
    assert "ds.get_publications()" in cell
    assert "render_bibtex_strings_to_apa(_bib_ds)" in cell


def test_bibtex_fmt_is_unchanged_default():
    cell = build_retrieval_cell("D", "PyLiPD")
    assert 'print("\\n".join(_bib_D))' in cell
    assert "render_bibtex_strings_to_apa" not in cell
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_data_workflow.py::test_pylipd_cell_apa_renders_via_bibliography -v`
Expected: FAIL with `TypeError: build_retrieval_cell() got an unexpected keyword argument 'fmt'`.

- [ ] **Step 3: Rewrite build_retrieval_cell**

In `src/data_workflow.py`, replace the whole `build_retrieval_cell` body so it builds the BibTeX into `_bib_{variable}` first, then chooses the output line by `fmt`:

```python
def build_retrieval_cell(
    variable: str,
    tool: str,
    endpoint: str | None = None,
    fmt: str = "bibtex",
) -> str:
    """
    Builds the Python source for a single dataset's citation-retrieval cell.

    Args:
        variable: the notebook variable holding the dataset (from detection)
        tool: the dataset's source library - "PyLiPD", "PyleoTUPS", or
            "LiPDGraph" (case-insensitive)
        endpoint: LiPDGraph only - the graph endpoint the notebook queried
            (from extract_lipdgraph_endpoint). Falls back to _LIPDVERSE_ENDPOINT
            when None.
        fmt: "bibtex" (default) prints the raw BibTeX; "apa" renders it to APA
            in-kernel via bibliography.render_bibtex_strings_to_apa.

    Returns:
        Python source that, run in the notebook's kernel, prints the dataset's
        citations

    Raises:
        ValueError: if tool is not one of the supported dataset sources
    """
    t = tool.lower()

    if t == "pylipd":
        body = f"_bib_{variable}, _ = {variable}.get_bibtex(remote=True)\n"
    elif t == "pyleotups":
        body = (
            f"_pub_{variable}, _ = {variable}.get_publications()\n"
            f'_bib_{variable} = [_pub_{variable}.to_string(bib_format="bibtex")]\n'
        )
    elif t == "lipdgraph":
        body = (
            "from pylipd.lipd import LiPD\n"
            f'_names_{variable} = {variable}["dataSetName"].unique().tolist()\n'
            f"_lipd_{variable} = LiPD()\n"
            f'_lipd_{variable}.set_endpoint("{endpoint or _LIPDVERSE_ENDPOINT}")\n'
            f"_lipd_{variable}.load_remote_datasets(_names_{variable})\n"
            f"_bib_{variable}, _ = _lipd_{variable}.get_bibtex(remote=True)\n"
        )
    else:
        raise ValueError(f"Unsupported dataset tool: {tool!r}")

    if fmt == "apa":
        out = (
            "from bibliography import render_bibtex_strings_to_apa\n"
            f"print(render_bibtex_strings_to_apa(_bib_{variable}))"
        )
    else:
        out = f'print("\\n".join(_bib_{variable}))'

    return body + out
```

- [ ] **Step 4: Thread fmt through inject_retrieval_cells and generate_data_workflow**

In `inject_retrieval_cells`, add the `fmt` parameter and pass it down:

```python
def inject_retrieval_cells(
    nb: nbformat.NotebookNode,
    pairs: list[list[str]],
    endpoint: str | None = None,
    fmt: str = "bibtex",
) -> nbformat.NotebookNode:
```

and change the append line to:

```python
        nb.cells.append(
            nbformat.v4.new_code_cell(build_retrieval_cell(variable, tool, endpoint, fmt))
        )
```

In `generate_data_workflow`, add `fmt: str = "bibtex"` to the signature (after `output_path`) and change the inject call to:

```python
    inject_retrieval_cells(nb, pairs, endpoint, fmt)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_data_workflow.py -v`
Expected: PASS (the three new tests plus all existing ones - the bibtex output is unchanged, so `test_pylipd_cell_calls_get_bibtex_on_variable`, `test_pyleotups_cell_calls_get_publications_on_variable`, and the LiPDGraph tests still pass).

- [ ] **Step 6: Commit**

```bash
git add src/data_workflow.py tests/test_data_workflow.py
git commit -m "data_workflow: add fmt=apa that renders citations in the injected cell"
```

---

## Task 3: cite_software tool

**Files:**
- Create: `src/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `notebook_parser.parse_notebook`, `notebook_parser.validate_libraries`, `bibliography.collect_library_entries`, `bibliography.render_apa`.
- Produces:
  - `_check_fmt(fmt)` -> raises `ValueError` on anything but `"apa"`/`"bibtex"`.
  - `cite_software(notebook_path, libraries=None, citation_types=None, fmt="apa") -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator.py`:

```python
"""
Unit tests for orchestrator.py. Offline where possible: cite_software's bibtex
path reads only local Citations/, and the apa path is exercised by monkeypatching
bibliography.render_apa so no Gemini call is made.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator import _check_fmt, cite_software

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sample.ipynb")


def test_check_fmt_rejects_unknown():
    with pytest.raises(ValueError):
        _check_fmt("markdown")


def test_cite_software_all_bibtex_contains_a_library():
    out = cite_software(SAMPLE, fmt="bibtex")
    assert "pyleoclim" in out.lower()


def test_cite_software_one_library_bibtex():
    out = cite_software(SAMPLE, libraries="pyleoclim", fmt="bibtex")
    assert "pyleoclim" in out.lower()
    assert "numpy" not in out.lower()


def test_cite_software_by_citation_type_software_only():
    out = cite_software(SAMPLE, libraries="pyleoclim", citation_types=["software"], fmt="bibtex")
    assert "pyleoclim_software" in out


def test_cite_software_reports_not_imported_library():
    out = cite_software(SAMPLE, libraries=["definitely_not_here"], fmt="bibtex")
    assert "definitely_not_here" in out


def test_cite_software_apa_routes_to_render(monkeypatch):
    import bibliography
    monkeypatch.setattr(bibliography, "render_apa", lambda entries: "APA_SENTINEL")
    out = cite_software(SAMPLE, libraries="pyleoclim", fmt="apa")
    assert out.startswith("APA_SENTINEL")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator'`.

- [ ] **Step 3: Write the implementation**

Create `src/orchestrator.py`:

```python
"""
Orchestrator: exposes the software and data workflows as two tools that route to
the correct workflow with the correct arguments.

Purpose:
    A future natural-language agent ("@provenance agent, generate citations")
    will route requests to these tools. For now they are called directly. Each
    tool is a thin wrapper over existing functions - no new citation logic.

Implementation:
    - cite_software(notebook_path, libraries, citation_types, fmt): in-process.
      parse_notebook -> collect_library_entries -> render_apa (when fmt="apa").
    - cite_data(notebook_path, targets, fmt, output_path): wraps
      generate_data_workflow, which injects a retrieval cell per dataset whose
      output is the citation. Data citations exist as cell output, not a return
      value, because retrieval needs the live kernel objects.
    - cite_software_tool / cite_data_tool: LangChain StructuredTool wrappers whose
      descriptions are the routing "system prompts".

Design decisions:
    - fmt defaults to "apa" (the finished product is a human-readable
      bibliography); fmt="bibtex" skips the Gemini call for the raw artifact.
    - Heavy imports (notebook_parser, bibliography, data_workflow) are deferred
      into the functions so importing this module stays cheap and side-effect free.
"""

_VALID_FMT = ("apa", "bibtex")


def _check_fmt(fmt: str) -> None:
    """Raises ValueError unless fmt is 'apa' or 'bibtex'."""
    if fmt not in _VALID_FMT:
        raise ValueError(f"fmt must be one of {_VALID_FMT}, got {fmt!r}")


def cite_software(
    notebook_path: str,
    libraries=None,
    citation_types: list[str] | None = None,
    fmt: str = "apa",
) -> str:
    """
    Cites the software libraries a notebook imports.

    Args:
        notebook_path: path to the .ipynb to analyze
        libraries: None (all imported libraries), a single name, or a list of
            names to cite
        citation_types: optional filter - "paper" and/or "software"
        fmt: "apa" (default) or "bibtex"

    Returns:
        the citation text (APA or BibTeX); when specific libraries were asked for
        but are not imported, a note line is appended for each
    """
    _check_fmt(fmt)
    from notebook_parser import parse_notebook, validate_libraries
    from bibliography import collect_library_entries, render_apa

    available = parse_notebook(notebook_path)
    if libraries is None:
        wanted, not_found = available, []
    else:
        requested = [libraries] if isinstance(libraries, str) else list(libraries)
        wanted, not_found = validate_libraries(requested, available)

    entries = collect_library_entries(wanted, citation_types)
    if fmt == "apa":
        body = render_apa(entries)
    else:
        body = entries.to_string(bib_format="bibtex")

    if not_found:
        notes = "\n".join(f"[Not imported in notebook: {lib}]" for lib in not_found)
        body = f"{body}\n\n{notes}" if body else notes
    return body
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.py tests/test_orchestrator.py
git commit -m "orchestrator: add cite_software tool"
```

---

## Task 4: cite_data tool

**Files:**
- Modify: `src/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `data_workflow.generate_data_workflow` (with `fmt` from Task 2), `filter_datasets` list support (Task 1).
- Produces: `cite_data(notebook_path, targets=None, fmt="apa", output_path=None) -> list[list[str]]` - returns the injected `[variable, tool]` pairs and writes the notebook with retrieval cells appended.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_orchestrator.py`:

```python
def _write_lipdgraph_notebook(path):
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        "url = 'https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic'\n"
        "filtered_df2 = None"
    ))
    with open(path, "w") as f:
        nbformat.write(nb, f)


def test_cite_data_injects_apa_cell(tmp_path, monkeypatch):
    import dataset_detection
    monkeypatch.setattr(
        dataset_detection, "detect_datasets",
        lambda code: [["filtered_df2", "LiPDGraph"]],
    )
    nb_in = tmp_path / "in.ipynb"
    nb_out = tmp_path / "out.ipynb"
    _write_lipdgraph_notebook(str(nb_in))

    from orchestrator import cite_data
    pairs = cite_data(str(nb_in), fmt="apa", output_path=str(nb_out))
    assert pairs == [["filtered_df2", "LiPDGraph"]]

    import nbformat
    out = nbformat.read(str(nb_out), as_version=4)
    injected = out.cells[-1].source
    assert "render_bibtex_strings_to_apa(_bib_filtered_df2)" in injected
    assert "repositories/LiPDVerse-dynamic" in injected


def test_cite_data_rejects_bad_fmt(tmp_path):
    from orchestrator import cite_data
    with pytest.raises(ValueError):
        cite_data(str(tmp_path / "x.ipynb"), fmt="html")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_orchestrator.py::test_cite_data_injects_apa_cell -v`
Expected: FAIL with `ImportError: cannot import name 'cite_data' from 'orchestrator'`.

- [ ] **Step 3: Write the implementation**

Append to `src/orchestrator.py`:

```python
def cite_data(
    notebook_path: str,
    targets=None,
    fmt: str = "apa",
    output_path: str | None = None,
) -> list[list[str]]:
    """
    Cites the datasets a notebook uses by injecting a retrieval cell per dataset.

    Detection is static (works even if the notebook was never run), but retrieval
    reuses the live kernel objects, so the citations appear as the injected
    cells' OUTPUT when the user runs them - not as this function's return value.

    Args:
        notebook_path: path to the .ipynb to analyze and modify
        targets: None (all detected datasets), a single variable name, or a list
        fmt: "apa" (default) or "bibtex"
        output_path: where to write the modified notebook (defaults to in place)

    Returns:
        the [variable, tool] pairs that had retrieval cells injected
    """
    _check_fmt(fmt)
    from data_workflow import generate_data_workflow

    return generate_data_workflow(
        notebook_path,
        variable=targets,
        output_path=output_path,
        fmt=fmt,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: PASS (both new tests plus the Task 3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.py tests/test_orchestrator.py
git commit -m "orchestrator: add cite_data tool"
```

---

## Task 5: LangChain StructuredTool wrappers

**Files:**
- Modify: `src/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `cite_software`, `cite_data`.
- Produces: `cite_software_tool` and `cite_data_tool` - `langchain_core.tools.StructuredTool` instances with `.name` and non-empty `.description`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_orchestrator.py`:

```python
def test_tools_are_structured_tools():
    from langchain_core.tools import StructuredTool
    from orchestrator import cite_software_tool, cite_data_tool
    assert isinstance(cite_software_tool, StructuredTool)
    assert isinstance(cite_data_tool, StructuredTool)


def test_tool_names_and_descriptions():
    from orchestrator import cite_software_tool, cite_data_tool
    assert cite_software_tool.name == "cite_software"
    assert cite_data_tool.name == "cite_data"
    assert "software" in cite_software_tool.description.lower()
    assert "dataset" in cite_data_tool.description.lower()


def test_cite_software_tool_invokes(monkeypatch):
    import bibliography
    monkeypatch.setattr(bibliography, "render_apa", lambda entries: "APA_SENTINEL")
    from orchestrator import cite_software_tool
    out = cite_software_tool.invoke(
        {"notebook_path": SAMPLE, "libraries": "pyleoclim", "fmt": "apa"}
    )
    assert out.startswith("APA_SENTINEL")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_orchestrator.py::test_tools_are_structured_tools -v`
Expected: FAIL with `ImportError: cannot import name 'cite_software_tool'`.

- [ ] **Step 3: Write the implementation**

Append to `src/orchestrator.py`:

```python
from langchain_core.tools import StructuredTool

cite_software_tool = StructuredTool.from_function(
    func=cite_software,
    name="cite_software",
    description=(
        "Cite the software libraries a Jupyter notebook imports. Use this for "
        "requests about citing software, packages, or libraries. Pass "
        "`notebook_path`; optionally `libraries` (a name or list to cite only "
        "those), `citation_types` ('paper' and/or 'software'), and `fmt` "
        "('apa' default, or 'bibtex')."
    ),
)

cite_data_tool = StructuredTool.from_function(
    func=cite_data,
    name="cite_data",
    description=(
        "Cite the datasets a Jupyter notebook uses (PyLiPD, PyleoTUPS, or "
        "LiPDGraph). Use this for requests about citing data or datasets. Pass "
        "`notebook_path`; optionally `targets` (a variable name or list to cite "
        "only those) and `fmt` ('apa' default, or 'bibtex'). This injects a "
        "retrieval cell per dataset; the user runs it to produce the citation."
    ),
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: PASS (all orchestrator tests).

- [ ] **Step 5: Run the full suite**

Run: `/opt/anaconda3/envs/lang/bin/python -m pytest tests/ -q`
Expected: PASS (all tests across the three test files).

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator.py tests/test_orchestrator.py
git commit -m "orchestrator: add LangChain StructuredTool wrappers"
```

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md` (local, gitignored - update the repo-structure block)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the repo structure**

In `CLAUDE.md`, in the `src/` tree under `## Repo Structure`, add this line after the `data_workflow.py` line:

```
│   ├── orchestrator.py        # Two tools (cite_software / cite_data) + LangChain wrappers; routes to each workflow
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add orchestrator.py to repo structure"
```

Note: `CLAUDE.md` is gitignored, so this commit may report nothing to commit. If so, skip it - the local file is still updated.

---

## Manual verification (after all tasks)

These need the `lang` env, a `GOOGLE_API_KEY` in `src/.env`, and network; they are not part of the automated suite.

- **Software APA end to end:** `cite_software("notebooks/sample.ipynb")` returns APA text for numpy/pandas/matplotlib/pyleoclim.
- **Data injection:** `cite_data("notebooks/testing/paleoPCAlite.ipynb", output_path="/tmp/pca_cited.ipynb")` returns `[["filtered_df2", "LiPDGraph"]]` and appends an APA-rendering cell. Open the copy, run the notebook through the injected cell in its kernel, and confirm APA citations print as cell output.
- **Filter to one dataset:** `cite_data(..., targets="filtered_df2")` injects a single cell.

---

## Self-review notes

- **Spec coverage:** two tools (Tasks 3-5), optional `str|list` target (Task 1 + `cite_software`/`cite_data` normalization), APA-default `fmt` (Task 2 + tool defaults), `filter_datasets` list change (Task 1), LangChain wrappers (Task 5), error handling for bad `fmt` and not-imported libraries (Tasks 3-4). The never-run contract and NL agent remain deferred (#24, later) as the spec states.
- **Type consistency:** `fmt` is `str` throughout; `libraries`/`targets` accept `str | list | None` and are normalized before use; `generate_data_workflow`'s new `fmt` param name matches its use in `cite_data`.
