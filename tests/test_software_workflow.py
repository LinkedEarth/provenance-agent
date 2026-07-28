"""
test_software_workflow.py

Purpose:
    Unit tests for the pure logic in software_workflow.py: generating the
    citation-metadata cell source (build_metadata_cell), injecting it into a
    notebook (inject_metadata_cell), and the end-to-end glue that filters the
    imports and writes the notebook back (generate_software_workflow).

Implementation:
    build_metadata_cell is a pure string function. inject_metadata_cell operates
    on an in-memory nbformat NotebookNode, so there is no file I/O. The end-to-end
    test runs fully offline: it reads a real fixture notebook (sample.ipynb) with
    nbformat and writes to a tmp path, so no Gemini or network call happens and the
    fixture is never mutated. The live-kernel execution (the user running the
    injected cell) is out of scope for these tests.

Design Decisions:
    - The injected cell imports collect_library_entries rather than baking BibTeX
      inline, binds the result to provenance_software and display()s it, so the
      tests assert on that import, the binding, and the display call. There is
      no combine cell: each workflow owns exactly one self-displaying cell, and
      a re-run replaces its own cell rather than appending a second.
    - generate_software_workflow returns the libraries it built a cell for, so a
      filter that matches nothing returns [] and leaves the notebook untouched.
"""

import os
import sys

import nbformat
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from software_workflow import (
    build_metadata_cell,
    generate_software_workflow,
    inject_metadata_cell,
)

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sample.ipynb")


# --- build_metadata_cell -----------------------------------------------------

def test_cell_imports_collect_library_entries():
    cell = build_metadata_cell(["pyleoclim", "pandas"])
    assert "from bibliography import collect_library_entries" in cell


def test_cell_bakes_in_the_library_list():
    cell = build_metadata_cell(["pyleoclim", "pandas"])
    assert "collect_library_entries(['pyleoclim', 'pandas'], None)" in cell


def test_cell_passes_citation_types_filter():
    cell = build_metadata_cell(["pyleoclim"], citation_types=["software"])
    assert "['pyleoclim'], ['software']" in cell


def test_cell_binds_and_displays_provenance_software():
    cell = build_metadata_cell(["pyleoclim"])
    assert "provenance_software = collect_library_entries(['pyleoclim'], None)" in cell
    assert "display(provenance_software)" in cell


# --- inject_metadata_cell ----------------------------------------------------

def test_inject_appends_a_single_code_cell():
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("import pyleoclim"))
    inject_metadata_cell(nb, ["pyleoclim"])
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 2  # original + 1 injected
    assert "collect_library_entries(['pyleoclim'], None)" in code_cells[-1].source


# --- generate_software_workflow ----------------------------------------------

def test_generate_all_libraries_injects_and_returns_them(tmp_path):
    out = tmp_path / "out.ipynb"
    wanted = generate_software_workflow(SAMPLE, output_path=str(out))
    assert "pyleoclim" in wanted

    nb = nbformat.read(str(out), as_version=4)
    assert "# provenance-combine-cell" not in "".join(c.source for c in nb.cells)
    assert "display(provenance_software)" in nb.cells[-1].source
    assert any("collect_library_entries(" in c.source for c in nb.cells)


def test_generate_one_library_filters(tmp_path):
    out = tmp_path / "out.ipynb"
    wanted = generate_software_workflow(SAMPLE, libraries="pyleoclim", output_path=str(out))
    assert wanted == ["pyleoclim"]

    nb = nbformat.read(str(out), as_version=4)
    assert any("['pyleoclim']" in c.source for c in nb.cells)


def test_generate_unimported_library_returns_empty_and_leaves_notebook(tmp_path):
    out = tmp_path / "out.ipynb"
    wanted = generate_software_workflow(
        SAMPLE, libraries=["definitely_not_here"], output_path=str(out)
    )
    assert wanted == []
    assert not out.exists()  # nothing matched, so nothing was written


def test_generate_does_not_mutate_the_source_notebook(tmp_path):
    before = nbformat.read(SAMPLE, as_version=4)
    generate_software_workflow(SAMPLE, output_path=str(tmp_path / "out.ipynb"))
    after = nbformat.read(SAMPLE, as_version=4)
    assert len(after.cells) == len(before.cells)


def test_generate_excludes_stdlib_from_cell_and_result(tmp_path):
    """A notebook importing sys/json must not cite them as dependencies."""
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        "import sys\nimport json\nimport pathlib\nimport numpy as np"
    ))
    source = tmp_path / "in.ipynb"
    with open(source, "w") as f:
        nbformat.write(nb, f)

    wanted = generate_software_workflow(str(source))

    assert wanted == ["numpy"]
    written = nbformat.read(str(source), as_version=4)
    metadata_cell = next(c for c in written.cells if "provenance_software" in c.source)
    for stdlib_name in ("'sys'", "'json'", "'pathlib'"):
        assert stdlib_name not in metadata_cell.source


def test_repeated_runs_replace_the_software_cell(tmp_path):
    """One software cell means one, however many times the workflow runs."""
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("import numpy as np"))
    path = tmp_path / "nb.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    for _ in range(3):
        generate_software_workflow(str(path))

    written = nbformat.read(str(path), as_version=4)
    assert sum("provenance_software" in c.source for c in written.cells) == 1
    assert len(written.cells) == 2  # the original cell plus one injected
