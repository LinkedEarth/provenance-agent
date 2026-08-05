"""
Structural checks on the notebook tree itself.

Purpose:
    The notebooks were moved and rewritten wholesale, and the two ways that
    goes wrong are silent. A notebook can be left structurally invalid by an
    edit that a JSON dump still accepts, and a relative fixture path can be
    left pointing at where a file used to be. Neither shows up in any other
    test, because nothing else in the suite opens most of these notebooks.

Implementation:
    `test_every_notebook_is_valid_and_round_trips` runs nbformat.validate over
    every .ipynb and then writes it back through nbformat to confirm the file
    survives a read/write cycle unchanged. `test_relative_data_paths_resolve`
    walks code-cell ASTs for string constants naming a local .lpd or .bib file
    and resolves each against its own notebook's directory.

Design decisions:
    - No cell is ever executed. Running these notebooks means Gemini calls,
      SPARQL queries against LiPDGraph, and remote dataset downloads. Every
      check here is static.
    - Round-trip equality is asserted against a re-serialization rather than
      the bytes on disk, so a notebook saved by Jupyter with different (but
      valid) formatting is not reported as broken.
    - Only .lpd and .bib literals are resolved. Notebook path literals are
      excluded on purpose: several are *output* paths for throwaway demo
      copies that do not exist until a demo runs, so requiring them to resolve
      would fail on correct notebooks.
    - Path literals containing "://" are skipped as remote URLs.
    - The instruction bundles load their data by bare sibling name, e.g.
      `lipd.load('Vostok.Bazin.2013.lpd')`. That is exactly what this test
      resolves relative to the notebook, so it is what proves each bundle
      moved intact rather than being flattened.
"""

import ast
import warnings
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").rglob("*.ipynb"))

LOCAL_DATA_SUFFIXES = (".lpd", ".bib")


def _notebook_id(path: Path) -> str:
    """Names a notebook by its repository-relative path in test output."""
    return str(path.relative_to(REPO_ROOT))


def test_the_notebook_tree_is_not_empty():
    """
    Guards the guard: an empty glob would make every check below vacuous.

    The count is a floor rather than the exact size, so adding a notebook does
    not fail the suite. It dropped when the four workflow demos were merged into
    one and the scratch notebooks under exploration/ were removed; fixtures/ no
    longer holds notebooks either, because the test inputs are now built in code
    by tests/notebook_fixtures.py.
    """
    assert len(NOTEBOOKS) > 10
    directories = {path.relative_to(REPO_ROOT / "notebooks").parts[0] for path in NOTEBOOKS}
    assert directories == {"demos", "examples", "instructions"}


def test_no_notebook_is_left_outside_the_new_layout():
    """The pre-move locations, including notebooks/testing/, are gone."""
    assert not (REPO_ROOT / "notebooks" / "testing").exists()
    assert [path for path in NOTEBOOKS if path.parent == REPO_ROOT / "notebooks"] == []


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=_notebook_id)
def test_every_notebook_is_valid_and_round_trips(notebook_path, tmp_path):
    notebook = nbformat.read(str(notebook_path), as_version=4)
    nbformat.validate(notebook)

    copy = tmp_path / notebook_path.name
    nbformat.write(notebook, str(copy))
    assert nbformat.read(str(copy), as_version=4) == notebook


def _local_data_paths(notebook) -> list[str]:
    """
    Collects string constants in code cells that name a local data file.

    Cells that do not parse are skipped rather than recovered: a path literal
    is only interesting when the cell around it is real code, and the
    line-by-line recovery that import scanning needs would not see string
    constants inside a call anyway.

    Args:
        notebook: an nbformat notebook node

    Returns:
        the .lpd/.bib path literals found, excluding remote URLs
    """
    found = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.endswith(LOCAL_DATA_SUFFIXES)
                and "://" not in node.value
            ):
                found.append(node.value)
    return found


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=_notebook_id)
def test_relative_data_paths_resolve(notebook_path):
    notebook = nbformat.read(str(notebook_path), as_version=4)
    unresolved = [
        literal
        for literal in _local_data_paths(notebook)
        if not (notebook_path.parent / literal).exists()
    ]
    assert unresolved == []


def test_the_data_path_check_covers_the_instruction_bundles():
    """Every bundle must still load its own .lpd, or the check above is empty."""
    bundles = sorted((REPO_ROOT / "notebooks" / "instructions").glob("Notebook*"))
    assert len(bundles) == 4
    for bundle in bundles:
        notebooks = list(bundle.glob("*.ipynb"))
        assert len(notebooks) == 1, bundle
        notebook = nbformat.read(str(notebooks[0]), as_version=4)
        assert _local_data_paths(notebook), bundle
