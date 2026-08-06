"""
Structural scan enforcing the project's two import rules.

Purpose:
    Nothing in the repository may mutate sys.path to find this project, and
    nothing may import its modules by their retired flat names. Both rules are
    invisible to the rest of the suite: a stray `sys.path.insert` makes tests
    pass rather than fail, and a leftover `from bibliography import ...` in a
    file nobody imports sits there indefinitely. This module is what turns
    either into a red test.

Implementation:
    Every tracked Python file and every notebook code cell is parsed with `ast`
    and walked once. `_retired_import_violations` inspects Import/ImportFrom
    nodes; `_sys_path_violations` inspects Call/Assign/AugAssign nodes. Both
    report `label:line` strings so a failure names the offending line directly.

Design decisions:
    - The scan matches on the module name the AST resolved, never on a
      substring of the source text. Two of the retired names are still live
      package-internal modules, so text matching produces false positives on
      correct code: `from .dataset_detection import ...` and
      `from provenance_agent.dataset_detection import ...` both contain a
      retired name and both are canonical.
    - Relative imports are skipped by testing `node.level > 0`, not by looking
      at the source. `from .dataset_detection import x` parses as
      ImportFrom(module="dataset_detection", level=1) - the leading dot lives in
      `level`, so the module name alone cannot distinguish it from the flat
      import this scan exists to reject.
    - Only the first dotted segment is compared, so `import agent.foo` is
      caught while `import provenance_agent.agent` is not.
    - `provenance` is deliberately absent from the retired set. It is the
      top-level `%load_ext provenance` shim and importing it is correct.
    - Only mutations of sys.path are flagged, not every mention of it. Reading
      sys.path is legitimate; rebinding, assigning into, or calling
      insert/append/extend on it is what this rule is about.
    - Notebooks are read through nbformat and scanned cell by cell, on the
      `source` of code cells only. Markdown and stored output are prose and are
      not scanned, so a historical explanation naming an old module is allowed
      to stay historical.
    - Cell sources go through `strip_ipython_directives` first and, when the
      cleaned cell still does not parse, through a line-by-line recovery pass.
      Both are necessary. A cell holding `%provenance cite the software` cleans
      to the bare words `cite the software`, which is not Python, and a research
      notebook routinely contains a cell that simply does not parse - without
      recovery, an import sitting in either cell would be invisible to this scan
      rather than caught by it.
"""

import ast
import warnings
from pathlib import Path

import nbformat

from provenance_agent.notebook_io import strip_ipython_directives

REPO_ROOT = Path(__file__).resolve().parents[1]

SCANNED_DIRECTORIES = ("src", "tests")
NOTEBOOK_DIRECTORY = "notebooks"

# Module names that do not exist at the top level. Importing any of them
# without a package qualifier means the file is reaching for a flat module
# rather than the installed package.
RETIRED_MODULES = frozenset({
    "notebook_parser",
    "bibliography",
    "software_workflow",
    "data_workflow",
    "dataset_detection",
    "deterministic_dataset_detection",
    "orchestrator",
    "agent",
    "llm",
})

_MUTATING_METHODS = frozenset({"insert", "append", "extend"})


def _python_sources() -> list[tuple[str, str]]:
    """
    Collects every Python file in the scanned directories.

    Returns:
        (repository-relative path, source text) pairs, sorted by path
    """
    sources = []
    for directory in SCANNED_DIRECTORIES:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            sources.append((str(path.relative_to(REPO_ROOT)), path.read_text()))
    return sources


def _notebook_sources() -> list[tuple[str, str]]:
    """
    Collects every notebook code cell, with IPython directives stripped.

    Markdown cells and stored outputs are deliberately excluded: they are prose,
    and a historical explanation naming a retired module is not a violation.

    Returns:
        ("path:cell N", cleaned source) pairs, sorted by path then cell index
    """
    sources = []
    for path in sorted((REPO_ROOT / NOTEBOOK_DIRECTORY).rglob("*.ipynb")):
        notebook = nbformat.read(str(path), as_version=4)
        label_path = path.relative_to(REPO_ROOT)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            sources.append(
                (f"{label_path}:cell {index}", strip_ipython_directives(cell.source))
            )
    return sources


def _all_sources() -> list[tuple[str, str]]:
    """Returns every scanned source: Python files first, then notebook cells."""
    return _python_sources() + _notebook_sources()


def _parse(source: str) -> ast.Module | None:
    """Parses source, returning None when it is not valid Python."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(source)
    except SyntaxError:
        return None


def _nodes_with_lines(source: str):
    """
    Yields (node, line number) for every AST node reachable in source.

    Source that does not parse as a whole is retried line by line, so an import
    sitting in a cell that also contains a bare magic argument or a genuine
    syntax error is still seen. Line numbers stay 1-based within the source.

    Args:
        source: Python source text, already stripped of IPython directives

    Yields:
        (ast.AST, int) pairs
    """
    tree = _parse(source)
    if tree is not None:
        for node in ast.walk(tree):
            yield node, getattr(node, "lineno", 0)
        return

    for offset, line in enumerate(source.splitlines(), start=1):
        line_tree = _parse(line.strip())
        if line_tree is None:
            continue
        for node in ast.walk(line_tree):
            yield node, offset


def _is_sys_path(node: ast.AST) -> bool:
    """Reports whether an AST node is the expression `sys.path`."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _mutates_sys_path(node: ast.AST) -> bool:
    """
    Reports whether a node mutates sys.path.

    Covers the three shapes that matter: calling a mutating list method on it,
    rebinding it, and assigning into or augmenting it. A plain read of
    sys.path is not a mutation and is not flagged.

    Args:
        node: any AST node

    Returns:
        True if the node changes what sys.path contains
    """
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATING_METHODS
            and _is_sys_path(node.func.value)
        )
    if isinstance(node, ast.AugAssign):
        return _is_sys_path(node.target)
    if isinstance(node, ast.Assign):
        return any(
            _is_sys_path(target)
            or (isinstance(target, ast.Subscript) and _is_sys_path(target.value))
            for target in node.targets
        )
    return False


def _retired_import_violations(label: str, source: str) -> list[str]:
    """
    Finds imports of retired top-level module names.

    Args:
        label: how to identify the source in a failure message
        source: Python source text

    Returns:
        "label:line name" strings, one per offending import
    """
    violations = []
    for node, lineno in _nodes_with_lines(source):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in RETIRED_MODULES:
                    violations.append(f"{label}:{lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import inside the package; its module
            # name is package-internal and not the retired flat name.
            if node.level or not node.module:
                continue
            root = node.module.split(".")[0]
            if root in RETIRED_MODULES:
                violations.append(f"{label}:{lineno} from {node.module}")
    return violations


def _sys_path_violations(label: str, source: str) -> list[str]:
    """
    Finds sys.path mutations.

    Args:
        label: how to identify the source in a failure message
        source: Python source text

    Returns:
        "label:line" strings, one per offending statement
    """
    return [
        f"{label}:{lineno}"
        for node, lineno in _nodes_with_lines(source)
        if _mutates_sys_path(node)
    ]


def test_nothing_imports_a_retired_flat_module():
    violations = [
        violation
        for label, source in _all_sources()
        for violation in _retired_import_violations(label, source)
    ]
    assert violations == []


def test_nothing_mutates_sys_path():
    violations = [
        violation
        for label, source in _all_sources()
        for violation in _sys_path_violations(label, source)
    ]
    assert violations == []


def test_the_scan_actually_reaches_the_notebooks():
    """Guards the guard: an empty source list would make both scans vacuous."""
    notebooks = _notebook_sources()
    assert len(notebooks) > 100
    assert any(label.startswith("notebooks/demos/") for label, _ in notebooks)
    assert any(label.startswith("notebooks/instructions/") for label, _ in notebooks)


# --- the scan itself ---------------------------------------------------------

def test_scan_flags_a_retired_flat_import():
    assert _retired_import_violations("x.py", "from bibliography import collect") == [
        "x.py:1 from bibliography"
    ]
    assert _retired_import_violations("x.py", "import orchestrator") == [
        "x.py:1 import orchestrator"
    ]


def test_scan_allows_the_canonical_and_relative_forms():
    """The two shapes a naive substring match would wrongly reject."""
    assert _retired_import_violations(
        "x.py", "from provenance_agent.dataset_detection import detect_datasets"
    ) == []
    assert _retired_import_violations("x.py", "from .dataset_detection import x") == []
    assert _retired_import_violations("x.py", "import provenance") == []


def test_scan_flags_sys_path_mutation_but_not_reading_it():
    assert _sys_path_violations("x.py", "import sys\nsys.path.insert(0, 'src')") == ["x.py:2"]
    assert _sys_path_violations("x.py", "import sys\nsys.path.append('src')") == ["x.py:2"]
    assert _sys_path_violations("x.py", "import sys\nsys.path += ['src']") == ["x.py:2"]
    assert _sys_path_violations("x.py", "import sys\nsys.path = ['src']") == ["x.py:2"]
    assert _sys_path_violations("x.py", "import sys\nprint(sys.path)") == []
    assert _sys_path_violations("x.py", "import sys\nprint(sys.argv)") == []


def test_scan_recovers_imports_from_a_cell_that_does_not_parse():
    """A notebook cell holding a bare magic argument still gets scanned."""
    cell = "cite the software\nfrom bibliography import collect\nx = 1"
    assert _parse(cell) is None
    assert _retired_import_violations("nb:cell 3", cell) == [
        "nb:cell 3:2 from bibliography"
    ]
    assert _sys_path_violations(
        "nb:cell 3", "cite the software\nimport sys\nsys.path.append('src')"
    ) == ["nb:cell 3:3"]
