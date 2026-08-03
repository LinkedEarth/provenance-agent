"""
Structural scan enforcing the two import rules the migration established.

Purpose:
    Nothing in the repository may mutate sys.path to find this project, and
    nothing may import its modules by their retired flat names. Both rules are
    invisible to the rest of the suite: a stray `sys.path.insert` makes tests
    pass rather than fail, and a leftover `from bibliography import ...` in a
    file nobody imports sits there indefinitely. This module is what turns
    either into a red test.

Implementation:
    Every tracked Python file is parsed with `ast` and walked once.
    `_retired_import_violations` inspects Import/ImportFrom nodes;
    `_sys_path_violations` inspects Call/Assign/AugAssign nodes. Both report
    `path:line` strings so a failure names the offending line directly.

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
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCANNED_DIRECTORIES = ("src", "tests")

# Module names that no longer exist at the top level. Importing any of them
# without a package qualifier means the file predates the package migration.
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
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in RETIRED_MODULES:
                    violations.append(f"{label}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import inside the package; its module
            # name is package-internal and not the retired flat name.
            if node.level or not node.module:
                continue
            root = node.module.split(".")[0]
            if root in RETIRED_MODULES:
                violations.append(f"{label}:{node.lineno} from {node.module}")
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
        f"{label}:{node.lineno}"
        for node in ast.walk(ast.parse(source))
        if _mutates_sys_path(node)
    ]


def test_no_source_or_test_imports_a_retired_flat_module():
    violations = [
        violation
        for label, source in _python_sources()
        for violation in _retired_import_violations(label, source)
    ]
    assert violations == []


def test_no_source_or_test_mutates_sys_path():
    violations = [
        violation
        for label, source in _python_sources()
        for violation in _sys_path_violations(label, source)
    ]
    assert violations == []


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
