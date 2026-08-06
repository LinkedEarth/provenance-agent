"""
Software side of the provenance agent: reads Jupyter Notebook (.ipynb) files
and extracts the Python libraries they import.

Purpose:
    The software workflow needs the set of imported libraries so it can look up
    their citations. This module does that extraction with `ast`, plus it
    provides `read_notebook_code()`, the raw-code reader used by the data
    workflow for code-oriented checks such as endpoint extraction.

Implementation:
    - strip_ipython_directives(code): removes magics (`%`, `%%`) and shell
      lines (`!`) so `ast.parse()` only sees valid Python; whole-cell magics
      whose body is not Python (e.g. `%%bash`) are dropped entirely.
    - extract_libraries(code): walks the AST and collects top-level package
      names from `import` / `from ... import` statements. Cells with syntax
      errors fall back to line-by-line import recovery, since a broken cell's
      imports are still real dependencies.
    - parse_notebook(path): reads a notebook and returns its sorted list of
      imported library names.
    - read_notebook_code(path): returns all code cells concatenated (directives
      stripped) for code-oriented workflow checks.
    - validate_libraries(requested, available): case-insensitive membership
      check used by the "cite one specific library" mode.
    - is_generated_cell(source): True for cells this tool injected. Both scans
      above skip them.

Design decisions:
    - Injected cells are excluded from both scans. They carry imports of their
      own (the software cell imports provenance_agent.citations, a LiPDGraph
      retrieval cell imports pylipd), which are the tool's machinery rather than
      the notebook's dependencies - and since pylipd has a citation on file,
      scanning them would make a second run cite a library the notebook never
      used. Injected cells carry PROVENANCE_CELL_MARKER; the signatures earlier
      versions used are matched too, so a notebook written by one of those needs
      no re-run.
    - Detection of *datasets* is NOT done here. It is delegated to the
      deterministic detector exposed by dataset_detection.py. This module is
      otherwise purely the software (import) side.
"""

import ast
import re
import warnings

import nbformat

# Cell magics whose body is not Python - discard the entire cell.
_NON_PYTHON_CELL_MAGICS = frozenset({
    "bash", "sh", "shell",
    "html", "javascript", "js", "svg", "latex", "markdown",
    "perl", "ruby",
    "writefile",
})


PROVENANCE_CELL_MARKER = "# provenance-agent-generated"

# Signatures of cells injected before PROVENANCE_CELL_MARKER existed, so a
# notebook written by an older version is still recognized without re-running.
_LEGACY_GENERATED_SIGNATURES = (
    "_provbib_software",
    "_provbib_data_",
    "# provenance-combine-cell",
)


def is_generated_cell(source: str) -> bool:
    """
    Reports whether a cell was injected by this tool rather than written by the user.

    The injected cells carry imports of their own - the software cell imports
    provenance_agent.citations, a LiPDGraph retrieval cell imports pylipd - and
    those are the tool's own machinery, not the notebook's dependencies.
    Scanning them would make a second run cite pylipd, which has a citation on
    file, so the notebook would gain a citation for a library it never used.

    Args:
        source: a notebook cell's source text

    Returns:
        True if the cell was injected by a provenance workflow
    """
    return (
        PROVENANCE_CELL_MARKER in source
        or any(sig in source for sig in _LEGACY_GENERATED_SIGNATURES)
    )


def strip_ipython_directives(code: str) -> str:
    """
    Cleans a code cell so ast.parse() only sees valid Python.

    Notebook cells routinely contain IPython syntax that is not valid Python:
    line magics (`%matplotlib inline`), cell magics (`%%time`), and shell
    escapes (`!pip install ...`). Without stripping them, ast.parse() raises on
    the whole cell and its real imports would be lost.

    A line magic keeps its argument, because for magics like `%time x = f()`
    the remainder is genuine Python. That means a magic whose argument is prose,
    such as `%provenance cite the software`, leaves the bare words behind and
    the cell still will not parse - which is why extract_libraries also has a
    line-by-line recovery pass.

    Args:
        code: a notebook code cell's source text

    Returns:
        the source with directives removed, or "" when the whole cell is a
        non-Python cell magic such as %%bash
    """
    lines = code.splitlines()
    if not lines:
        return code

    first = lines[0].lstrip()
    if first.startswith("%%"):
        magic_name = first[2:].split()[0].lower() if first[2:].split() else ""
        if magic_name in _NON_PYTHON_CELL_MAGICS:
            return ""

    cleaned = []
    for line in lines:
        s = line.lstrip()
        if s.startswith("%%"):
            continue
        elif s.startswith("%"):
            rest = s[1:].split(None, 1)
            if len(rest) > 1:
                cleaned.append(rest[1])
        elif s.startswith("!"):
            continue
        else:
            cleaned.append(line)

    return "\n".join(cleaned)


def _libraries_from_tree(tree: ast.AST) -> set[str]:
    """Collects top-level package names from Import/ImportFrom nodes in an AST."""
    libraries = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                libraries.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                libraries.add(node.module.split(".")[0])
    return libraries


def _recover_imports_linewise(code: str) -> set[str]:
    """
    Salvages imports from source that does not parse as a whole.

    Research notebooks routinely contain cells with syntax errors (e.g. a
    function whose docstring and body disagree on indentation); their imports
    are still real dependencies. Each line that looks like an import statement
    is parsed on its own; lines that still fail (e.g. an open parenthesis in
    `from x import (`) fall back to a regex for the leading module name.

    Args:
        code: Python source that raised SyntaxError when parsed whole

    Returns:
        the top-level package names recovered from import-like lines
    """
    libraries = set()
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("import ", "from ")):
            continue
        try:
            libraries |= _libraries_from_tree(ast.parse(stripped))
        except SyntaxError:
            match = re.match(r"(?:import|from)\s+([A-Za-z_][\w.]*)", stripped)
            if match:
                libraries.add(match.group(1).split(".")[0])
    return libraries


def extract_libraries(code: str) -> set[str]:
    """
    Extracts top-level package names imported in a Python source string.

    Only the top-level package is kept, so `import matplotlib.pyplot as plt`
    and `from matplotlib.pyplot import plot` both yield "matplotlib". Imports
    anywhere in the source count, including inside a function body.

    Falls back to line-by-line import recovery when the source has a syntax
    error, so a broken cell still contributes its imports.

    Args:
        code: Python source, typically one notebook code cell

    Returns:
        the set of top-level package names imported, empty if there are none
        or if nothing could be recovered from unparseable source
    """
    cleaned = strip_ipython_directives(code)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(cleaned)
    except SyntaxError:
        return _recover_imports_linewise(cleaned)
    return _libraries_from_tree(tree)


def validate_libraries(requested: list[str], available: list[str]) -> tuple[list[str], list[str]]:
    """
    Checks which requested libraries are present in the notebook's
    import list. Comparison is case-insensitive.

    Args:
        requested: library names the user asked for
        available: library names returned by parse_notebook()

    Returns:
        tuple of (found, not_found) - libraries that were/weren't in
        the notebook
    """
    available_lower = {lib.lower() for lib in available}
    found = [lib for lib in requested if lib.lower() in available_lower]
    not_found = [lib for lib in requested if lib.lower() not in available_lower]
    return found, not_found


def read_notebook_code(path: str) -> str:
    """
    Reads a .ipynb file and returns all code cells concatenated into one string,
    with IPython directives (magics, shell lines) stripped so the result is
    valid Python. Used by code-oriented workflow checks such as endpoint
    extraction.

    Cells this tool injected are skipped, so a caller never sees the retrieval
    scaffolding (_lipd_x = LiPD(), _meta_x = ...) and mistakes it for one of
    the notebook's own dataset variables.

    Args:
        path: path to a .ipynb file

    Returns:
        the notebook's code cells joined by newlines, directives removed
    """
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    return "\n".join(
        strip_ipython_directives(cell.source)
        for cell in nb.cells
        if cell.cell_type == "code" and not is_generated_cell(cell.source)
    )


def parse_notebook(path: str | None = None) -> list[str]:
    """
    Reads a .ipynb file and returns the sorted list of libraries it imports.

    Args:
        path: path to a .ipynb file, or None to auto-detect the current
            notebook (requires ipynbname and a running kernel)

    Returns:
        sorted list of imported top-level library names
    """
    if path is None:
        try:
            import ipynbname
            path = str(ipynbname.path())
        except Exception:
            raise RuntimeError(
                "Could not auto-detect the current notebook path. "
                "Install ipynbname (`pip install ipynbname`) and call from inside a running notebook, "
                "or pass an explicit path to parse_notebook()."
            )
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    libraries = set()
    for cell in nb.cells:
        if cell.cell_type == "code" and not is_generated_cell(cell.source):
            libraries |= extract_libraries(cell.source)

    return sorted(libraries)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m provenance_agent.notebook_io <path_to_notebook.ipynb>")
        sys.exit(1)
    print("Libraries:", parse_notebook(sys.argv[1]))
