"""
notebook_parser.py

Purpose:
    Parses Jupyter Notebook (.ipynb) files to extract structural and dependency
    information. Specifically, identifies the cell layout (code vs. markdown) and
    extracts all top-level Python library imports from code cells. This feeds into
    a broader provenance agent that generates bibliographies for data and software
    used in scientific notebooks.

Implementation:
    Uses nbformat to read .ipynb files as structured Python objects. Uses Python's
    built-in ast (Abstract Syntax Tree) module to parse code cells and identify
    import statements. AST parsing is preferred over regex because it correctly
    handles multi-line imports, aliased imports, and from-imports without false
    positives from comments or strings.

Design Decisions:
    - Only the top-level package name is extracted: "from matplotlib.pyplot import plt"
      yields "matplotlib", since that is the citable unit.
    - SyntaxError is silently skipped per cell so one malformed cell does not abort
      the entire parse.
    - parse_notebook() returns a dict so additional fields (e.g. data sources) can
      be added later without breaking callers.
    - extract_libraries() is kept separate so it can be tested or reused on arbitrary
      code strings independently of notebook I/O.
    - Non-Python cell magics (%%bash, %%writefile, %%html, etc.) cause the entire
      cell to be discarded — keeping their content would produce false-positive imports
      (e.g. an import inside a %%writefile block is written to disk, not executed).
    - Line magics (%time, %run, etc.) drop only the %magic_name prefix and preserve
      any trailing Python code so that `%time import pandas as pd` still yields pandas.
"""

import ast
import nbformat
import warnings

# Cell magics whose body is not Python — discard the entire cell to avoid
# false-positive imports (e.g. imports inside %%writefile are written to disk).
_NON_PYTHON_CELL_MAGICS = frozenset({
    "bash", "sh", "shell",
    "html", "javascript", "js", "svg", "latex", "markdown",
    "perl", "ruby",
    "writefile",
})


def strip_ipython_directives(code: str) -> str:
    """
    @brief: Cleans a notebook code cell so ast.parse() only sees valid Python.
            Non-Python cell magics (%%bash, %%writefile, etc.) cause the entire cell
            to be discarded. Python cell magic headers (%%time, %%timeit, etc.) are
            dropped but the rest of the cell is kept. Line magics (%time, %run, etc.)
            lose only the %magic_name prefix so any trailing Python code is preserved.
            Shell commands (!) are dropped entirely.
    @params[in]: code - raw source string from a notebook code cell
    @params[out]: cleaned string safe to pass to ast.parse(), or "" for non-Python cells
    """
    lines = code.splitlines()
    if not lines:
        return code

    first = lines[0].lstrip()
    if first.startswith("%%"):
        magic_name = first[2:].split()[0].lower() if first[2:].split() else ""
        if magic_name in _NON_PYTHON_CELL_MAGICS:
            # A non-Python cell magic means the ENTIRE cell body is interpreted by
            # that magic's handler (bash, writefile, etc.), not by Python. Dropping
            # only the %% line and keeping the body would cause ast.parse to see
            # non-Python content (e.g. shell commands, file contents) and either
            # raise SyntaxError or, worse, silently extract false-positive imports
            # from code that is never actually executed in this notebook.
            return ""

    cleaned = []
    for line in lines:
        s = line.lstrip()
        if s.startswith("%%"):
            continue  # Python cell magic header — drop just this line
        elif s.startswith("%"):
            rest = s[1:].split(None, 1)
            if len(rest) > 1:
                cleaned.append(rest[1])  # keep inline Python after %magic_name
        elif s.startswith("!"):
            continue
        else:
            cleaned.append(line)

    return "\n".join(cleaned)


def extract_libraries(code: str) -> set[str]:
    """
    @brief: Parses a Python source string and returns the set of top-level package
            names that are imported. Handles both `import X` and `from X import Y`
            forms. Strips IPython magic and shell commands before parsing. Silently
            skips cells that still fail to parse after stripping.
    @params[in]: code - a string of Python source code (typically one notebook cell)
    @params[out]: a set of strings, each a top-level package name (e.g. "matplotlib")
    """
    libraries = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(strip_ipython_directives(code))
    except SyntaxError:
        return libraries
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                libraries.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                libraries.add(node.module.split(".")[0])
    return libraries


def parse_notebook(path: str | None = None) -> dict:
    """
    @brief: Reads a Jupyter Notebook file and extracts its cell structure and all
            imported libraries across all code cells. Calls extract_libraries() on
            each code cell and aggregates the results. When called with no argument
            from inside a running notebook, auto-detects the current notebook path
            via ipynbname.
    @params[in]: path - file path string pointing to a .ipynb notebook file, or None
                        to auto-detect the currently running notebook
    @params[out]: a dict with two keys:
                    'libraries' - sorted list of unique top-level package names imported
                    'datasets'  - placeholder empty list (dataset extraction not yet implemented)
    """
    if path is None:
        try:
            import ipynbname  # optional runtime dep; not in base env, so Pylance flags it as unresolved
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
        if cell.cell_type == "code":
            libraries |= extract_libraries(cell.source)

    return {"libraries": sorted(libraries), "datasets": []}


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python notebook_parser.py <path_to_notebook.ipynb>")
        sys.exit(1)
    result = parse_notebook(sys.argv[1])
    print("Libraries:", result["libraries"])
