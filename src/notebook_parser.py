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
"""

import ast
import nbformat


def strip_ipython_directives(code: str) -> str:
    """
    @brief: Removes IPython-specific lines that are not valid Python syntax so that
            ast.parse() does not raise SyntaxError on cells containing them. Without
            this, any cell with a magic or shell command loses all its imports.
    @params[in]: code - raw source string from a notebook code cell
    @params[out]: the same string with lines beginning with % or ! removed
    """
    cleaned = "\n".join(
        line for line in code.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )
    return cleaned


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


def parse_notebook(path: str) -> dict:
    """
    @brief: Reads a Jupyter Notebook file and extracts its cell structure and all
            imported libraries across all code cells. Calls extract_libraries() on
            each code cell and aggregates the results.
    @params[in]: path - file path string pointing to a .ipynb notebook file
    @params[out]: a dict with two keys:
                    'cell_types' - list of 'code' or 'markdown' for each cell in order
                    'libraries'  - sorted list of unique top-level package names imported
    """
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    cell_types = [cell.cell_type for cell in nb.cells]
    libraries = set()
    for cell in nb.cells:
        if cell.cell_type == "code":
            libraries |= extract_libraries(cell.source)

    return {"cell_types": cell_types, "libraries": sorted(libraries)}
