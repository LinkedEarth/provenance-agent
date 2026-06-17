"""
Extracts library imports and LiPD dataset references from Jupyter Notebooks.

Notebooks contain code cells with Python source, but also IPython-specific
syntax (line magics like %matplotlib, shell commands like !pip, and cell
magics like %%bash) that would cause ast.parse() to fail. This module first
strips those directives, then walks the AST to collect import statements and
calls to LiPD load methods.

Libraries are identified by their top-level package name (e.g. "import
pandas as pd" yields "pandas"). Datasets are identified by finding calls
to PyLiPD methods (load, load_remote_datasets, load_from_dir) and resolving
their string arguments, including cases where the path is stored in a
variable.

The main entry point is parse_notebook(), which reads a .ipynb file via
nbformat, processes each code cell, and returns a dict with sorted lists
of library names and dataset names.
"""

import ast
import nbformat
import warnings

# Cell magics whose body is not Python — discard the entire cell.
_NON_PYTHON_CELL_MAGICS = frozenset({
    "bash", "sh", "shell",
    "html", "javascript", "js", "svg", "latex", "markdown",
    "perl", "ruby",
    "writefile",
})


def strip_ipython_directives(code: str) -> str:
    """
    Removes IPython-specific syntax from a code cell so ast.parse() can
    process it as valid Python.

    Handles three cases: cell magics (%%bash, %%html, etc.) that make
    the entire cell non-Python are discarded completely; line magics
    (%matplotlib inline) keep only the argument if present; and shell
    commands (!pip install) are removed entirely.

    Args:
        code: raw source string from a notebook code cell

    Returns:
        cleaned Python source string safe for ast.parse()
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


def extract_libraries(code: str) -> set[str]:
    """
    Extracts top-level package names from import statements in Python source.

    Parses the code with ast and walks the tree for Import and ImportFrom
    nodes. Only the root package is kept (e.g. "from matplotlib.pyplot"
    yields "matplotlib"). IPython directives are stripped before parsing.

    Args:
        code: Python source string (may contain IPython syntax)

    Returns:
        set of top-level package name strings
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


_LIPD_LOAD_METHODS = frozenset({"load", "load_remote_datasets", "load_from_dir"})


def _collect_string_variables(tree: ast.AST) -> dict[str, str]:
    """
    Collects simple variable assignments of the form name = 'string' from
    the AST. Used to resolve variable references in LiPD load calls
    (e.g. path = 'my_file.lpd'; lipd.load(path)).

    Args:
        tree: parsed AST of a Python source string

    Returns:
        dict mapping variable names to their string values
    """
    variables = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            variables[node.targets[0].id] = node.value.value
    return variables


def _resolve_to_strings(node: ast.AST, variables: dict[str, str]) -> list[str]:
    """
    Resolves an AST node to its string value(s). Handles three cases:
    string literals, lists of strings, and variable names that were
    previously assigned a string value.

    Args:
        node: AST node to resolve (e.g. a function argument)
        variables: variable-to-string mapping from _collect_string_variables()

    Returns:
        list of resolved string values (empty if the node can't be resolved)
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.List):
        result = []
        for elt in node.elts:
            result.extend(_resolve_to_strings(elt, variables))
        return result
    if isinstance(node, ast.Name) and node.id in variables:
        return [variables[node.id]]
    return []


def _normalize_dataset_name(raw: str) -> str:
    """
    Extracts a clean dataset name from a file path, URL, or raw string.
    Strips trailing slashes, takes the last path component, and removes
    the .lpd extension if present.

    Args:
        raw: file path, URL, or dataset name string

    Returns:
        normalized dataset name (e.g. "http://example.com/data.lpd" -> "data")
    """
    name = raw.rstrip("/")
    name = name.rsplit("/", 1)[-1] if "/" in name else name
    if name.endswith(".lpd"):
        name = name[:-4]
    return name


def extract_datasets(code: str) -> set[str]:
    """
    Extracts LiPD dataset names from calls to PyLiPD load methods
    (load, load_remote_datasets, load_from_dir).

    Parses the code with ast, finds method calls matching the known
    load methods, and resolves their first argument to string values
    (handling literals, lists, and variable references).

    Args:
        code: Python source string (may contain IPython syntax)

    Returns:
        set of normalized dataset name strings
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(strip_ipython_directives(code))
    except SyntaxError:
        return set()

    variables = _collect_string_variables(tree)
    datasets = set()

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LIPD_LOAD_METHODS
                and node.args):
            continue

        for s in _resolve_to_strings(node.args[0], variables):
            datasets.add(_normalize_dataset_name(s))

    return datasets


def parse_notebook(path: str | None = None) -> dict[str, list[str]]:
    """
    Main entry point. Reads a .ipynb file and extracts all imported
    libraries and LiPD dataset references from its code cells.

    If no path is given, attempts to auto-detect the current notebook
    path using ipynbname (must be called from inside a running notebook).

    Args:
        path: file path to a .ipynb notebook, or None for auto-detection

    Returns:
        dict with keys "libraries" and "datasets", each a sorted list
        of strings
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
    datasets = set()
    for cell in nb.cells:
        if cell.cell_type == "code":
            libraries |= extract_libraries(cell.source)
            datasets |= extract_datasets(cell.source)

    return {"libraries": sorted(libraries), "datasets": sorted(datasets)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python notebook_parser.py <path_to_notebook.ipynb>")
        sys.exit(1)
    result = parse_notebook(sys.argv[1])
    print("Libraries:", result["libraries"])
    print("Datasets:", result["datasets"])
