"""
Parses Jupyter Notebook (.ipynb) files to extract library imports and LiPD
dataset references. Uses nbformat for notebook I/O and ast for code analysis.
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
    """Cleans a code cell so ast.parse() only sees valid Python."""
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
    """Extracts top-level package names imported in a Python source string."""
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
    """Tracks simple name = 'string' assignments for variable resolution."""
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
    """Resolves an AST node (constant, list, or variable) to string values."""
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
    """Extracts a dataset name from a file path, URL, or raw name."""
    name = raw.rstrip("/")
    name = name.rsplit("/", 1)[-1] if "/" in name else name
    if name.endswith(".lpd"):
        name = name[:-4]
    return name


def extract_datasets(code: str) -> set[str]:
    """Extracts LiPD dataset names from calls to LiPD load methods."""
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


def parse_notebook(path: str | None = None) -> dict:
    """Reads a .ipynb file and returns its imported libraries and datasets."""
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
