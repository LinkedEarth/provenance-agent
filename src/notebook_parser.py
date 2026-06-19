"""
Parses Jupyter Notebook (.ipynb) files to extract library imports and
dataset references. Uses nbformat for notebook I/O and ast for code
analysis. Delegates dataset extraction to pylipd_helper and pyleotups_helper.
"""

import ast
import nbformat
import warnings
from pylipd_helper import extract_datasets
from pyleotups_helper import extract_pyleotups_ids

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


def validate_libraries(requested: list[str], available: list[str]) -> tuple[list[str], list[str]]:
    """
    Checks which requested libraries are present in the notebook's
    import list. Comparison is case-insensitive.

    Args:
        requested: library names the user asked for
        available: library names returned by parse_notebook()

    Returns:
        tuple of (found, not_found) — libraries that were/weren't in
        the notebook
    """
    available_lower = {lib.lower() for lib in available}
    found = [lib for lib in requested if lib.lower() in available_lower]
    not_found = [lib for lib in requested if lib.lower() not in available_lower]
    return found, not_found


def parse_notebook(path: str | None = None) -> dict:
    """
    Reads a .ipynb file and returns its imported libraries and dataset
    references. Dataset details (LiPD names/dirs, PyleoTUPS PANGAEA/NOAA
    IDs) are kept in internal structures that the fetch functions in
    each helper know how to consume.

    Args:
        path: path to a .ipynb file, or None to auto-detect

    Returns:
        dict with:
            libraries: sorted list of imported library names
            _lipd: internal dict for pylipd_helper.fetch_lipd_citations()
            _pyleotups: internal dict for pyleotups_helper.fetch_pyleotups_citations()
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
    lipd_names = set()
    lipd_dirs = set()
    pyleotups_ids = {"pangaea": [], "noaa": []}

    for cell in nb.cells:
        if cell.cell_type == "code":
            cleaned = strip_ipython_directives(cell.source)
            libraries |= extract_libraries(cell.source)

            cell_datasets = extract_datasets(cleaned)
            lipd_names |= cell_datasets["names"]
            lipd_dirs |= cell_datasets["directories"]

            cell_ids = extract_pyleotups_ids(cleaned)
            pyleotups_ids["pangaea"].extend(cell_ids["pangaea"])
            pyleotups_ids["noaa"].extend(cell_ids["noaa"])

    pyleotups_ids["pangaea"] = sorted(set(pyleotups_ids["pangaea"]))
    pyleotups_ids["noaa"] = sorted(set(pyleotups_ids["noaa"]))

    return {
        "libraries": sorted(libraries),
        "_lipd": {"names": sorted(lipd_names), "directories": sorted(lipd_dirs)},
        "_pyleotups": pyleotups_ids,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python notebook_parser.py <path_to_notebook.ipynb>")
        sys.exit(1)
    result = parse_notebook(sys.argv[1])
    print("Libraries:", result["libraries"])
    print("LiPD:", result["_lipd"])
    print("PyleoTUPS:", result["_pyleotups"])
