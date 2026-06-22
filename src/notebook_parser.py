"""
Parses Jupyter Notebook (.ipynb) files to extract library imports and
dataset sources. Uses nbformat for notebook I/O and ast for code analysis.
Delegates dataset object detection to pylipd_helper, pyleotups_helper, and
lipdgraph_helper. Output is structured as actionable instructions for the
PaleoPAL Code Agent and SPARQL Agent.
"""

import ast
import nbformat
import warnings
from pylipd_helper import extract_lipd_objects
from pyleotups_helper import extract_pyleotups_objects
from lipdgraph_helper import detect_lipdgraph_queries

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


def _build_datasets(
    lipd_objects: dict[str, str],
    pyleotups_objects: dict[str, str],
    lipdgraph: dict | None,
) -> list[dict]:
    """
    Converts raw detection results into a unified list of dataset actions.

    Each entry tells the provenance agent which agent to call, which
    variable to reference, and what action to take to retrieve citations.

    Args:
        lipd_objects: {var_name: "LiPD"} from extract_lipd_objects()
        pyleotups_objects: {var_name: class_name} from extract_pyleotups_objects()
        lipdgraph: dict from detect_lipdgraph_queries(), or None

    Returns:
        list of dataset action dicts
    """
    datasets = []

    for var, cls in lipd_objects.items():
        datasets.append({
            "variable": var,
            "source_type": "PyLiPD",
            "agent": "code",
            "action": f"{var}.get_bibtex(remote=True)",
        })

    for var, cls in pyleotups_objects.items():
        datasets.append({
            "variable": var,
            "source_type": "PyleoTUPS",
            "class": cls,
            "agent": "code",
            "action": f"{var}.get_publications()",
        })

    if lipdgraph and lipdgraph.get("result_var"):
        datasets.append({
            "variable": lipdgraph["result_var"],
            "source_type": "LiPDGraph",
            "agent": "sparql",
            "endpoint": lipdgraph["endpoint"],
        })

    return datasets


def parse_notebook(path: str | None = None) -> dict:
    """
    Reads a .ipynb file and returns its imported libraries and dataset
    sources. Each dataset entry is an actionable instruction: which agent
    to call, which variable to reference, and what to do to get citations.

    Args:
        path: path to a .ipynb file, or None to auto-detect

    Returns:
        dict with:
            libraries: sorted list of imported library names
            datasets: list of dataset action dicts, each with:
                - variable: the object name in the notebook
                - source_type: "PyLiPD", "PyleoTUPS", or "LiPDGraph"
                - agent: "code" or "sparql"
                - action: the function call string (code agent)
                - endpoint: the SPARQL endpoint URL (sparql agent)
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
    all_cleaned_code = []

    for cell in nb.cells:
        if cell.cell_type == "code":
            cleaned = strip_ipython_directives(cell.source)
            libraries |= extract_libraries(cell.source)
            all_cleaned_code.append(cleaned)

    full_code = "\n".join(all_cleaned_code)

    datasets = _build_datasets(
        lipd_objects=extract_lipd_objects(full_code),
        pyleotups_objects=extract_pyleotups_objects(full_code),
        lipdgraph=detect_lipdgraph_queries(full_code),
    )

    return {
        "libraries": sorted(libraries),
        "datasets": datasets,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python notebook_parser.py <path_to_notebook.ipynb>")
        sys.exit(1)
    result = parse_notebook(sys.argv[1])
    print("Libraries:", result["libraries"])
    print("Datasets:")
    for ds in result["datasets"]:
        print(f"  [{ds['agent']}] {ds['variable']} ({ds['source_type']})"
              + (f" → {ds['action']}" if "action" in ds else ""))
