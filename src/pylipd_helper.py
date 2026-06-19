"""
Extracts LiPD dataset names from notebook code and fetches their citations.

Uses AST parsing to find calls to LiPD load methods (load,
load_remote_datasets, load_from_dir) and resolves their arguments
to dataset names. Citation fetching loads datasets from the LiPDVerse
endpoint and calls get_bibtex() to retrieve BibTeX strings.
"""

import ast
import warnings


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


def extract_datasets(code: str) -> dict[str, set[str]]:
    """
    Extracts LiPD dataset references from calls to LiPD load methods.

    Separates named datasets (from load/load_remote_datasets) from
    directory paths (from load_from_dir) since they need different
    loading strategies for citation fetching.

    Args:
        code: pre-cleaned Python source string (strip_ipython_directives
              should be called by the caller before passing code here)

    Returns:
        dict with "names" (normalized dataset names) and "directories"
        (raw directory paths)
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return {"names": set(), "directories": set()}

    variables = _collect_string_variables(tree)
    names = set()
    directories = set()

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LIPD_LOAD_METHODS
                and node.args):
            continue

        method = node.func.attr
        for s in _resolve_to_strings(node.args[0], variables):
            if method == "load_from_dir":
                directories.add(s)
            else:
                names.add(_normalize_dataset_name(s))

    return {"names": names, "directories": directories}


_LIPDVERSE_ENDPOINT = "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic"


def fetch_lipd_citations(
    dataset_names: list[str] | None = None,
    directories: list[str] | None = None,
) -> tuple[list[str], "pandas.DataFrame"]:
    """
    Loads LiPD datasets and retrieves their BibTeX citations.

    Named datasets are loaded from LiPDVerse. Directories are loaded
    locally via load_from_dir. Both are merged into a single LiPD object
    before calling get_bibtex().

    Args:
        dataset_names: list of LiPD dataset name strings
            (e.g. ["Ocn-RedSea.Felis.2000"])
        directories: list of local directory paths containing .lpd files
            (e.g. ["Pages2k/"])

    Returns:
        tuple of (bibtex_strings, dataframe) from PyLiPD's get_bibtex().
        bibtex_strings is a list of BibTeX entry strings.
        dataframe has columns: dsname, title, authors, doi, year, journal, etc.
    """
    from pylipd.lipd import LiPD

    D = LiPD()

    if dataset_names:
        D.set_endpoint(_LIPDVERSE_ENDPOINT)
        D.load_remote_datasets(dataset_names)

    if directories:
        for dir_path in directories:
            D_dir = LiPD()
            D_dir.load_from_dir(dir_path)
            for ds in D_dir.get_datasets():
                D.load_datasets([ds])

    return D.get_bibtex(remote=True)
