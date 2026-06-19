"""
Extracts PyleoTUPS study IDs from notebook code and fetches their citations.

Uses AST parsing to find search_studies() calls and pull out study IDs from
keyword arguments. The keyword name disambiguates the data source internally:
  - study_ids → PANGAEA
  - noaa_id or xml_id → NOAA

The caller doesn't need to know which source a study ID came from — the
fetch function handles routing.
"""

import ast
import warnings


_PANGAEA_KWARGS = frozenset({"study_ids"})
_NOAA_KWARGS = frozenset({"noaa_id", "xml_id"})


def _collect_int_variables(tree: ast.AST) -> dict[str, int]:
    """Tracks simple name = integer assignments for variable resolution."""
    variables = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)):
            variables[node.targets[0].id] = node.value.value
    return variables


def _resolve_to_ints(node: ast.AST, variables: dict[str, int]) -> list[int]:
    """Resolves an AST node (constant, list, or variable) to integer values."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return [node.value]
    if isinstance(node, ast.List):
        result = []
        for elt in node.elts:
            result.extend(_resolve_to_ints(elt, variables))
        return result
    if isinstance(node, ast.Name) and node.id in variables:
        return [variables[node.id]]
    return []


def extract_pyleotups_ids(code: str) -> dict[str, list[int]]:
    """
    Extracts PyleoTUPS study IDs from search_studies() calls.
    Internal use — returns PANGAEA/NOAA split for the fetch function.

    Args:
        code: pre-cleaned Python source string

    Returns:
        dict with "pangaea" and "noaa" keys mapping to lists of study IDs
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return {"pangaea": [], "noaa": []}

    int_vars = _collect_int_variables(tree)
    pangaea_ids = []
    noaa_ids = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "search_studies"):
            continue

        for kw in node.keywords:
            if kw.arg in _PANGAEA_KWARGS:
                pangaea_ids.extend(_resolve_to_ints(kw.value, int_vars))
            elif kw.arg in _NOAA_KWARGS:
                noaa_ids.extend(_resolve_to_ints(kw.value, int_vars))

    return {"pangaea": pangaea_ids, "noaa": noaa_ids}


def fetch_pyleotups_citations(ids: dict[str, list[int]]) -> list[str]:
    """
    Fetches BibTeX citations for PyleoTUPS study IDs.

    Routes PANGAEA IDs to PangaeaDataset and NOAA IDs to NOAADataset,
    calls get_publications() on each, and returns combined BibTeX strings.

    Args:
        ids: dict with "pangaea" and "noaa" keys from extract_pyleotups_ids()

    Returns:
        list of BibTeX entry strings
    """
    import pyleotups as pt

    all_bibtex = []

    if ids.get("pangaea"):
        ds = pt.PangaeaDataset()
        ds.search_studies(study_ids=ids["pangaea"])
        bib_data, _ = ds.get_publications()
        all_bibtex.append(bib_data.to_string(bib_format="bibtex"))

    if ids.get("noaa"):
        for noaa_id in ids["noaa"]:
            ds = pt.NOAADataset()
            ds.search_studies(noaa_id=noaa_id)
            bib_data, _ = ds.get_publications()
            all_bibtex.append(bib_data.to_string(bib_format="bibtex"))

    return all_bibtex
