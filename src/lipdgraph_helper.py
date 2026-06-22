"""
Detects LiPDGraph usage in notebook code via SPARQL query detection.

LiPDGraph is a graph database queried directly via SPARQL using
requests.post() to the LinkedEarth endpoint. Unlike PyLiPD and PyleoTUPS
which use Python library constructors, LiPDGraph data access is identified
by the presence of the endpoint URL and SPARQL query strings containing
LinkedEarth ontology prefixes.

When detected, the provenance agent can send a bibliography SPARQL query
to the same endpoint to retrieve dataset citations.
"""

import ast
import warnings


_LINKEDEARTH_ENDPOINT_MARKER = "linkedearth.graphdb.mint.isi.edu"


def detect_lipdgraph_queries(code: str) -> dict | None:
    """
    Detects SPARQL queries to the LinkedEarth graph database.

    Scans string constants for the LinkedEarth endpoint URL and SPARQL
    query patterns (PREFIX le: <http://linked.earth/ontology#>). Returns
    the endpoint URL and the variable name holding the query result
    DataFrame if found.

    Expects the full concatenated notebook code (all cells).

    Args:
        code: full pre-cleaned Python source from all notebook cells

    Returns:
        dict with "endpoint" (the URL string) and "result_var" (the variable
        name assigned to pd.read_csv() of the response), or None if no
        LiPDGraph usage detected
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return None

    endpoint_url = None
    result_var = None

    for node in tree.body:
        # url = 'https://linkedearth.graphdb.mint.isi.edu/...'
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and _LINKEDEARTH_ENDPOINT_MARKER in node.value.value):
            endpoint_url = node.value.value

        # df_res = pd.read_csv(...) — the DataFrame holding query results
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "read_csv"):
            result_var = node.targets[0].id

    if endpoint_url is None:
        return None

    return {
        "endpoint": endpoint_url,
        "result_var": result_var,
    }
