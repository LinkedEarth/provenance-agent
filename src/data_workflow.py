"""
Data workflow orchestrator: turns detected dataset variables into dataset
citations by injecting retrieval cells into the notebook's live kernel.

Purpose:
    Given the [variable, tool] pairs from dataset_detection, generate a code
    cell per dataset that retrieves its BibTeX, and append those cells to the
    notebook with nbformat. The cells are meant to run in the notebook's own
    kernel, where the already-loaded objects (LiPD objects, PyleoTUPS datasets,
    the LiPDGraph result DataFrame) are reused - no re-querying or re-loading.
    The fmt parameter controls whether cells output raw BibTeX or render it to
    APA format.

Implementation:
    - extract_lipdgraph_endpoint(code): AST-scans the notebook for the
      LinkedEarth graph endpoint URL (the string passed to requests.post), so
      the LiPDGraph pathway loads from the same repository the notebook queried.
    - build_retrieval_cell(variable, tool, endpoint, fmt): returns the Python
      source for one dataset's retrieval cell. PyLiPD/PyleoTUPS call the library
      method directly on the in-memory object (Approach C: {var}.{method}).
      LiPDGraph is special: the terminal variable is a DataFrame, so the cell
      pulls its dataSetName column, loads those datasets into a fresh LiPD object
      from the endpoint, then calls get_bibtex(). When fmt="apa", the cell pipes
      the collected BibTeX to bibliography.render_bibtex_strings_to_apa() for
      APA rendering in-kernel. Every cell also display()s the metadata DataFrame
      that get_bibtex()/get_publications() return alongside the BibTeX.
    - filter_datasets(pairs, tool, variable): narrows the detected pairs so the
      workflow can cite all datasets, only one tool's datasets, or one variable.
    - inject_retrieval_cells(nb, pairs, endpoint, fmt): appends one retrieval code
      cell per pair to an nbformat notebook node. fmt defaults to "bibtex" and
      accepts "apa" to render citations in APA format.
    - generate_data_workflow(..., fmt): top-level glue - detect, filter, inject,
      write. fmt defaults to "bibtex" and can be "apa" for APA-formatted output.

Design decisions:
    - Cells are written for the user to run (live-kernel model); this module does
      not execute them. Collecting the printed BibTeX happens at notebook runtime.
    - The LiPDGraph endpoint is lifted from the notebook via AST rather than
      hardcoded, so a notebook pointed at a different repository is handled
      correctly. _LIPDVERSE_ENDPOINT is only a fallback when no URL is found.
    - Unsupported tools raise ValueError so a mis-detected pair fails loudly
      rather than silently producing an empty bibliography.
    - APA rendering happens in the injected cell (via render_bibtex_strings_to_apa)
      so the user can see formatted citations as output without re-running code.
    - Both PyLiPD's get_bibtex() and PyleoTUPS' get_publications() return
      (citations, metadata DataFrame). The cell binds that DataFrame to
      _meta_{variable} and displays it, so the per-dataset metadata behind each
      citation is visible instead of being dropped on the floor.
"""

import ast
import warnings

import nbformat


_LIPDVERSE_ENDPOINT = "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic"

# Every LinkedEarth GraphDB query endpoint has this prefix, e.g.
# .../repositories/LiPDVerse-dynamic. Matching the full prefix picks up the
# query endpoint set_endpoint() needs and skips the bare host URL.
_LIPDGRAPH_ENDPOINT_PREFIX = "https://linkedearth.graphdb.mint.isi.edu/repositories/"


def extract_lipdgraph_endpoint(code: str) -> str | None:
    """
    Lifts the LinkedEarth graph endpoint URL from notebook code via AST.

    Scans string constants for one starting with the GraphDB query-endpoint
    prefix (.../repositories/) and returns it, so the LiPDGraph retrieval cell
    loads from the same repository the notebook queried.

    Args:
        code: notebook Python source (all code cells concatenated)

    Returns:
        the endpoint URL string, or None if the notebook has no LiPDGraph URL
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(_LIPDGRAPH_ENDPOINT_PREFIX)):
            return node.value
    return None


def build_retrieval_cell(
    variable: str,
    tool: str,
    endpoint: str | None = None,
    fmt: str = "bibtex",
) -> str:
    """
    Builds the Python source for a single dataset's citation-retrieval cell.

    Args:
        variable: the notebook variable holding the dataset (from detection)
        tool: the dataset's source library - "PyLiPD", "PyleoTUPS", or
            "LiPDGraph" (case-insensitive)
        endpoint: LiPDGraph only - the graph endpoint the notebook queried
            (from extract_lipdgraph_endpoint). Falls back to _LIPDVERSE_ENDPOINT
            when None.
        fmt: "bibtex" (default) prints the raw BibTeX; "apa" renders it to APA
            in-kernel via bibliography.render_bibtex_strings_to_apa. When
            fmt="apa", the injected cell imports from bibliography, so src/ must
            be on the kernel's sys.path for the import to succeed.

    Returns:
        Python source that, run in the notebook's kernel, prints the dataset's
        citations and displays the accompanying metadata DataFrame

    Raises:
        ValueError: if tool is not one of the supported dataset sources
    """
    t = tool.lower()

    if t == "pylipd":
        body = f"_bib_{variable}, _meta_{variable} = {variable}.get_bibtex(remote=True)\n"
    elif t == "pyleotups":
        body = (
            f"_pub_{variable}, _meta_{variable} = {variable}.get_publications()\n"
            f'_bib_{variable} = [_pub_{variable}.to_string(bib_format="bibtex")]\n'
        )
    elif t == "lipdgraph":
        body = (
            "from pylipd.lipd import LiPD\n"
            f'_names_{variable} = {variable}["dataSetName"].unique().tolist()\n'
            f"_lipd_{variable} = LiPD()\n"
            f'_lipd_{variable}.set_endpoint("{endpoint or _LIPDVERSE_ENDPOINT}")\n'
            f"_lipd_{variable}.load_remote_datasets(_names_{variable})\n"
            f"_bib_{variable}, _meta_{variable} = _lipd_{variable}.get_bibtex(remote=True)\n"
        )
    else:
        raise ValueError(f"Unsupported dataset tool: {tool!r}")

    if fmt == "apa":
        out = (
            "from bibliography import render_bibtex_strings_to_apa\n"
            f"print(render_bibtex_strings_to_apa(_bib_{variable}))\n"
        )
    else:
        out = f'print("\\n".join(_bib_{variable}))\n'

    return body + out + f"display(_meta_{variable})"


def filter_datasets(
    pairs: list[list[str]],
    tool: str | None = None,
    variable: str | list[str] | None = None,
) -> list[list[str]]:
    """
    Narrows detected [variable, tool] pairs by tool and/or variable.

    Args:
        pairs: detected [variable, tool] pairs
        tool: if given, keep only pairs whose tool matches (case-insensitive)
        variable: if given, keep only pairs for this variable name or, if a
            list, any variable in it

    Returns:
        the filtered list of pairs (all pairs when no filter is given)
    """
    result = list(pairs)
    if tool is not None:
        result = [p for p in result if p[1].lower() == tool.lower()]
    if variable is not None:
        wanted = {variable} if isinstance(variable, str) else set(variable)
        result = [p for p in result if p[0] in wanted]
    return result


def inject_retrieval_cells(
    nb: nbformat.NotebookNode,
    pairs: list[list[str]],
    endpoint: str | None = None,
    fmt: str = "bibtex",
) -> nbformat.NotebookNode:
    """
    Appends one retrieval code cell per dataset pair to a notebook node.

    Args:
        nb: an nbformat notebook node (modified in place)
        pairs: [variable, tool] pairs to generate cells for
        endpoint: LiPDGraph endpoint to bake into LiPDGraph cells (see
            build_retrieval_cell); falls back to _LIPDVERSE_ENDPOINT when None
        fmt: "bibtex" (default) or "apa" - format for citations in the
            injected cell (see build_retrieval_cell)

    Returns:
        the same notebook node, with the retrieval cells appended
    """
    for variable, tool in pairs:
        nb.cells.append(
            nbformat.v4.new_code_cell(build_retrieval_cell(variable, tool, endpoint, fmt))
        )
    return nb


def generate_data_workflow(
    notebook_path: str,
    tool: str | None = None,
    variable: str | list[str] | None = None,
    output_path: str | None = None,
    fmt: str = "bibtex",
) -> list[list[str]]:
    """
    Detects datasets in a notebook and injects their citation-retrieval cells.

    Reads the notebook, detects its dataset variables via the LLM, optionally
    filters them, appends a retrieval cell per dataset, and writes the notebook
    back. The user then runs the injected cells in the live kernel to print the
    citations.

    Args:
        notebook_path: path to the .ipynb to analyze and modify
        tool: optional tool filter (e.g. "PyLiPD")
        variable: optional single-variable filter
        output_path: where to write the modified notebook (defaults to
            notebook_path, i.e. in place)
        fmt: "bibtex" (default) or "apa" - format for citations in the
            injected cell

    Returns:
        the [variable, tool] pairs that had cells injected
    """
    from dataset_detection import detect_datasets
    from notebook_parser import read_notebook_code

    code = read_notebook_code(notebook_path)
    pairs = filter_datasets(detect_datasets(code), tool=tool, variable=variable)
    endpoint = extract_lipdgraph_endpoint(code)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    inject_retrieval_cells(nb, pairs, endpoint, fmt)
    with open(output_path or notebook_path, "w") as f:
        nbformat.write(nb, f)

    return pairs
