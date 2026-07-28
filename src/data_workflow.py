"""
Data workflow orchestrator: turns detected dataset variables into dataset
citations by injecting retrieval cells into the notebook's live kernel.

Purpose:
    Given the [variable, tool] pairs from dataset_detection, generate one code
    cell that retrieves every dataset's BibTeX, and append it to the notebook
    with nbformat. The cell is meant to run in the notebook's own kernel, where
    the already-loaded objects (LiPD objects, PyleoTUPS datasets, the LiPDGraph
    result DataFrame) are reused - no re-querying or re-loading. Its output is
    the citations plus a provenance_datasets DataFrame. The fmt parameter
    controls whether it prints raw BibTeX or renders APA.

Implementation:
    - extract_lipdgraph_endpoint(code): AST-scans the notebook for the
      LinkedEarth graph endpoint URL (the string passed to requests.post), so
      the LiPDGraph pathway loads from the same repository the notebook queried.
    - build_retrieval_cell(variable, tool, endpoint, dataset_names): returns the
      retrieval block for one dataset - a fragment, not a standalone cell.
      PyLiPD/PyleoTUPS call the library method directly on the in-memory object
      (Approach C: {var}.{method}).
      Optional dataSetName filters are applied case-insensitively and exactly to
      the returned metadata DataFrame, after retrieval.
      LiPDGraph is special: the terminal variable is a DataFrame, so the cell
      pulls its dataSetName column, loads those datasets into a fresh LiPD object
      from the endpoint, then calls get_bibtex(). When fmt="apa", the cell pipes
      the collected BibTeX to bibliography.render_bibtex_strings_to_apa() for
      APA rendering in-kernel.
    - build_dataset_cell(pairs, endpoint, fmt, dataset_names): composes one
      retrieval block per dataset into the single injected cell, prints the
      citations once, and concatenates every _meta_{variable} into one
      provenance_datasets DataFrame that the cell displays.
    - filter_datasets(pairs, tool, variable): retains the legacy variable-level
      filter behavior.
    - split_targets(pairs, targets): separates detected variable names from
      unmatched dataSetName filters; unmatched names keep all detected pairs so
      each retrieval cell can filter its returned metadata.
    - inject_retrieval_cells(nb, pairs, endpoint, fmt, dataset_names): appends
      the single dataset cell covering every pair to an nbformat notebook node,
      or nothing when pairs is empty. fmt defaults to "bibtex" and accepts "apa"
      to render citations in APA format.
    - generate_data_workflow(..., fmt): top-level glue - detect, filter, inject
      the single dataset cell, then write. fmt defaults to "bibtex" and can be
      "apa" for APA-formatted output.

Design decisions:
    - Cells are written for the user to run (live-kernel model); this module does
      not execute them. Collecting the printed BibTeX happens at notebook runtime.
    - The LiPDGraph endpoint is lifted from the notebook via AST rather than
      hardcoded, so a notebook pointed at a different repository is handled
      correctly. _LIPDVERSE_ENDPOINT is only a fallback when no URL is found.
    - Dataset-name filters are applied after get_bibtex()/get_publications() to
      the source-provided metadata DataFrame. This keeps the source retrieval
      path unchanged and lets provenance_datasets preserve every field returned
      by the data library.
    - Unsupported tools raise ValueError so a mis-detected pair fails loudly
      rather than silently producing an empty bibliography.
    - APA rendering happens in the injected cell (via render_bibtex_strings_to_apa)
      so the user can see formatted citations as output without re-running code.
    - Both PyLiPD's get_bibtex() and PyleoTUPS' get_publications() return
      (citations, metadata DataFrame). Each retrieval block keeps its
      _meta_{variable} bound, and the cell pd.concat()s them into
      provenance_datasets, so sources with different metadata columns produce
      the union with nulls filled in.
    - One cell for every dataset, not one per dataset, so the notebook gains
      exactly two provenance cells at most: provenance_software and
      provenance_datasets. The tradeoff is that a retrieval failure for one
      dataset stops the whole cell.
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
    dataset_names: list[str] | None = None,
) -> str:
    """
    Builds one dataset's retrieval block for the shared dataset cell.

    This is a fragment, not a standalone cell: it leaves _bib_{variable} and
    _meta_{variable} bound in the kernel and prints nothing. build_dataset_cell
    concatenates one block per dataset and appends the shared print/display.

    Args:
        variable: the notebook variable holding the dataset (from detection)
        tool: the dataset's source library - "PyLiPD", "PyleoTUPS", or
            "LiPDGraph" (case-insensitive)
        endpoint: LiPDGraph only - the graph endpoint the notebook queried
            (from extract_lipdgraph_endpoint). Falls back to _LIPDVERSE_ENDPOINT
            when None.
        dataset_names: optional dataSetName values to match case-insensitively
            and exactly in the returned metadata DataFrame. If the source does
            not expose a ``dsname`` column, the metadata is left unchanged.

    Returns:
        Python source binding _bib_{variable} and _meta_{variable}

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
        names = f'_names_{variable} = {variable}["dataSetName"].unique().tolist()\n'
        body = (
            "from pylipd.lipd import LiPD\n"
            + names
            + f"_lipd_{variable} = LiPD()\n"
            + f'_lipd_{variable}.set_endpoint("{endpoint or _LIPDVERSE_ENDPOINT}")\n'
            + f"_lipd_{variable}.load_remote_datasets(_names_{variable})\n"
            + f"_bib_{variable}, _meta_{variable} = _lipd_{variable}.get_bibtex(remote=True)\n"
        )
    else:
        raise ValueError(f"Unsupported dataset tool: {tool!r}")


    #D=LiPD()
    #D.setendpoint()
    #D.load_remote_datasets("datasetname")
    #Bib,df = D.get_bibtext()


    metadata_filter = ""
    if dataset_names:
        metadata_filter = (
            f"_want_{variable} = {{name.casefold() for name in {dataset_names!r}}}\n"
            f"if \"dsname\" in _meta_{variable}.columns:\n"
            f"    _meta_{variable} = _meta_{variable}[\n"
            f"        _meta_{variable}[\"dsname\"].astype(\"string\").str.casefold().isin(_want_{variable})\n"
            "    ]\n"
        )

    return body + metadata_filter


def build_dataset_cell(
    pairs: list[list[str]],
    endpoint: str | None = None,
    fmt: str = "bibtex",
    dataset_names: list[str] | None = None,
) -> str:
    """
    Builds the source for the single dataset-citation cell.

    Composes one retrieval block per detected dataset, prints the collected
    citations, and concatenates every per-dataset metadata frame into one
    ``provenance_datasets`` DataFrame that the cell displays.

    Args:
        pairs: the [variable, tool] pairs to retrieve citations for
        endpoint: LiPDGraph endpoint baked into any LiPDGraph block
        fmt: "bibtex" (default) prints raw BibTeX; "apa" renders APA in-kernel
        dataset_names: optional exact, case-insensitive dataSetName filters

    Returns:
        Python source that, run in the notebook's kernel, prints the datasets'
        citations and displays provenance_datasets

    Raises:
        ValueError: if any pair names an unsupported dataset source
    """
    variables = [pair[0] for pair in pairs]
    blocks = "".join(
        build_retrieval_cell(variable, tool, endpoint, dataset_names)
        for variable, tool in pairs
    )

    collected = " + ".join(f"_bib_{variable}" for variable in variables)
    if fmt == "apa":
        out = (
            "from bibliography import render_bibtex_strings_to_apa\n"
            f"print(render_bibtex_strings_to_apa({collected}))\n"
        )
    else:
        out = f'print("\\n".join({collected}))\n'

    # The marker goes on the cell, not on each retrieval block, since the block
    # is a fragment that never becomes a cell of its own.
    from notebook_parser import PROVENANCE_CELL_MARKER

    frames = ", ".join(f"_meta_{variable}" for variable in variables)
    display = (
        "import pandas as pd\n"
        f"provenance_datasets = pd.concat([{frames}], ignore_index=True)\n"
        "display(provenance_datasets)"
    )
    return f"{PROVENANCE_CELL_MARKER}\n" + blocks + out + display


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


def split_targets(
    pairs: list[list[str]],
    targets: str | list[str] | None,
) -> tuple[list[list[str]], list[str]]:
    """
    Splits requested targets into variable matches and dataset-name filters.

    Args:
        pairs: detected [variable, tool] dataset pairs
        targets: None, a variable name, a dataSetName, or a list containing
            either kind of target

    Returns:
        A tuple of (pairs_to_inject, dataset_names). Exact variable matches
        select only those pairs when no dataset-name target is present. Any
        unmatched target is treated as a dataSetName and keeps all detected
        pairs so each retrieval source can apply its own filter.
    """
    if targets is None or targets == []:
        return list(pairs), []

    requested = [targets] if isinstance(targets, str) else list(targets)
    variables = {pair[0].casefold(): pair[0] for pair in pairs}
    variable_targets = {
        variables[target.casefold()]
        for target in requested
        if target.casefold() in variables
    }
    dataset_names = [
        target for target in requested if target.casefold() not in variables
    ]

    if dataset_names:
        return list(pairs), dataset_names
    return [pair for pair in pairs if pair[0] in variable_targets], []


def inject_retrieval_cells(
    nb: nbformat.NotebookNode,
    pairs: list[list[str]],
    endpoint: str | None = None,
    fmt: str = "bibtex",
    dataset_names: list[str] | None = None,
) -> nbformat.NotebookNode:
    """
    Appends the single dataset-citation cell covering every detected dataset.

    One cell, not one per dataset, so the notebook gains exactly one cell whose
    output is the datasets' citations and the provenance_datasets DataFrame.

    Args:
        nb: an nbformat notebook node (modified in place)
        pairs: [variable, tool] pairs to generate the cell for; an empty list
            appends nothing
        endpoint: LiPDGraph endpoint to bake into any LiPDGraph block; falls
            back to _LIPDVERSE_ENDPOINT when None
        fmt: "bibtex" (default) or "apa" - citation format for the cell
        dataset_names: optional exact, case-insensitive dataSetName filters

    Returns:
        the same notebook node, with the dataset cell appended
    """
    if not pairs:
        return nb
    nb.cells.append(
        nbformat.v4.new_code_cell(
            build_dataset_cell(pairs, endpoint, fmt, dataset_names)
        )
    )
    return nb


def generate_data_workflow(
    notebook_path: str,
    tool: str | None = None,
    variable: str | list[str] | None = None,
    output_path: str | None = None,
    fmt: str = "bibtex",
    targets: str | list[str] | None = None,
    detected_pairs: list[list[str]] | None = None,
) -> list[list[str]]:
    """
    Detects datasets in a notebook and injects their citation-retrieval cells.

    Reads the notebook, detects its dataset variables via the LLM, optionally
    filters them, appends the single dataset-citation cell, and writes the
    notebook back. The user then runs that cell in the live kernel to print the
    citations and see the provenance_datasets DataFrame as its output.

    Args:
        notebook_path: path to the .ipynb to analyze and modify
        tool: optional tool filter (e.g. "PyLiPD")
        variable: legacy variable-only filter. Use targets for new callers.
        output_path: where to write the modified notebook (defaults to
            notebook_path, i.e. in place)
        fmt: "bibtex" (default) or "apa" - format for citations in the
            injected cell
        targets: optional variable names and/or exact dataSetName values;
            unmatched targets are applied inside retrieval cells
        detected_pairs: optional precomputed detector result used by the LCEL
            agent to avoid running dataset detection twice

    Returns:
        the [variable, tool] pairs that had cells injected
    """
    from dataset_detection import detect_datasets
    from notebook_parser import read_notebook_code

    if targets is not None and variable is not None:
        raise ValueError("pass either targets or variable, not both")

    code = read_notebook_code(notebook_path)
    detected = detect_datasets(code) if detected_pairs is None else detected_pairs
    pairs, dataset_names = split_targets(
        detected,
        targets if targets is not None else variable,
    )
    pairs = filter_datasets(pairs, tool=tool)
    endpoint = extract_lipdgraph_endpoint(code)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    from bibliography import (
        DATASETS_FRAME,
        remove_legacy_combine_cells,
        remove_provenance_cells,
    )
    # Clear the previous run's cells first, then append, so a re-run replaces
    # its own cell rather than stacking a second, stale one beside it.
    remove_provenance_cells(nb, DATASETS_FRAME)
    remove_legacy_combine_cells(nb)
    inject_retrieval_cells(nb, pairs, endpoint, fmt, dataset_names)

    with open(output_path or notebook_path, "w") as f:
        nbformat.write(nb, f)

    return pairs
