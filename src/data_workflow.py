"""
Data workflow orchestrator: turns detected dataset variables into dataset
citations by injecting retrieval cells into the notebook's live kernel.

Purpose:
    Given the [variable, tool] pairs from dataset_detection, generate a code
    cell per dataset that retrieves its BibTeX, and append those cells to the
    notebook with nbformat. The cells are meant to run in the notebook's own
    kernel, where the already-loaded objects (LiPD objects, PyleoTUPS datasets,
    the LiPDGraph result DataFrame) are reused - no re-querying or re-loading.

Implementation:
    - build_retrieval_cell(variable, tool): returns the Python source for one
      dataset's retrieval cell. PyLiPD/PyleoTUPS call the library method directly
      on the in-memory object (Approach C: {var}.{method}). LiPDGraph is special:
      the terminal variable is a DataFrame, so the cell pulls its dataSetName
      column, loads those datasets into a fresh LiPD object from the LiPDVerse
      endpoint, then calls get_bibtex().
    - filter_datasets(pairs, tool, variable): narrows the detected pairs so the
      workflow can cite all datasets, only one tool's datasets, or one variable.
    - inject_retrieval_cells(nb, pairs): appends one retrieval code cell per pair
      to an nbformat notebook node.
    - generate_data_workflow(...): top-level glue - detect, filter, inject, write.

Design decisions:
    - Cells are written for the user to run (live-kernel model); this module does
      not execute them. Collecting the printed BibTeX happens at notebook runtime.
    - The LiPDVerse endpoint is hardcoded (same constant as pylipd_helper) rather
      than parsed from the notebook, because it is a fixed, known endpoint and the
      detector does not surface the notebook's url variable name.
    - Unsupported tools raise ValueError so a mis-detected pair fails loudly
      rather than silently producing an empty bibliography.
"""

import nbformat


_LIPDVERSE_ENDPOINT = "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic"


def build_retrieval_cell(variable: str, tool: str) -> str:
    """
    Builds the Python source for a single dataset's citation-retrieval cell.

    Args:
        variable: the notebook variable holding the dataset (from detection)
        tool: the dataset's source library - "PyLiPD", "PyleoTUPS", or
            "LiPDGraph" (case-insensitive)

    Returns:
        Python source that, run in the notebook's kernel, prints the dataset's
        BibTeX

    Raises:
        ValueError: if tool is not one of the supported dataset sources
    """
    t = tool.lower()

    if t == "pylipd":
        return (
            f"_bib_{variable}, _ = {variable}.get_bibtex(remote=True)\n"
            f'print("\\n".join(_bib_{variable}))'
        )

    if t == "pyleotups":
        return (
            f"_pub_{variable}, _ = {variable}.get_publications()\n"
            f'print(_pub_{variable}.to_string(bib_format="bibtex"))'
        )

    if t == "lipdgraph":
        return (
            "from pylipd.lipd import LiPD\n"
            f'_names_{variable} = {variable}["dataSetName"].unique().tolist()\n'
            f"_lipd_{variable} = LiPD()\n"
            f'_lipd_{variable}.set_endpoint("{_LIPDVERSE_ENDPOINT}")\n'
            f"_lipd_{variable}.load_remote_datasets(_names_{variable})\n"
            f"_bib_{variable}, _ = _lipd_{variable}.get_bibtex(remote=True)\n"
            f'print("\\n".join(_bib_{variable}))'
        )

    raise ValueError(f"Unsupported dataset tool: {tool!r}")


def filter_datasets(
    pairs: list[list[str]],
    tool: str | None = None,
    variable: str | None = None,
) -> list[list[str]]:
    """
    Narrows detected [variable, tool] pairs by tool and/or variable.

    Args:
        pairs: detected [variable, tool] pairs
        tool: if given, keep only pairs whose tool matches (case-insensitive)
        variable: if given, keep only the pair for this variable

    Returns:
        the filtered list of pairs (all pairs when no filter is given)
    """
    result = list(pairs)
    if tool is not None:
        result = [p for p in result if p[1].lower() == tool.lower()]
    if variable is not None:
        result = [p for p in result if p[0] == variable]
    return result


def inject_retrieval_cells(
    nb: nbformat.NotebookNode,
    pairs: list[list[str]],
) -> nbformat.NotebookNode:
    """
    Appends one retrieval code cell per dataset pair to a notebook node.

    Args:
        nb: an nbformat notebook node (modified in place)
        pairs: [variable, tool] pairs to generate cells for

    Returns:
        the same notebook node, with the retrieval cells appended
    """
    for variable, tool in pairs:
        nb.cells.append(nbformat.v4.new_code_cell(build_retrieval_cell(variable, tool)))
    return nb


def generate_data_workflow(
    notebook_path: str,
    tool: str | None = None,
    variable: str | None = None,
    output_path: str | None = None,
) -> list[list[str]]:
    """
    Detects datasets in a notebook and injects their citation-retrieval cells.

    Reads the notebook, detects its dataset variables via the LLM, optionally
    filters them, appends a retrieval cell per dataset, and writes the notebook
    back. The user then runs the injected cells in the live kernel to print the
    BibTeX.

    Args:
        notebook_path: path to the .ipynb to analyze and modify
        tool: optional tool filter (e.g. "PyLiPD")
        variable: optional single-variable filter
        output_path: where to write the modified notebook (defaults to
            notebook_path, i.e. in place)

    Returns:
        the [variable, tool] pairs that had cells injected
    """
    from dataset_detection import detect_datasets
    from notebook_parser import read_notebook_code

    pairs = filter_datasets(
        detect_datasets(read_notebook_code(notebook_path)),
        tool=tool,
        variable=variable,
    )

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    inject_retrieval_cells(nb, pairs)
    with open(output_path or notebook_path, "w") as f:
        nbformat.write(nb, f)

    return pairs
