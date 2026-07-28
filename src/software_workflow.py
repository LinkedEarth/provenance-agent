"""
Software workflow orchestrator: turns a notebook's imported libraries into a
citation-metadata cell injected into the notebook.

Purpose:
    The mirror image of data_workflow.py on the software side. Where the data
    workflow injects one retrieval cell per dataset (each calling a library
    method on a live kernel object), the software workflow injects a single cell
    that, when run, builds a pandas DataFrame of the software citations' metadata
    and binds it to _provbib_software. Both workflows now share the same shape:
    analyze the notebook, append cell(s) with nbformat, append the shared combine
    cell (bibliography.ensure_combine_cell) that displays every _provbib_*
    variable as one combined frame, write the notebook back, and let the user run
    the injected cells to see the result as cell output.

Implementation:
    - build_metadata_cell(libraries, citation_types): returns the Python source
      for the injected cell. The cell imports collect_library_entries from
      bibliography and calls it on the baked-in library list, binding the result
      to _provbib_software (no display() call - the combine cell handles that).
      collect_library_entries parses the local Citations/ .bib files with
      bibtexparser, so the DataFrame columns (library, citation_type, key,
      title, author, year, doi, bibtex, note) are the parsed metadata of each
      BibTeX entry. Imported libraries without a matching citation remain as a
      note row instead of disappearing.
    - inject_metadata_cell(nb, libraries, citation_types): appends that single
      code cell to an nbformat notebook node.
    - generate_software_workflow(notebook_path, libraries, citation_types,
      output_path): top-level glue - parse the imports, optionally filter to the
      requested libraries, inject the metadata cell, append/refresh the shared
      combine cell via bibliography.ensure_combine_cell, and write the notebook
      back.

Design decisions:
    - The cell imports from bibliography rather than baking the collected BibTeX
      inline, so it stays short and always reflects the current Citations/ data;
      this means src/ must be on the kernel's sys.path for the import to succeed
      (the demo notebooks add it, and the data workflow's APA cell has the same
      requirement).
    - One cell for all libraries (not one per library like the data workflow),
      because all software citations resolve to a single DataFrame - there is no
      per-library live object to reuse.
    - Citations are surfaced as the combine cell's OUTPUT, not as this module's
      return value, so the contract matches cite_data: the return value is the
      list of libraries the metadata cell was built for.
    - When no libraries match (empty notebook or a filter that hits nothing), no
      cell is injected and the notebook is left untouched, mirroring the data
      workflow's "nothing detected" path.
    - Standard-library imports (sys, os, json, io, ast, pathlib, ...) are dropped
      from the import list before the cell is built. They ship with the
      interpreter and nobody cites them, and leaving them in produced one
      "No citation found" row per stdlib module. collect_library_entries drops
      them too, so a direct caller passing a stdlib name in gets the same
      answer.
"""

import nbformat


def build_metadata_cell(
    libraries: list[str],
    citation_types: list[str] | None = None,
) -> str:
    """
    Builds the Python source for the software citation-metadata cell.

    Args:
        libraries: library names to cite (baked into the cell as a literal)
        citation_types: optional filter - "paper" and/or "software"; None means
            both

    Returns:
        Python source that, run in the notebook's kernel, binds
        _provbib_software to a pandas DataFrame of the libraries' citation
        metadata. It does not display() the frame - the combine cell appended
        by generate_software_workflow displays the combined frame instead. The
        cell imports collect_library_entries from bibliography, so src/ must be
        on the kernel's sys.path.
    """
    from notebook_parser import PROVENANCE_CELL_MARKER

    return (
        f"{PROVENANCE_CELL_MARKER}\n"
        "from bibliography import collect_library_entries\n"
        f"_provbib_software = collect_library_entries({libraries!r}, {citation_types!r})"
    )


def inject_metadata_cell(
    nb: nbformat.NotebookNode,
    libraries: list[str],
    citation_types: list[str] | None = None,
) -> nbformat.NotebookNode:
    """
    Appends the software citation-metadata cell to a notebook node.

    Args:
        nb: an nbformat notebook node (modified in place)
        libraries: library names to cite
        citation_types: optional "paper"/"software" filter passed to the cell

    Returns:
        the same notebook node, with the metadata cell appended
    """
    nb.cells.append(
        nbformat.v4.new_code_cell(build_metadata_cell(libraries, citation_types))
    )
    return nb


def generate_software_workflow(
    notebook_path: str,
    libraries: str | list[str] | None = None,
    citation_types: list[str] | None = None,
    output_path: str | None = None,
) -> list[str]:
    """
    Detects a notebook's imported libraries and injects their metadata cell.

    Reads the notebook, extracts its imports, optionally narrows them to the
    requested libraries, appends a cell that binds _provbib_software to the
    citation-metadata DataFrame, then appends (or refreshes) the shared combine
    cell via bibliography.ensure_combine_cell, and writes the notebook back. The
    user then runs the injected cells to see the combined DataFrame as cell
    output.

    Args:
        notebook_path: path to the .ipynb to analyze and modify
        libraries: None (all imported libraries), a single name, or a list of
            names to cite; names not imported by the notebook are dropped
        citation_types: optional filter - "paper" and/or "software"
        output_path: where to write the modified notebook (defaults to
            notebook_path, i.e. in place)

    Returns:
        the library names the metadata cell was built for (empty when nothing
        matched, in which case the notebook is left untouched)
    """
    from bibliography import is_stdlib
    from notebook_parser import parse_notebook, validate_libraries

    # Dropped before the cell is built, so the baked-in library list and the
    # reported result name only libraries someone would actually cite.
    available = [lib for lib in parse_notebook(notebook_path) if not is_stdlib(lib)]
    if libraries is None:
        wanted = available
    else:
        requested = [libraries] if isinstance(libraries, str) else list(libraries)
        wanted, _not_found = validate_libraries(requested, available)

    if not wanted:
        return []

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    inject_metadata_cell(nb, wanted, citation_types)

    from bibliography import ensure_combine_cell
    ensure_combine_cell(nb)

    with open(output_path or notebook_path, "w") as f:
        nbformat.write(nb, f)

    return wanted
