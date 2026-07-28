"""
Software workflow orchestrator: turns a notebook's imported libraries into a
citation-metadata cell injected into the notebook.

Purpose:
    The mirror image of data_workflow.py on the software side. Each workflow
    owns exactly one cell: this one builds a pandas DataFrame of the software
    citations' metadata, binds it to provenance_software, and displays it. Both
    workflows share the same shape - analyze the notebook, append one cell with
    nbformat, write the notebook back, and let the user run the cell to see its
    DataFrame as the cell's output.

Implementation:
    - build_metadata_cell(libraries, citation_types): returns the Python source
      for the injected cell. The cell imports collect_library_entries from
      bibliography, calls it on the baked-in library list, binds the result to
      provenance_software, and display()s it. collect_library_entries parses the
      local Citations/ .bib files with bibtexparser, so the DataFrame columns
      (library, citation_type, key, title, author, year, doi, bibtex, note) are
      the parsed metadata of each BibTeX entry. Imported libraries without a
      matching citation remain as a note row instead of disappearing.
    - inject_metadata_cell(nb, libraries, citation_types): appends that single
      code cell to an nbformat notebook node.
    - generate_software_workflow(notebook_path, libraries, citation_types,
      output_path): top-level glue - parse the imports, optionally filter to the
      requested libraries, inject the metadata cell, and write the notebook back.

Design decisions:
    - The cell imports from bibliography rather than baking the collected BibTeX
      inline, so it stays short and always reflects the current Citations/ data;
      this means src/ must be on the kernel's sys.path for the import to succeed
      (the demo notebooks add it, and the data workflow's APA cell has the same
      requirement).
    - One cell for all libraries, because all software citations resolve to a
      single DataFrame - there is no per-library live object to reuse.
    - Citations are surfaced as the injected cell's OUTPUT, not as this module's
      return value, so the contract matches cite_data: the return value is the
      list of libraries the metadata cell was built for.
    - There is no combined software-plus-data frame. The two workflows produce
      two independent, self-displaying cells (provenance_software and
      provenance_datasets), so each can be read, re-run, or deleted on its own.
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
        provenance_software to a pandas DataFrame of the libraries' citation
        metadata and displays it, so the citations are the cell's output. The
        cell imports collect_library_entries from bibliography, so src/ must be
        on the kernel's sys.path.
    """
    from notebook_parser import PROVENANCE_CELL_MARKER

    return (
        f"{PROVENANCE_CELL_MARKER}\n"
        "from bibliography import collect_library_entries\n"
        f"provenance_software = collect_library_entries({libraries!r}, {citation_types!r})\n"
        "display(provenance_software)"
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
    citation-metadata DataFrame and displays it, then writes the notebook back.
    The user runs that cell to see the DataFrame as its output.

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

    from bibliography import remove_legacy_combine_cells
    remove_legacy_combine_cells(nb)

    with open(output_path or notebook_path, "w") as f:
        nbformat.write(nb, f)

    return wanted
