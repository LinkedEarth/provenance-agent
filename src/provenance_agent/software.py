"""
The whole software side: a notebook's imported libraries in, one injected
citation-metadata cell out, plus the public function and LangChain tool that
callers reach it by.

Purpose:
    The mirror image of data.py on the software side. Each workflow owns
    exactly one cell: this one builds a pandas DataFrame of the software
    citations' metadata, binds it to provenance_software, and displays it. Both
    workflows share the same shape - analyze the notebook, append one cell with
    nbformat, write the notebook back, and let the user run the cell to see its
    DataFrame as the cell's output.

Implementation:
    - build_metadata_cell(libraries, citation_types): returns the Python source
      for the injected cell. The cell imports collect_library_entries from
      provenance_agent.citations, calls it on the baked-in library list,
      binds the result to
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
    - cite_software(...): the public name for that workflow, re-exported from
      the package root.
    - cite_software_tool: the LangChain StructuredTool wrapper, for direct and
      API callers. It is not bound to the LCEL classifier.

Design decisions:
    - The public function, the workflow, and the cell builder live in one
      module because they are one operation at three levels of detail. A reader
      chasing "what does cite_software actually do" never crosses a file
      boundary to reach a one-line delegation.
    - cite_software delegates to generate_software_workflow rather than
      replacing it. Both names are public: cite_software is the API callers and
      the agent use, generate_software_workflow is the step name notebook
      tooling regenerates cells with. One implementation, two entry points.
    - The cell imports from provenance_agent.citations rather than baking the
      collected BibTeX inline, so it stays short and always reflects the current
      Citations/ data. The import is package-qualified, so the notebook's kernel
      only needs the package installed (`pip install -e ".[dev]"`) - no
      sys.path manipulation in the notebook.
    - One cell for all libraries, because all software citations resolve to a
      single DataFrame - there is no per-library live object to reuse.
    - Citations are surfaced as the injected cell's OUTPUT, not as this module's
      return value, so the contract matches cite_data: the return value is the
      list of libraries the metadata cell was built for.
    - There is no combined software-plus-data frame. The software workflow
      produces a self-displaying cell, while the data workflow produces an
      independent retrieval cell; each can be re-run or deleted on its own.
    - When no libraries match (empty notebook or a filter that hits nothing), no
      cell is injected and the notebook is left untouched, mirroring the data
      workflow's "nothing detected" path.
    - Standard-library imports (sys, os, json, io, ast, pathlib, ...) are dropped
      from the import list before the cell is built. They ship with the
      interpreter and nobody cites them, so keeping them would add one
      "No citation found" row per stdlib module. collect_library_entries drops
      them too, so a direct caller passing a stdlib name in gets the same
      answer.
"""

import nbformat
from langchain_core.tools import StructuredTool


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
        cell imports collect_library_entries from provenance_agent.citations,
        so the package must be installed in the kernel's environment
        (`pip install -e ".[dev]"`).
    """
    from .notebook_io import PROVENANCE_CELL_MARKER

    return (
        f"{PROVENANCE_CELL_MARKER}\n"
        "from provenance_agent.citations import collect_library_entries\n"
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
    requested libraries, appends a cell that binds provenance_software to the
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
    from .citations import is_stdlib
    from .notebook_io import parse_notebook, validate_libraries

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
    from .citations import (
        SOFTWARE_FRAME,
        remove_legacy_combine_cells,
        remove_provenance_cells,
    )
    # Clear the previous run's cells first, then append, so a re-run replaces
    # its own cell rather than stacking a second, stale one beside it.
    remove_provenance_cells(nb, SOFTWARE_FRAME)
    remove_legacy_combine_cells(nb)
    inject_metadata_cell(nb, wanted, citation_types)

    with open(output_path or notebook_path, "w") as f:
        nbformat.write(nb, f)

    return wanted


def cite_software(
    notebook_path: str,
    libraries: str | list[str] | None = None,
    citation_types: list[str] | None = None,
    output_path: str | None = None,
) -> list[str]:
    """
    Cites the software libraries a notebook imports by injecting a metadata cell.

    Detection is static (works even if the notebook was never run), and the
    injected cell builds a pandas DataFrame of the citations' metadata, so the
    citations appear as the cell's OUTPUT when the user runs it - not as this
    function's return value.

    Args:
        notebook_path: path to the .ipynb to analyze and modify
        libraries: None (all imported libraries), a single name, or a list of
            names to cite; names not imported by the notebook are dropped
        citation_types: optional filter - "paper" and/or "software"
        output_path: where to write the modified notebook (defaults to in place)

    Returns:
        the library names the injected metadata cell was built for (empty when
        nothing matched)
    """
    return generate_software_workflow(
        notebook_path,
        libraries=libraries,
        citation_types=citation_types,
        output_path=output_path,
    )


cite_software_tool = StructuredTool.from_function(
    func=cite_software,
    name="cite_software",
    description=(
        "Cite the software libraries a Jupyter notebook imports. Use this for "
        "requests about citing software, packages, or libraries. Pass "
        "`notebook_path`; optionally `libraries` (a name or list to cite only "
        "those) and `citation_types` ('paper' and/or 'software'). This injects a "
        "cell that builds a pandas DataFrame of the citation metadata; the user "
        "runs it to produce the output."
    ),
)
