"""
Orchestrator: exposes the software and data workflows as two tools that route to
the correct workflow with the correct arguments.

Purpose:
    A future natural-language agent ("@provenance agent, generate citations")
    will route requests to these tools. For now they are called directly. Each
    tool is a thin wrapper over existing functions - no new citation logic.

Implementation:
    - cite_software(notebook_path, libraries, citation_types, fmt): in-process.
      parse_notebook -> collect_library_entries -> render_apa (when fmt="apa").
    - cite_data(notebook_path, targets, fmt, output_path): wraps
      generate_data_workflow, which injects a retrieval cell per dataset whose
      output is the citation. Data citations exist as cell output, not a return
      value, because retrieval needs the live kernel objects.
    - cite_software_tool / cite_data_tool: LangChain StructuredTool wrappers whose
      descriptions are the routing "system prompts".

Design decisions:
    - fmt defaults to "apa" (the finished product is a human-readable
      bibliography); fmt="bibtex" skips the Gemini call for the raw artifact.
    - Heavy imports (notebook_parser, bibliography, data_workflow) are deferred
      into the functions so importing this module stays cheap and side-effect free.
"""

_VALID_FMT = ("apa", "bibtex")


def _check_fmt(fmt: str) -> None:
    """Raises ValueError unless fmt is 'apa' or 'bibtex'."""
    if fmt not in _VALID_FMT:
        raise ValueError(f"fmt must be one of {_VALID_FMT}, got {fmt!r}")


def cite_software(
    notebook_path: str,
    libraries=None,
    citation_types: list[str] | None = None,
    fmt: str = "apa",
) -> str:
    """
    Cites the software libraries a notebook imports.

    Args:
        notebook_path: path to the .ipynb to analyze
        libraries: None (all imported libraries), a single name, or a list of
            names to cite
        citation_types: optional filter - "paper" and/or "software"
        fmt: "apa" (default) or "bibtex"

    Returns:
        the citation text (APA or BibTeX); when specific libraries were asked for
        but are not imported, a note line is appended for each
    """
    _check_fmt(fmt)
    from notebook_parser import parse_notebook, validate_libraries
    from bibliography import collect_library_entries, render_apa

    available = parse_notebook(notebook_path)
    if libraries is None:
        wanted, not_found = available, []
    else:
        requested = [libraries] if isinstance(libraries, str) else list(libraries)
        wanted, not_found = validate_libraries(requested, available)

    entries = collect_library_entries(wanted, citation_types)
    if fmt == "apa":
        body = render_apa(entries)
    else:
        body = entries.to_string(bib_format="bibtex")

    if not_found:
        notes = "\n".join(f"[Not imported in notebook: {lib}]" for lib in not_found)
        body = f"{body}\n\n{notes}" if body else notes
    return body


def cite_data(
    notebook_path: str,
    targets=None,
    fmt: str = "apa",
    output_path: str | None = None,
) -> list[list[str]]:
    """
    Cites the datasets a notebook uses by injecting a retrieval cell per dataset.

    Detection is static (works even if the notebook was never run), but retrieval
    reuses the live kernel objects, so the citations appear as the injected
    cells' OUTPUT when the user runs them - not as this function's return value.

    Args:
        notebook_path: path to the .ipynb to analyze and modify
        targets: None (all detected datasets), a single variable name, or a list
        fmt: "apa" (default) or "bibtex"
        output_path: where to write the modified notebook (defaults to in place)

    Returns:
        the [variable, tool] pairs that had retrieval cells injected
    """
    _check_fmt(fmt)
    from data_workflow import generate_data_workflow

    return generate_data_workflow(
        notebook_path,
        variable=targets,
        output_path=output_path,
        fmt=fmt,
    )


from langchain_core.tools import StructuredTool

cite_software_tool = StructuredTool.from_function(
    func=cite_software,
    name="cite_software",
    description=(
        "Cite the software libraries a Jupyter notebook imports. Use this for "
        "requests about citing software, packages, or libraries. Pass "
        "`notebook_path`; optionally `libraries` (a name or list to cite only "
        "those), `citation_types` ('paper' and/or 'software'), and `fmt` "
        "('apa' default, or 'bibtex')."
    ),
)

cite_data_tool = StructuredTool.from_function(
    func=cite_data,
    name="cite_data",
    description=(
        "Cite the datasets a Jupyter notebook uses (PyLiPD, PyleoTUPS, or "
        "LiPDGraph). Use this for requests about citing data or datasets. Pass "
        "`notebook_path`; optionally `targets` (a variable name or list to cite "
        "only those) and `fmt` ('apa' default, or 'bibtex'). This injects a "
        "retrieval cell per dataset; the user runs it to produce the citation."
    ),
)
