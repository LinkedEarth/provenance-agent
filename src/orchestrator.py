"""
Orchestrator: exposes the software and data workflows as two tools that route to
the correct workflow with the correct arguments.

Purpose:
    A future natural-language agent ("@provenance agent, generate citations")
    will route requests to these tools. For now they are called directly. Each
    tool is a thin wrapper over existing functions - no new citation logic.

Implementation:
    - cite_software(notebook_path, libraries, citation_types, output_path): wraps
      generate_software_workflow, which injects a single cell that builds a
      pandas DataFrame of the software citations' metadata. Like cite_data, the
      citations exist as the injected cell's output, not as a return value.
    - cite_data(notebook_path, targets, fmt, output_path): wraps
      generate_data_workflow, which injects a retrieval cell per dataset whose
      output is the citation. Data citations exist as cell output, not a return
      value, because retrieval needs the live kernel objects.
    - cite_software_tool / cite_data_tool: LangChain StructuredTool wrappers whose
      descriptions are the routing "system prompts".

Design decisions:
    - Both tools now mutate the notebook and surface results as injected-cell
      output, so their contracts are symmetric: each returns what it injected a
      cell for (cite_software -> library names, cite_data -> [variable, tool]
      pairs), never the citation text itself.
    - fmt applies only to cite_data (whose cell can print BibTeX or render APA);
      the software cell always outputs a metadata DataFrame, so cite_software has
      no fmt argument.
    - Workflow modules (software_workflow, data_workflow) are deferred into the
      functions so importing this module stays cheap. LangChain StructuredTool is
      imported at module level to build the tool instances at import time (needed
      for agent routing).
"""

_VALID_FMT = ("apa", "bibtex")


def _check_fmt(fmt: str) -> None:
    """Raises ValueError unless fmt is 'apa' or 'bibtex'."""
    if fmt not in _VALID_FMT:
        raise ValueError(f"fmt must be one of {_VALID_FMT}, got {fmt!r}")


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
    function's return value (symmetric with cite_data).

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
    from software_workflow import generate_software_workflow

    return generate_software_workflow(
        notebook_path,
        libraries=libraries,
        citation_types=citation_types,
        output_path=output_path,
    )


def cite_data(
    notebook_path: str,
    targets: str | list[str] | None = None,
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
        "those) and `citation_types` ('paper' and/or 'software'). This injects a "
        "cell that builds a pandas DataFrame of the citation metadata; the user "
        "runs it to produce the output."
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
