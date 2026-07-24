"""
IPython extension exposing the provenance agent as a notebook magic.

Purpose:
    Let a user ask for citations from inside the notebook they are working in:

        import sys; sys.path.append("../src")
        %load_ext provenance

        %provenance cite the software
        %provenance cite the datasets
        %provenance cite Pyleoclim

    This is the notebook-native delivery of the "@provenance agent, generate
    citations" experience.

Implementation:
    - resolve_notebook_path(): the path the tools analyze. An explicit path set
      by %provenance_notebook wins; otherwise _autodetect_path() (ipynbname);
      otherwise UsageError.
    - set_notebook_path(path): stores the session override, returns the
      confirmation line the magic prints.
    - cite(request): resolve path -> agent.run() -> formatted text.
    - _format_result(call) / _format_results(calls): render what agent.run()
      returned. Both tools inject cells and return a list of what they injected
      a cell for (cite_software -> library names, cite_data -> [variable, tool]
      pairs), so formatting dispatches on tool name and reports what to run.
    - ProvenanceMagics / load_ipython_extension(ipython): registration, so
      %load_ext provenance works.

Design decisions:
    - No citation or routing logic lives here. agent.run() already resolves a
      request to a tool call and executes it; this module only resolves the
      notebook path and renders the result. It imports `agent` and nothing else
      from the project, so the routing contract stays in one place.
    - The logic is module-level functions rather than methods on the Magics
      class, so tests exercise it without constructing an IPython shell.
    - UsageError (not RuntimeError) for user mistakes: IPython renders it as a
      single clean line instead of a traceback.
    - A session override exists because ipynbname works by matching the running
      kernel against the Jupyter server's session list, which routinely fails in
      VSCode notebooks - the primary environment for this project. Auto-detect
      stays the default; the override is the recovery path.
    - Neither tool's citations exist yet when it returns: both inject cells whose
      OUTPUT is the citation (cite_data a retrieval cell per dataset, cite_software
      a single metadata-DataFrame cell). So each summary reports what was injected
      and tells the user to run the cells, rather than implying citations were
      produced.
    - Every workflow reads the .ipynb from disk, so unsaved cells are invisible.
      Empty results say so instead of being reported as the answer.
"""

from IPython.core.error import UsageError
from IPython.core.magic import Magics, line_magic, magics_class

from agent import run


# Set by %provenance_notebook; None means fall back to auto-detection.
_notebook_path: str | None = None

_DISK_NOTE = (
    "The notebook was read from disk, so unsaved cells are not visible. "
    "Save the notebook and try again if something is missing."
)

_NO_ROUTE = (
    "Could not route that request. This agent cites the software a notebook "
    "imports (e.g. 'cite the software', 'cite Pyleoclim') and the datasets it "
    "uses (e.g. 'cite the datasets', 'cite the PyLiPD datasets')."
)


def _autodetect_path() -> str | None:
    """
    Detects the running notebook's path via ipynbname.

    Returns:
        the notebook path, or None if detection is unavailable or fails
        (ipynbname not installed, no running kernel, or no matching Jupyter
        server session - the usual case in VSCode)
    """
    try:
        import ipynbname

        return str(ipynbname.path())
    except Exception:
        return None


def set_notebook_path(path: str) -> str:
    """
    Stores the notebook path to use for the rest of the kernel session.

    Args:
        path: path to the .ipynb the tools should analyze

    Returns:
        a confirmation line naming the stored path
    """
    global _notebook_path
    _notebook_path = path
    return f"provenance: notebook set to {path}"


def resolve_notebook_path() -> str:
    """
    Returns the notebook path the tools should analyze.

    Precedence: an override set by %provenance_notebook, then ipynbname
    auto-detection.

    Returns:
        the resolved notebook path

    Raises:
        UsageError: if no override is set and auto-detection fails
    """
    if _notebook_path:
        return _notebook_path

    detected = _autodetect_path()
    if detected:
        return detected

    raise UsageError(
        "Could not auto-detect the current notebook path (this is common in "
        "VSCode). Set it once with: %provenance_notebook <path.ipynb>"
    )


def _format_result(call: dict) -> str:
    """
    Renders one executed tool call as text for the cell output.

    Args:
        call: a {"name", "args", "result"} entry from agent.run()

    Returns:
        the text to print for that call
    """
    notebook = call["args"].get("notebook_path", "the notebook")
    result = call["result"]

    if call["name"] == "cite_data":
        if not result:
            return f"No datasets detected in {notebook}.\n{_DISK_NOTE}"
        lines = "\n".join(f"  - {variable} ({tool})" for variable, tool in result)
        return (
            f"Injected {len(result)} retrieval cell(s) into {notebook}:\n"
            f"{lines}\n\n"
            "Reload the notebook and run the new cells to produce the citations."
        )

    if call["name"] == "cite_software":
        if not result:
            return f"No software libraries found in {notebook}.\n{_DISK_NOTE}"
        lines = "\n".join(f"  - {lib}" for lib in result)
        noun = "library" if len(result) == 1 else "libraries"
        return (
            f"Injected a metadata cell into {notebook} for {len(result)} {noun}:\n"
            f"{lines}\n\n"
            "Reload the notebook and run the new cell to see the citation "
            "metadata DataFrame."
        )

    if not str(result).strip():
        return f"No citations found for {notebook}.\n{_DISK_NOTE}"
    return str(result)


def _format_results(calls: list[dict]) -> str:
    """
    Renders every executed tool call, or explains an unroutable request.

    Args:
        calls: the list agent.run() returned

    Returns:
        the full cell output text
    """
    if not calls:
        return _NO_ROUTE
    return "\n\n".join(_format_result(call) for call in calls)


def cite(request: str) -> str:
    """
    Routes a natural-language citation request and renders the result.

    Args:
        request: the user's request, e.g. "cite the software"

    Returns:
        the text to print in the cell

    Raises:
        UsageError: if the request is empty or the notebook path is unresolvable
    """
    request = request.strip()
    if not request:
        raise UsageError(
            "Usage: %provenance <request>, e.g. %provenance cite the software"
        )
    return _format_results(run(request, resolve_notebook_path()))


@magics_class
class ProvenanceMagics(Magics):
    """The %provenance and %provenance_notebook line magics."""

    @line_magic
    def provenance(self, line: str) -> None:
        """Routes a citation request and prints the result in the cell."""
        print(cite(line))

    @line_magic
    def provenance_notebook(self, line: str) -> None:
        """Sets the notebook path to use when auto-detection fails."""
        path = line.strip()
        if not path:
            raise UsageError("Usage: %provenance_notebook <path.ipynb>")
        print(set_notebook_path(path))


def load_ipython_extension(ipython) -> None:
    """
    Registers the magics. Called by %load_ext provenance.

    Args:
        ipython: the active InteractiveShell
    """
    ipython.register_magics(ProvenanceMagics)
