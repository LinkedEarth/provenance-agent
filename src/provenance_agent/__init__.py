"""
Public package root for the provenance agent.

Purpose:
    Give the two direct workflow functions one canonical import path:

        from provenance_agent import cite_data, cite_software

    Everything else in the project is reached through its own module, so a
    caller only pays for what it uses.

Implementation:
    Re-exports ``cite_data`` and ``cite_software`` from ``.orchestrator`` and
    names them in ``__all__``. No logic lives here.

Design decisions:
    - The LangChain tools and the LCEL router are deliberately NOT re-exported.
      ``provenance_agent.agent`` constructs the shared Gemini client at import
      time, so re-exporting ``run`` here would make ``import provenance_agent``
      require credentials even for callers that only want the direct functions.
      They stay at ``provenance_agent.agent.run``,
      ``provenance_agent.data.cite_data_tool``, and
      ``provenance_agent.software.cite_software_tool``.
    - ``.orchestrator`` is the temporary source of these two functions. The
      module-cleanup phase moves them into ``.data`` and ``.software`` and
      repoints this file, after which ``orchestrator.py`` is deleted.
"""

from .orchestrator import cite_data, cite_software

__all__ = ["cite_data", "cite_software"]
