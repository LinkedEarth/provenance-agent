"""
Public package root for the provenance agent.

Purpose:
    Give the two direct workflow functions one canonical import path:

        from provenance_agent import cite_data, cite_software

    Everything else in the project is reached through its own module, so a
    caller only pays for what it uses.

Implementation:
    Re-exports ``cite_data`` from ``.data`` and ``cite_software`` from
    ``.software`` and names them in ``__all__``. No logic lives here.

Design decisions:
    - The LangChain tools and the LCEL router are deliberately NOT re-exported.
      Re-exporting ``run`` here would put the routing layer, its LangChain
      dependencies, and its provider configuration on the import path of every
      caller that only wants the two direct functions. They stay at
      ``provenance_agent.agent.run``, ``provenance_agent.data.cite_data_tool``,
      and ``provenance_agent.software.cite_software_tool``.
      Note that importing the agent no longer costs credentials either - the
      chat client is built on first use, not at import - so this separation is
      about dependency weight rather than about credentials.
    - Each function is imported from the module that implements it, not from a
      routing module in between. There is no longer an ``orchestrator`` layer to
      pass through.
"""

from .data import cite_data
from .software import cite_software

__all__ = ["cite_data", "cite_software"]
