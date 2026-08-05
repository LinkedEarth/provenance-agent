"""
Stable ``%load_ext provenance`` entry point.

Purpose:
    ``%load_ext`` takes a top-level module name, and ``%load_ext provenance`` is
    the notebook-facing command this project has always documented. Keeping that
    command working means keeping an importable top-level ``provenance`` module,
    even though the implementation now lives inside the ``provenance_agent``
    package.

Implementation:
    A forwarding shim, and nothing else. It re-exports
    ``load_ipython_extension`` - the function IPython calls - plus the public
    helpers the magic API has always exposed (``cite``, ``set_notebook_path``,
    ``resolve_notebook_path``, ``ProvenanceMagics``) from
    ``provenance_agent.magic``.

Design decisions:
    - This is the one intentional top-level module outside the package. It is
      installed via ``py-modules = ["provenance"]`` in ``pyproject.toml``.
    - It holds no second implementation. Every name here is the same object as
      the one in ``provenance_agent.magic``, so behavior cannot drift between
      the shim and the module, and there is only one place to change the magic.
    - It uses the absolute package name because it is outside the package;
      modules inside ``provenance_agent`` import each other relatively.
    - ``set_notebook_path`` mutates module state in ``provenance_agent.magic``,
      which is the module the magic itself reads, so calling it through this
      shim and calling it through the package affect the same session override.
"""

from provenance_agent.magic import (
    ProvenanceMagics,
    cite,
    load_ipython_extension,
    resolve_notebook_path,
    set_notebook_path,
)

__all__ = [
    "ProvenanceMagics",
    "cite",
    "load_ipython_extension",
    "resolve_notebook_path",
    "set_notebook_path",
]
