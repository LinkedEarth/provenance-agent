"""
Tests for the top-level `provenance` module - the `%load_ext provenance` shim.

Purpose:
    `%load_ext` resolves a top-level module name, so `%load_ext provenance` only
    keeps working while an importable top-level `provenance` module exists and
    exposes `load_ipython_extension`. The implementation lives in
    `provenance_agent.magic`; this module is the installed forwarding surface
    over it.

Implementation:
    Every test here works on the installed shim, never on the package module,
    so a shim regression cannot be masked by a passing implementation test.
    `test_magic.py` covers the behavior these names forward to. The extension
    registration test uses a fake shell object that only records what it was
    handed, so no IPython kernel and no model call is involved.

Design decisions:
    - Identity, not behavior, is what is asserted for the forwarded names.
      Re-testing the formatting and path-resolution behavior here would give
      the shim its own expectations, and the point of a shim is that it has
      none of its own: `provenance.cite is magic.cite` proves there is exactly
      one implementation, which no behavioral assertion can prove.
    - The session-override test is the exception, because `set_notebook_path`
      writes module state. Reaching it through the shim must mutate the same
      module the magic later reads, or `%provenance_notebook` would set a path
      that `%provenance` cannot see.
"""

import provenance
from provenance_agent import magic


class _FakeShell:
    """Stands in for an InteractiveShell, recording registered magics classes."""

    def __init__(self):
        self.registered = []

    def register_magics(self, magics_class) -> None:
        self.registered.append(magics_class)


def test_shim_is_importable_as_a_top_level_module():
    assert provenance.__name__ == "provenance"


def test_load_ipython_extension_registers_the_magics_through_the_shim():
    shell = _FakeShell()
    provenance.load_ipython_extension(shell)
    assert shell.registered == [magic.ProvenanceMagics]


def test_forwarded_names_are_the_implementation_not_a_copy():
    for name in ("cite", "set_notebook_path", "resolve_notebook_path",
                 "ProvenanceMagics", "load_ipython_extension"):
        assert getattr(provenance, name) is getattr(magic, name)


def test_magic_methods_are_reachable_through_the_shim():
    assert hasattr(provenance.ProvenanceMagics, "provenance")
    assert hasattr(provenance.ProvenanceMagics, "provenance_notebook")


def test_session_override_set_through_the_shim_is_what_the_magic_reads():
    original = magic._notebook_path
    try:
        provenance.set_notebook_path("explicit.ipynb")
        assert magic.resolve_notebook_path() == "explicit.ipynb"
    finally:
        magic._notebook_path = original
