"""
test_notebook_parser.py

Purpose:
    Unit and integration tests for notebook_parser.py covering all three public
    functions: strip_ipython_directives, extract_libraries, and parse_notebook.

Implementation:
    Unit tests for strip_ipython_directives and extract_libraries operate on raw
    strings with no file I/O. Integration tests for parse_notebook use fixture
    notebooks from notebooks/: sample.ipynb (normal notebook) and
    test_magic_commands.ipynb (cells with %, !, and %% directives).

Design Decisions:
    - Unit tests are file-independent so they run fast and in isolation.
    - Each edge case (magic commands, shell lines, syntax errors, nested imports)
      gets its own test so failures pinpoint the exact issue.
    - Filtering standard library names (os, sys) is not notebook_parser's job,
      so those are included in edge case tests without asserting they are absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from notebook_parser import extract_libraries, parse_notebook, strip_ipython_directives

SAMPLE     = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sample.ipynb")
MAGIC_NB   = os.path.join(os.path.dirname(__file__), "..", "notebooks", "test_magic_commands.ipynb")


# ---------------------------------------------------------------------------
# strip_ipython_directives
# ---------------------------------------------------------------------------

def test_strip_removes_percent_magic():
    result = strip_ipython_directives("%matplotlib inline\nimport numpy as np")
    assert "%matplotlib" not in result
    assert "import numpy as np" in result


def test_strip_removes_shell_command():
    result = strip_ipython_directives("!pip install pyleoclim\nimport pyleoclim")
    assert "!pip" not in result
    assert "import pyleoclim" in result


def test_strip_preserves_normal_code():
    code = "import numpy as np\nx = 1"
    assert strip_ipython_directives(code) == code


def test_strip_handles_indented_magic():
    result = strip_ipython_directives("    %timeit x = 1\nimport os")
    assert "%timeit" not in result
    assert "import os" in result


def test_strip_empty_string():
    assert strip_ipython_directives("") == ""


# ---------------------------------------------------------------------------
# extract_libraries
# ---------------------------------------------------------------------------

def test_extract_simple_import():
    assert extract_libraries("import numpy") == {"numpy"}


def test_extract_import_as():
    assert extract_libraries("import numpy as np") == {"numpy"}


def test_extract_from_import():
    assert extract_libraries("from matplotlib.pyplot import plt") == {"matplotlib"}


def test_extract_submodule_import():
    assert extract_libraries("import scipy.stats") == {"scipy"}


def test_extract_multiple_imports():
    code = "import numpy as np\nimport pandas as pd\nfrom matplotlib import pyplot"
    assert extract_libraries(code) == {"numpy", "pandas", "matplotlib"}


def test_extract_ignores_magic_after_strip():
    assert extract_libraries("%matplotlib inline\nimport numpy as np") == {"numpy"}


def test_extract_ignores_shell_after_strip():
    assert extract_libraries("!pip install pyleoclim\nimport pyleoclim") == {"pyleoclim"}


def test_extract_syntax_error_returns_empty():
    assert extract_libraries("def broken(:\n    pass") == set()


def test_extract_empty_cell_returns_empty():
    assert extract_libraries("") == set()


def test_extract_no_imports_returns_empty():
    assert extract_libraries("x = 1 + 2\nprint(x)") == set()


def test_extract_import_inside_function():
    code = "def foo():\n    import os\n    return os.getcwd()"
    assert "os" in extract_libraries(code)


# ---------------------------------------------------------------------------
# parse_notebook — sample.ipynb
# ---------------------------------------------------------------------------

def test_parse_returns_dict():
    assert isinstance(parse_notebook(SAMPLE), dict)


def test_parse_has_expected_keys():
    result = parse_notebook(SAMPLE)
    assert "cell_types" in result and "libraries" in result


def test_parse_cell_types_correct():
    result = parse_notebook(SAMPLE)
    assert result["cell_types"] == ["markdown", "code", "markdown", "code", "markdown", "code"]


def test_parse_libraries_is_sorted_list():
    libs = parse_notebook(SAMPLE)["libraries"]
    assert isinstance(libs, list)
    assert libs == sorted(libs)


def test_parse_libraries_contains_expected():
    libs = parse_notebook(SAMPLE)["libraries"]
    for expected in ("numpy", "pandas", "matplotlib", "pyleoclim"):
        assert expected in libs


# ---------------------------------------------------------------------------
# parse_notebook — test_magic_commands.ipynb
# ---------------------------------------------------------------------------

def test_parse_magic_notebook_has_correct_cell_types():
    result = parse_notebook(MAGIC_NB)
    assert result["cell_types"] == ["markdown", "code", "code", "code"]


def test_parse_magic_notebook_extracts_imports_despite_magic():
    libs = parse_notebook(MAGIC_NB)["libraries"]
    for expected in ("numpy", "pandas", "pyleoclim", "scipy"):
        assert expected in libs, f"expected '{expected}' in libraries but got {libs}"
