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

import pytest

from provenance_agent.notebook_parser import extract_libraries, parse_notebook, strip_ipython_directives

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


def test_extract_recovers_imports_from_broken_cell():
    # Mirrors a real notebook cell (C02_b cell 18): the docstring is indented
    # 3 spaces but the body 4, an IndentationError - the imports above the
    # broken function must still be recovered.
    code = (
        "import cfr.psm as psm\n"
        "from tqdm import tqdm\n"
        "def f():\n"
        '   """doc"""\n'
        "    for x in y:\n"
        "        pass\n"
    )
    assert extract_libraries(code) == {"cfr", "tqdm"}


def test_extract_recovers_multi_import_line_from_broken_cell():
    assert extract_libraries("import os, json\ndef broken(:\n    pass") == {"os", "json"}


def test_extract_recovers_parenthesized_from_import_in_broken_cell():
    code = "from pylipd.lipd import (\n    LiPD,\n)\ndef broken(:\n    pass"
    assert extract_libraries(code) == {"pylipd"}


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

def test_parse_returns_sorted_list():
    libs = parse_notebook(SAMPLE)
    assert isinstance(libs, list)
    assert libs == sorted(libs)


def test_parse_libraries_contains_expected():
    libs = parse_notebook(SAMPLE)
    for expected in ("numpy", "pandas", "matplotlib", "pyleoclim"):
        assert expected in libs


# ---------------------------------------------------------------------------
# parse_notebook — test_magic_commands.ipynb
# ---------------------------------------------------------------------------

def test_parse_magic_notebook_extracts_imports_despite_magic():
    libs = parse_notebook(MAGIC_NB)
    for expected in ("numpy", "pandas", "pyleoclim", "scipy"):
        assert expected in libs, f"expected '{expected}' in libraries but got {libs}"


# --- injected cells are not the notebook's dependencies -----------------------

def test_is_generated_cell_recognizes_marker_and_legacy_signatures():
    from provenance_agent.notebook_parser import PROVENANCE_CELL_MARKER, is_generated_cell

    assert is_generated_cell(f"{PROVENANCE_CELL_MARKER}\nimport bibliography")
    assert is_generated_cell("_provbib_software = collect_library_entries([], None)")
    assert is_generated_cell("_provbib_data_D = _meta_D")
    assert is_generated_cell("# provenance-combine-cell\nimport pandas as pd")
    assert not is_generated_cell("import numpy as np\nx = np.array([1])")


def test_parse_notebook_ignores_injected_cells(tmp_path):
    """The software cell imports bibliography; a retrieval cell imports pylipd.

    Scanning them would cite the tool's own machinery, and pylipd has a
    citation on file, so the notebook would gain a citation it never earned.
    """
    import nbformat
    from provenance_agent.notebook_parser import parse_notebook

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("import numpy as np"))
    nb.cells.append(nbformat.v4.new_code_cell(
        "from bibliography import collect_library_entries\n"
        "_provbib_software = collect_library_entries(['numpy'], None)"
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
        "from pylipd.lipd import LiPD\n"
        "_lipd_df = LiPD()\n"
        "_provbib_data_df = _meta_df"
    ))
    path = tmp_path / "nb.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    assert parse_notebook(str(path)) == ["numpy"]


def test_read_notebook_code_ignores_injected_cells(tmp_path):
    """The detector must not mistake retrieval scaffolding for a dataset variable."""
    import nbformat
    from provenance_agent.notebook_parser import read_notebook_code

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("df_res = pd.read_csv(data)"))
    nb.cells.append(nbformat.v4.new_code_cell(
        "from pylipd.lipd import LiPD\n"
        "_lipd_df_res = LiPD()\n"
        "_provbib_data_df_res = _meta_df_res"
    ))
    path = tmp_path / "nb.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    code = read_notebook_code(str(path))
    assert "df_res = pd.read_csv(data)" in code
    assert "_lipd_df_res" not in code


def test_injected_cells_carry_the_marker():
    """New cells are recognized by the marker, not by inferring their contents."""
    from provenance_agent.notebook_parser import PROVENANCE_CELL_MARKER, is_generated_cell
    from provenance_agent.software_workflow import build_metadata_cell
    from provenance_agent.data_workflow import build_dataset_cell

    software = build_metadata_cell(["numpy"])
    # The marker belongs on the injected cell, not on build_retrieval_cell's
    # fragment, which never becomes a cell of its own.
    data = build_dataset_cell([["D", "PyLiPD"]])
    assert software.startswith(PROVENANCE_CELL_MARKER)
    assert data.startswith(PROVENANCE_CELL_MARKER)
    assert is_generated_cell(software) and is_generated_cell(data)


def test_repeated_software_runs_do_not_accumulate_self_citations(tmp_path):
    import nbformat
    from provenance_agent.software_workflow import generate_software_workflow

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("import numpy as np"))
    path = tmp_path / "nb.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    first = generate_software_workflow(str(path))
    second = generate_software_workflow(str(path))
    assert first == second == ["numpy"]
    assert "bibliography" not in second
