"""
Notebook inputs for the test suite, built in code rather than checked in.

Purpose:
    Several test modules need a small notebook to parse, inject into, and route
    against. These used to be two files under ``notebooks/fixtures/``:
    ``sample.ipynb`` (a normal notebook) and ``test_magic_commands.ipynb``
    (cells carrying ``%``, ``!``, and ``%%`` directives). Both were deleted, so
    this module reconstructs them.

Implementation:
    ``_write`` builds an nbformat notebook from a list of ``(cell_type, source)``
    pairs and writes it into one session-lifetime temporary directory, cleaned up
    at interpreter exit. ``SAMPLE`` and ``MAGIC_NOTEBOOK`` are the resulting
    paths, built eagerly at import so importing modules can keep using them as
    plain module-level constants:

        from notebook_fixtures import SAMPLE

Design decisions:
    - Paths, not pytest fixtures. The consuming modules already treat these as
      module-level constants, several using them inside parametrize data where a
      fixture cannot reach. Keeping the contract a path is what made this a
      drop-in replacement instead of a rewrite of every test signature.
    - The content is reproduced deliberately, not approximated. ``SAMPLE``
      imports numpy, pandas, pyleoclim, and matplotlib because the tests assert
      on that exact library set, and ``MAGIC_NOTEBOOK`` keeps one cell per
      directive form because that is the distinction it exists to pin.
    - Built once per session rather than per test. Nothing mutates these files;
      the workflows under test all write to an ``output_path`` or a ``tmp_path``
      copy. ``test_software`` explicitly asserts the source notebook is
      unmodified, which would catch a violation of that assumption.
    - One temporary directory, removed by ``atexit``. No test artifact is left
      in the repository, which is what makes the checked-in fixtures
      unnecessary in the first place.
"""

import atexit
import tempfile
from pathlib import Path

import nbformat

_TEMPORARY_DIRECTORY = tempfile.TemporaryDirectory(prefix="provenance-notebook-fixtures-")
atexit.register(_TEMPORARY_DIRECTORY.cleanup)

_ROOT = Path(_TEMPORARY_DIRECTORY.name)


def _write(name: str, cells: list[tuple[str, str]]) -> str:
    """
    Builds one notebook and writes it into the session temporary directory.

    Args:
        name: file name to write, e.g. "sample.ipynb"
        cells: (cell_type, source) pairs, where cell_type is "code" or "markdown"

    Returns:
        the absolute path to the written notebook, as a string
    """
    notebook = nbformat.v4.new_notebook()
    for cell_type, source in cells:
        if cell_type == "code":
            notebook.cells.append(nbformat.v4.new_code_cell(source))
        else:
            notebook.cells.append(nbformat.v4.new_markdown_cell(source))

    path = _ROOT / name
    nbformat.write(notebook, str(path))
    return str(path)


_SAMPLE_CELLS = [
    ("markdown", "# Sample Paleoclimate Notebook\n"
                 "A minimal notebook for testing the provenance agent parser."),
    ("code", "import numpy as np\n"
             "import pandas as pd\n"
             "import pyleoclim as pyleo"),
    ("markdown", "## Load Data\nLoad a sample dataset for analysis."),
    ("code", "df = pd.DataFrame({'time': range(10), 'value': np.random.randn(10)})\n"
             "df.head()"),
    ("markdown", "## Plot\nVisualize the time series."),
    ("code", "import matplotlib.pyplot as plt\n"
             "\n"
             "plt.plot(df['time'], df['value'])\n"
             "plt.xlabel('Time')\n"
             "plt.ylabel('Value')\n"
             "plt.show()"),
    ("code", "np.array([1, 2, 3])"),
]

_MAGIC_CELLS = [
    ("markdown", "# Magic Command Test Notebook\n"
                 "Tests that imports are still extracted when cells contain "
                 "IPython magic or shell commands."),
    ("code", "%matplotlib inline\n"
             "import numpy as np\n"
             "import pandas as pd"),
    ("code", "!pip install pyleoclim\n"
             "import pyleoclim as pyleo"),
    ("code", "%%time\n"
             "import scipy.stats as stats\n"
             "x = stats.norm.rvs(size=100)"),
]

#: A normal notebook importing numpy, pandas, pyleoclim, and matplotlib.
SAMPLE = _write("sample.ipynb", _SAMPLE_CELLS)

#: A notebook whose cells carry %, !, and %% directives around real imports.
MAGIC_NOTEBOOK = _write("test_magic_commands.ipynb", _MAGIC_CELLS)
