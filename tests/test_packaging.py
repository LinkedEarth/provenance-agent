"""
Tests for the installed package boundary rather than for any one module's logic.

Purpose:
    Everything here is a property of the *installation*: that the package
    imports from outside the checkout, that the citation data travels with it
    as package data, that importing the direct API does not drag in the agent
    and its LLM client, that credentials are discovered instead of shipped, and
    that no secret ends up in the built artifact. None of these can be observed
    from a normal in-repo import, because the repository root supplies the
    modules and the data files whether or not the install is correct.

Implementation:
    Most tests run a short program in a subprocess whose working directory is
    tmp_path and whose environment has PYTHONPATH stripped. That is the whole
    point: from there the only way `import provenance_agent` can succeed is the
    editable install, so a missing entry in pyproject.toml fails the test
    instead of being masked by the checkout. `_run_isolated` is the helper for
    that. The artifact test builds a real wheel with `pip wheel --no-deps
    --no-build-isolation` and reads its manifest with zipfile.

Design decisions:
    - Subprocesses, not monkeypatched sys.path. Import side effects (the eager
      Gemini client, the dotenv load) happen once per process, so the only
      honest way to assert on "what a fresh interpreter sees" is a fresh
      interpreter.
    - `_run_isolated` writes the snippet to a *file* and runs that file, rather
      than passing it with `python -c`. This is not cosmetic. python-dotenv
      decides between its two search strategies by asking whether `__main__`
      has a `__file__`, and `-c` has none, so `-c` silently takes the same
      cwd-only branch a Jupyter kernel takes and the source-tree fallback can
      never be reached. Running a real script is what an installed consumer
      does and is the only way these tests observe both branches.
    - The script is written outside the working directory, so `sys.path[0]` is
      the script's directory and the tests can place files in cwd without
      making them importable.
    - The credential tests use fake keys written into tmp_path. The client is
      constructed at import time and validates that *a* key exists, not that it
      works, so a fake value exercises the whole lookup without a model call.
      Only the resolved path is ever asserted on, never a real key's value.
    - The notebook-shaped case (a .env in an ancestor of the working directory)
      is tested against a synthetic directory tree rather than this checkout,
      so it holds on a clean clone. The one test that does depend on the real
      src/.env skips when it is absent: that file is developer-local and
      untracked by design, so a failure there would be reporting the absence of
      a secret, not a defect.
    - The magic-shim import test supplies a fake key. Importing the shim pulls
      in the agent and its eagerly constructed client by design, and this test
      is about whether the module is installed and reachable, not about where
      credentials come from - the credential tests below cover that.
    - The artifact test asserts on the absence of any dotenv file anywhere in
      the wheel, not just at the two paths that exist today, so a future
      packaging change that sweeps one in still fails.
"""

import glob
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Enough to satisfy the eager client's "a key is set" check. It is never sent
# anywhere: no test in this file makes a model call.
FAKE_KEY = "not-a-real-key"


def _run_isolated(code: str, cwd, extra_env: dict | None = None,
                  drop_env: tuple[str, ...] = ()) -> str:
    """
    Runs a snippet in a fresh interpreter that cannot see the repository.

    The snippet is written to a script outside `cwd` and run as a file, so
    sys.path[0] is that script's directory and `__main__` has a `__file__` -
    the shape a real installed consumer has, and the one python-dotenv needs to
    see before it will consider its source-tree search. PYTHONPATH is stripped
    so the only route to `provenance_agent` is the installed distribution.

    Args:
        code: the Python source to run
        cwd: the working directory to run it from
        extra_env: environment variables to set for the subprocess
        drop_env: environment variables to remove for the subprocess

    Returns:
        the subprocess's stdout, stripped

    Raises:
        AssertionError: if the subprocess exits non-zero, with its stderr
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for name in drop_env:
        env.pop(name, None)
    env.update(extra_env or {})

    with tempfile.TemporaryDirectory() as script_dir:
        script = Path(script_dir) / "isolated_check.py"
        script.write_text(code)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# --- import boundary ---------------------------------------------------------

def test_package_imports_from_outside_the_repository(tmp_path):
    out = _run_isolated(
        "import provenance_agent, os;"
        "print(os.path.dirname(provenance_agent.__file__))",
        cwd=tmp_path,
    )
    assert out.endswith(os.path.join("src", "provenance_agent"))


def test_direct_api_does_not_import_the_agent(tmp_path):
    out = _run_isolated(
        "import sys;"
        "from provenance_agent import cite_data, cite_software;"
        "print('provenance_agent.agent' in sys.modules,"
        " 'provenance_agent.llm' in sys.modules)",
        cwd=tmp_path,
    )
    assert out == "False False"


def test_magic_shim_is_importable_from_outside_the_repository(tmp_path):
    out = _run_isolated(
        "import provenance\n"
        "shell = type('S', (), {'register_magics': lambda self, m: print(m.__name__)})()\n"
        "provenance.load_ipython_extension(shell)\n",
        cwd=tmp_path,
        extra_env={"GOOGLE_API_KEY": FAKE_KEY},
    )
    assert out == "ProvenanceMagics"


# --- packaged citation data --------------------------------------------------

def test_citation_lookup_uses_packaged_resources_from_outside_the_repository(tmp_path):
    out = _run_isolated(
        "from provenance_agent.bibliography import collect_library_entries;"
        "df = collect_library_entries(['pyleoclim']);"
        "print(sorted(df['citation_type']))",
        cwd=tmp_path,
    )
    assert out == "['paper', 'software']"


def test_citation_index_loads_from_outside_the_repository(tmp_path):
    out = _run_isolated(
        "from provenance_agent.bibliography import load_citation_index;"
        "print('pyleoclim' in load_citation_index())",
        cwd=tmp_path,
    )
    assert out == "True"


# --- credential discovery ----------------------------------------------------

def test_working_directory_dotenv_is_discovered(tmp_path):
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=from-working-directory\n")
    out = _run_isolated(
        "import os, provenance_agent.llm as m\n"
        "print(m.dotenv_path == os.path.abspath('.env'), os.environ['GOOGLE_API_KEY'])\n",
        cwd=tmp_path,
        drop_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    assert out == "True from-working-directory"


def test_ancestor_dotenv_is_discovered_from_a_subdirectory(tmp_path):
    """The notebook case: the kernel's cwd is a subdirectory of the project."""
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=from-project-root\n")
    notebook_dir = tmp_path / "notebooks" / "demos"
    notebook_dir.mkdir(parents=True)

    out = _run_isolated(
        "import os, provenance_agent.llm\nprint(os.environ['GOOGLE_API_KEY'])\n",
        cwd=notebook_dir,
        drop_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    assert out == "from-project-root"


def test_environment_variable_beats_the_dotenv_file(tmp_path):
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=from-dotenv\n")
    out = _run_isolated(
        "import os, provenance_agent.llm\nprint(os.environ['GOOGLE_API_KEY'])\n",
        cwd=tmp_path,
        extra_env={"GOOGLE_API_KEY": "from-environment"},
    )
    assert out == "from-environment"


def test_source_tree_dotenv_is_the_fallback_when_the_cwd_has_none(tmp_path):
    checkout_dotenv = REPO_ROOT / "src" / ".env"
    if not checkout_dotenv.exists():
        pytest.skip("no src/.env in this checkout; the fallback has nothing to find")

    out = _run_isolated(
        "import provenance_agent.llm as m\nprint(m.dotenv_path)\n",
        cwd=tmp_path,
        drop_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    assert Path(out) == checkout_dotenv


# --- built artifact ----------------------------------------------------------

@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> zipfile.ZipFile:
    """Builds the distribution once and hands back the wheel for inspection."""
    out_dir = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--no-build-isolation", "-w", str(out_dir), str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    wheels = glob.glob(str(out_dir / "*.whl"))
    assert len(wheels) == 1, wheels
    return zipfile.ZipFile(wheels[0])


def test_artifact_ships_the_citation_data(built_wheel):
    citations = [
        name for name in built_wheel.namelist()
        if name.startswith("provenance_agent/Citations/")
    ]
    assert "provenance_agent/Citations/library_citations.yml" in citations
    assert sum(name.endswith(".bib") for name in citations) >= 10


def test_artifact_ships_the_magic_shim(built_wheel):
    assert "provenance.py" in built_wheel.namelist()


def test_artifact_contains_no_dotenv_file(built_wheel):
    secrets = [
        name for name in built_wheel.namelist()
        if Path(name).name == ".env" or name.endswith("/.env")
    ]
    assert secrets == []
