"""
Dataset detection facade for the data workflow.

Purpose:
    Identify which notebook variables hold source-backed datasets used for
    analysis or terminal tabular results, so the data workflow can inject a
    retrieval cell ({var}.{method}) for each. The active detector is the
    deterministic, AST/data-flow analyzer in deterministic_dataset_detection.py.

Implementation:
    - detect_datasets(notebook_path): delegates to the deterministic detector
      and returns the existing [variable, tool] pair contract. This is the
      active path.
    - DETECTION_PROMPT, build_detection_prompt(), and
      parse_detection_response(): DEPRECATED. They are the retained LLM
      detection path, kept as a rollback option and never called by active
      detection. See below.

Deprecated LLM fallback:
    Detection used to send the notebook's code to Gemini and parse a JSON list
    of pairs out of the reply. That path is deprecated but deliberately intact,
    so the project can switch back without reconstructing it. It consists of
    DETECTION_PROMPT, build_detection_prompt(), _strip_code_fences(),
    parse_detection_response(), and the commented-out call inside
    detect_datasets(). Restoring it means uncommenting that call; nothing else
    references these helpers, they emit no deprecation warning, and their tests
    are kept so the fallback stays known-good.

Design decisions:
    - The public active detector accepts a notebook path so the static analyzer
      can preserve notebook cell boundaries and ignore generated cells. The
      deprecated LLM path took concatenated code instead, which is why the
      commented-out call reads differently from the active one.
    - detect_datasets() preserves its list return contract while emitting
      UserWarning messages for unresolved analysis source lineage. Callers that
      need structured diagnostics can use detect_datasets_with_diagnostics().
    - The prompt is spliced with str.replace("{code}", ...) rather than
      str.format() so notebook code containing braces can never break templating.
"""

import json
import warnings


# DEPRECATED: the prompt for the retained LLM fallback. Active detection is
# deterministic and never sends this to a model.
DETECTION_PROMPT = """Analyze a Jupyter notebook to identify which dataset source variables are actually used for
scientific inquiry and analysis that would yield citations. For each dataset source in the notebook,
trace the data flow through the notebook and identify the final variable that holds the data actually
used for analysis, along with the software library that it is using.
A dataset source variable is a variable that directly loads or queries external data and could be one of these:
- PyLiPD: LiPD() objects that load data via load(), load_from_dir(), or load_remote_datasets()
- PyleoTUPS: PangaeaDataset() or NOAADataset() objects that search for studies via search_studies()
- LiPDGraph: SPARQL queries sent via requests.post() to the LinkedEarth endpoint
  (linkedearth.graphdb.mint.isi.edu), with results parsed into a DataFrame via pd.read_csv()
Other libraries like xarray, intake, cfr, pandas, or requests may also load external data.
Only return variables that hold dataset objects; They should be the same type as the source variable.
Return ONLY a JSON list of [variable, tool] pairs. No explanation.
Example: [["D", "PyLiPD"], ["ds", "PyleoTUPS"], ["df_res", "LiPDGraph"], ["iso_ds", "xarray"], ["intcal20", "pandas"]]
Walk through this context in manageable parts step by step, analyzing as you go.
Notebook code:
{code}"""


def build_detection_prompt(code: str) -> str:
    """
    DEPRECATED. Fills the detection prompt template with the notebook code.

    Part of the retained LLM fallback; active detection is deterministic and
    never calls this. See the module docstring.

    Args:
        code: the notebook's Python source (all code cells concatenated)

    Returns:
        the complete prompt string to send to the LLM
    """
    return DETECTION_PROMPT.replace("{code}", code)


def _strip_code_fences(text: str) -> str:
    """Removes a leading ```/```json fence and trailing ``` fence, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]  # drop the opening ``` or ```json line
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def parse_detection_response(text: str) -> list[list[str]]:
    """
    DEPRECATED. Extracts the JSON list of [variable, tool] pairs from the
    model's reply.

    Part of the retained LLM fallback; active detection is deterministic and
    never calls this. See the module docstring.

    Tolerates markdown code fences and surrounding prose by parsing the first
    JSON array found in the text. Only well-formed 2-element pairs are kept;
    anything unparseable yields an empty list so the caller can degrade
    gracefully instead of crashing.

    Args:
        text: raw text returned by the LLM

    Returns:
        list of [variable, tool] string pairs (possibly empty)
    """
    cleaned = _strip_code_fences(text)

    candidates = [cleaned]
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        return [
            [str(item[0]), str(item[1])]
            for item in data
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]

    return []


def detect_datasets(notebook_path: str) -> list[list[str]]:
    """
    Detects source-backed dataset variables in a notebook deterministically.

    Args:
        notebook_path: path to the target .ipynb file

    Returns:
        deterministic list of [variable, tool] pairs
    """
    # DEPRECATED LLM fallback, retained so the project can switch back. To
    # restore it, read the notebook's code with
    # .notebook_parser.read_notebook_code(notebook_path) and run:
    # from .llm import llm, message_text
    # response = llm.invoke(build_detection_prompt(code))
    # return parse_detection_response(message_text(response))

    diagnostics = detect_datasets_with_diagnostics(notebook_path)
    for message in diagnostics["warnings"]:
        warnings.warn(message, UserWarning, stacklevel=2)
    return diagnostics["pairs"]


def detect_datasets_with_diagnostics(notebook_path: str) -> dict[str, list]:
    """
    Detects dataset pairs and returns unresolved-analysis diagnostics.

    Args:
        notebook_path: path to the target .ipynb file

    Returns:
        a mapping with ``pairs`` and ``warnings`` keys
    """
    from .deterministic_dataset_detection import detect_datasets_with_diagnostics

    return detect_datasets_with_diagnostics(notebook_path)


def detect_datasets_in_notebook(path: str) -> list[list[str]]:
    """
    Compatibility alias for detect_datasets().

    Args:
        path: path to the target .ipynb file

    Returns:
        deterministic list of [variable, tool] pairs
    """
    return detect_datasets(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m provenance_agent.dataset_detection <path_to_notebook.ipynb>")
        sys.exit(1)
    for variable, tool in detect_datasets_in_notebook(sys.argv[1]):
        print(f"{variable}\t{tool}")
