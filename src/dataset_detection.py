"""
LLM-based dataset detection for the data workflow.

Purpose:
    Identify which notebook variables hold the datasets actually used for
    analysis, so the data workflow can inject a retrieval cell ({var}.{method})
    for each. Detection is done by an LLM (Gemini via LangChain), NOT by AST
    parsing, because tracing data flow to the *terminal* analysis variable
    across many cells is far more robust with an LLM than with static analysis.

Implementation:
    - DETECTION_PROMPT: the prompt template, kept verbatim as the single source
      of truth (it mirrors the "Dataset detection prompt" documented in
      CLAUDE.md). The notebook code is spliced in at the "{code}" marker.
    - build_detection_prompt(code): fills the template with the notebook code.
    - parse_detection_response(text): extracts the JSON [variable, tool] list
      from the model's reply, tolerating markdown code fences and surrounding
      prose, and returns [] on anything unparseable.
    - detect_datasets(code): thin wrapper that invokes the Gemini chat model and
      returns the parsed pairs.
    - detect_datasets_in_notebook(path): reads a .ipynb into code, then detects.

Design decisions:
    - Reuses the Gemini model configured in src/llm.py (gemini-2.5-flash,
      temperature=0) so detection is as deterministic as the API allows and the
      credential/config lives in one place. That import is deferred into
      detect_datasets() so the pure helpers can be imported and unit-tested
      without a GOOGLE_API_KEY or network access.
    - The prompt is spliced with str.replace("{code}", ...) rather than
      str.format() so notebook code containing braces can never break templating.
"""

import json


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
    Fills the detection prompt template with the notebook code.

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
    Extracts the JSON list of [variable, tool] pairs from the model's reply.

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


def detect_datasets(code: str) -> list[list[str]]:
    """
    Detects dataset source variables in notebook code via the Gemini LLM.

    Args:
        code: the notebook's Python source (all code cells concatenated)

    Returns:
        list of [variable, tool] pairs identifying the terminal variables that
        hold the datasets used for analysis
    """
    from llm import llm

    response = llm.invoke(build_detection_prompt(code))
    return parse_detection_response(response.content)


def detect_datasets_in_notebook(path: str) -> list[list[str]]:
    """
    Reads a notebook file and detects its dataset source variables.

    Args:
        path: path to a .ipynb file

    Returns:
        list of [variable, tool] pairs (see detect_datasets)
    """
    from notebook_parser import read_notebook_code

    return detect_datasets(read_notebook_code(path))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python dataset_detection.py <path_to_notebook.ipynb>")
        sys.exit(1)
    for variable, tool in detect_datasets_in_notebook(sys.argv[1]):
        print(f"{variable}\t{tool}")
