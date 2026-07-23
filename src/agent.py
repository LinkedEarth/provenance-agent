"""
Natural-language routing agent over the two orchestrator tools.

Purpose:
    Let a user say "cite the software" or "cite the datasets" and have the model
    pick the right tool (cite_software / cite_data) and call it with the right
    arguments. This is the natural-language layer on top of orchestrator.py.

Implementation:
    - SYSTEM_PROMPT: the agent's routing instructions; {notebook_path} is spliced
      in so the model knows which notebook to hand to the chosen tool.
    - build_messages(request, notebook_path): builds the [system, human] message
      list. Pure - no model call.
    - route(request, notebook_path): binds the two StructuredTools to the Gemini
      model (native tool-calling) and returns the tool calls the model chose,
      as [{"name", "args"}], with notebook_path filled in if the model omitted it.
    - run(request, notebook_path): route, then execute each chosen tool, returning
      [{"name", "args", "result"}].

Design decisions:
    - Native tool-calling via llm.bind_tools, NOT a create_tool_calling_agent /
      AgentExecutor loop. The routing is single-step (one request -> one tool), so
      the model's structured tool_calls are all we need, and this avoids adding the
      `langchain` umbrella dependency (only langchain_core is installed). The tool
      DESCRIPTIONS in orchestrator.py are the routing signal; this module supplies
      the surrounding system prompt.
    - notebook_path is spliced with str.replace (not str.format) so a notebook path
      containing braces can never break templating.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from llm import llm
from orchestrator import cite_data_tool, cite_software_tool

_TOOLS = [cite_software_tool, cite_data_tool]
_TOOLS_BY_NAME = {t.name: t for t in _TOOLS}

SYSTEM_PROMPT = (
    "You generate citations for the software and datasets used in a Jupyter "
    "notebook. The notebook to analyze is at `{notebook_path}`. Choose the "
    "correct tool for the user's request and always pass notebook_path to it.\n"
    "- Use cite_software for libraries, packages, or software (e.g. 'cite the "
    "software', 'cite Pyleoclim').\n"
    "- Use cite_data for datasets from PyLiPD, PyleoTUPS, or LiPDGraph (e.g. "
    "'cite the data', 'cite the datasets').\n"
    "If the user names a specific library or dataset, pass it as the tool's "
    "filter argument (libraries / targets). Default the output format to APA "
    "unless the user explicitly asks for BibTeX."
)


def build_messages(request: str, notebook_path: str) -> list:
    """
    Builds the [system, human] messages for a routing request.

    Args:
        request: the user's natural-language request
        notebook_path: path to the notebook the tools should analyze

    Returns:
        a two-element list [SystemMessage, HumanMessage]
    """
    system = SYSTEM_PROMPT.replace("{notebook_path}", notebook_path)
    return [SystemMessage(system), HumanMessage(request)]


def route(request: str, notebook_path: str) -> list[dict]:
    """
    Asks the model which tool(s) to call for the request (no execution).

    Binds the two StructuredTools to the Gemini model and returns the tool calls
    the model chose. notebook_path is filled in when the model omits it, so the
    chosen tool always receives the notebook to analyze.

    Args:
        request: the user's natural-language request
        notebook_path: path to the notebook the tools should analyze

    Returns:
        list of {"name": tool_name, "args": {...}} chosen by the model
    """
    ai = llm.bind_tools(_TOOLS).invoke(build_messages(request, notebook_path))

    calls = []
    for tool_call in ai.tool_calls:
        args = dict(tool_call["args"])
        args.setdefault("notebook_path", notebook_path)
        calls.append({"name": tool_call["name"], "args": args})
    return calls


def run(request: str, notebook_path: str) -> list[dict]:
    """
    Routes a request to the correct tool(s) and executes them.

    Args:
        request: the user's natural-language request
        notebook_path: path to the notebook the tools should analyze

    Returns:
        list of {"name", "args", "result"} - one per executed tool. Unknown tool
        names are skipped.
    """
    results = []
    for call in route(request, notebook_path):
        tool = _TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            continue
        result = tool.invoke(call["args"])
        results.append({"name": call["name"], "args": call["args"], "result": result})
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print('Usage: python agent.py <notebook.ipynb> "<request>"')
        sys.exit(1)
    notebook, user_request = sys.argv[1], sys.argv[2]
    for r in run(user_request, notebook):
        print(f"# {r['name']}({r['args']})")
        print(r["result"])
        print()
