"""
HTTP client for delegating citation retrieval to PaleoPAL agents.

The provenance agent identifies datasets in a notebook (via parse_notebook)
but does not execute code or SPARQL itself. Instead, it sends structured
requests to PaleoPAL's Code Agent and SPARQL Agent through the REST API
at /api/agents/request.

Flow:
  1. Create a conversation via POST /api/conversations/
  2. For each dataset action from parse_notebook(), send a request to the
     appropriate agent (code or sparql) via POST /api/agents/request
  3. Collect and return the citation results

The base URL defaults to http://localhost:8000 but can be overridden
via the PALEOPAL_API_URL environment variable.
"""

import os
import requests


_BASE_URL = os.environ.get("PALEOPAL_API_URL", "http://localhost:8000")


def create_conversation(
    agent_type: str = "code",
    title: str = "Provenance Agent Citation Retrieval",
) -> str:
    """
    Creates a new conversation in PaleoPAL and returns its ID.

    Args:
        agent_type: which agent this conversation is for ("code" or "sparql")
        title: human-readable conversation title

    Returns:
        the conversation ID string
    """
    resp = requests.post(
        f"{_BASE_URL}/api/conversations/",
        json={
            "title": title,
            "selected_agent": agent_type,
            "enable_clarification": False,
            "enable_execution": True,
        },
    )
    resp.raise_for_status()
    return resp.json()["id"]


def send_agent_request(
    agent_type: str,
    capability: str,
    user_input: str,
    conversation_id: str,
    context: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Sends a prompt to a PaleoPAL agent and returns the response.

    Args:
        agent_type: "code" or "sparql"
        capability: "generate_code" or "generate_sparql"
        user_input: the natural-language prompt for the agent
        conversation_id: ID from create_conversation()
        context: optional additional context dict
        metadata: optional metadata dict

    Returns:
        the full JSON response from the agent
    """
    resp = requests.post(
        f"{_BASE_URL}/api/agents/request",
        json={
            "agent_type": agent_type,
            "capability": capability,
            "conversation_id": conversation_id,
            "user_input": user_input,
            "context": context or {},
            "notebook_context": {},
            "metadata": metadata or {},
        },
    )
    resp.raise_for_status()
    return resp.json()


def request_dataset_citations(datasets: list[dict]) -> list[dict]:
    """
    Sends each dataset action from parse_notebook() to the appropriate
    PaleoPAL agent and collects the responses.

    Creates one conversation per agent type to keep context grouped.

    Args:
        datasets: the "datasets" list from parse_notebook() output,
            each with keys: variable, source_type, agent, action/endpoint

    Returns:
        list of dicts, each with:
            - variable: the dataset variable name
            - source_type: "PyLiPD", "PyleoTUPS", or "LiPDGraph"
            - agent_response: the raw response dict from the agent
    """
    if not datasets:
        return []

    conversations: dict[str, str] = {}
    results = []

    for ds in datasets:
        agent_type = ds["agent"]

        if agent_type not in conversations:
            conversations[agent_type] = create_conversation(
                agent_type=agent_type,
                title=f"Provenance: {agent_type} citations",
            )

        conversation_id = conversations[agent_type]

        if agent_type == "code":
            prompt = _build_code_prompt(ds)
            capability = "generate_code"
        elif agent_type == "sparql":
            prompt = _build_sparql_prompt(ds)
            capability = "generate_sparql"
        else:
            continue

        response = send_agent_request(
            agent_type=agent_type,
            capability=capability,
            user_input=prompt,
            conversation_id=conversation_id,
        )

        results.append({
            "variable": ds["variable"],
            "source_type": ds["source_type"],
            "agent_response": response,
        })

    return results


def _build_code_prompt(dataset: dict) -> str:
    """
    Builds a natural-language prompt for the Code Agent to retrieve
    citations for a PyLiPD or PyleoTUPS dataset.

    Args:
        dataset: a single dataset action dict from parse_notebook()

    Returns:
        prompt string for the Code Agent
    """
    action = dataset["action"]
    source = dataset["source_type"]
    var = dataset["variable"]

    if source == "PyLiPD":
        return (
            f"The variable `{var}` is a LiPD object. "
            f"Run `{action}` to get its BibTeX citations and return the result."
        )
    elif source == "PyleoTUPS":
        return (
            f"The variable `{var}` is a PyleoTUPS object. "
            f"Run `{action}` to get its publication information and return the result."
        )
    else:
        return f"Run `{action}` and return the result."


def _build_sparql_prompt(dataset: dict) -> str:
    """
    Builds a natural-language prompt for the SPARQL Agent to retrieve
    citations for datasets found via LiPDGraph.

    Args:
        dataset: a single dataset action dict from parse_notebook()

    Returns:
        prompt string for the SPARQL Agent
    """
    endpoint = dataset["endpoint"]
    var = dataset.get("variable", "the query results")

    return (
        f"Query the LinkedEarth graph database at {endpoint} to retrieve "
        f"bibliography/publication information for the datasets in `{var}`. "
        f"Use the QUERY_BIBLIO query to get BibTeX citation data."
    )
