"""
Shared Gemini client for the whole project.

Purpose:
    One place to configure the LLM and credentials. Every LLM call routes through
    the single `llm` object defined here, so the model, temperature, and API key
    live in exactly one spot.

Consumers:
    - agent.py imports `llm` for the LCEL classification stage. This is the only
      active LLM call in the project.
    - dataset_detection.py's deprecated LLM fallback imports `llm` and
      `message_text`. That path is retained for rollback and is not called by
      active detection, which is deterministic.

Implementation:
    - Loads GOOGLE_API_KEY from src/.env via dotenv.
    - `llm`: a ChatGoogleGenerativeAI client (temperature=0 for determinism).
    - `message_text(message)`: normalizes a response's content to plain text
      (Gemini may return a string or a list of typed content parts).

Design decisions:
    - APA rendering has been removed. There is no `bibtex_to_apa` chain here
      anymore, because APA output is no longer produced by an LLM. `fmt` is
      still accepted by the data workflow but is ignored, so nothing in this
      module renders citation text.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# gemini-flash-latest is an alias that tracks the current Gemini Flash model, so
# a specific version being retired does not 404 us (gemini-2.5-flash was retired).
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)


def message_text(message) -> str:
    """
    Extracts the plain text of a LangChain AI message.

    Gemini responses arrive either as a plain string or as a list of typed
    content parts like {"type": "text", "text": ...} - newer
    langchain-google-genai versions return the list form (with thought
    signatures in "extras"). Text parts are joined; non-text parts (e.g.
    thinking) are dropped.

    Args:
        message: a LangChain message object with a `content` attribute

    Returns:
        the message's text content as one string
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)
