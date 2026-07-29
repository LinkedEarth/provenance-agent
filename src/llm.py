"""
Shared Gemini client for the whole project.

Purpose:
    One place to configure the LLM and credentials. Every LLM call routes through
    the single `llm` object defined here, so the model, temperature, and API key
    live in exactly one spot.

Consumers:
    - dataset_detection.py imports `llm` for LLM-based dataset detection.
    - bibliography.py imports `bibtex_to_apa` for APA rendering.
    - agent.py imports `llm` for the LCEL classification stage.

Implementation:
    - Loads GOOGLE_API_KEY from src/.env via dotenv.
    - `llm`: a ChatGoogleGenerativeAI client (temperature=0 for determinism).
    - `message_text(message)`: normalizes a response's content to plain text
      (Gemini may return a string or a list of typed content parts).
    - `bibtex_to_apa(bibtex)`: a prompt | llm chain that converts one BibTeX
      entry to an APA 7th edition string.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# gemini-flash-latest is an alias that tracks the current Gemini Flash model, so
# a specific version being retired does not 404 us (gemini-2.5-flash was retired).
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

bibtex_to_apa_prompt = ChatPromptTemplate.from_template(
    "Convert this BibTeX entry to APA 7th edition format. "
    "Return only the formatted citation, nothing else.\n\n{bibtex}"
)

bibtex_to_apa_chain = bibtex_to_apa_prompt | llm


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


def bibtex_to_apa(bibtex: str) -> str:
    """Converts a BibTeX entry to an APA 7th edition citation string."""
    response = bibtex_to_apa_chain.invoke({"bibtex": bibtex})
    return message_text(response)
