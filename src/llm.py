"""
LangChain agent wrapper for citation formatting. Uses Gemini to convert
BibTeX entries into APA 7th edition citations.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

bibtex_to_apa_prompt = ChatPromptTemplate.from_template(
    "Convert this BibTeX entry to APA 7th edition format. "
    "Return only the formatted citation, nothing else.\n\n{bibtex}"
)

bibtex_to_apa_chain = bibtex_to_apa_prompt | llm


def bibtex_to_apa(bibtex: str) -> str:
    """Converts a BibTeX entry to an APA 7th edition citation string."""
    response = bibtex_to_apa_chain.invoke({"bibtex": bibtex})
    return response.content
