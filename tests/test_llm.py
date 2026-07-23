"""
Unit tests for llm.py's offline pieces.

The Gemini client itself needs credentials and network, so these tests cover
only message_text(), the normalizer that turns a LangChain AI message into
plain text. Gemini responses arrive either as a plain string or as a list of
typed content parts (newer langchain-google-genai versions return the list
form), and both shapes must normalize to the same text. Fake messages are
built with SimpleNamespace so no model call happens.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm import message_text


def test_message_text_plain_string():
    assert message_text(SimpleNamespace(content="hello")) == "hello"


def test_message_text_list_of_text_parts():
    msg = SimpleNamespace(content=[
        {"type": "text", "text": '[["df", "LiPDGraph"]]', "extras": {"signature": "abc"}},
    ])
    assert message_text(msg) == '[["df", "LiPDGraph"]]'


def test_message_text_joins_multiple_parts_and_skips_non_text():
    msg = SimpleNamespace(content=[
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": "part one, "},
        "part two",
    ])
    assert message_text(msg) == "part one, part two"
