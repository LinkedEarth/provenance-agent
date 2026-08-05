"""
Shared chat-model client for the whole project.

Purpose:
    One place to configure the LLM and credentials. Every LLM call routes
    through the single `llm` object defined here, so the provider, model,
    temperature, and API key live in exactly one spot.

Consumers:
    - agent.py imports `llm` for the LCEL classification stage. This is the only
      active LLM call in the project.
    - dataset_detection.py's deprecated LLM fallback imports `llm` and
      `message_text`. That path is retained for rollback and is not called by
      active detection, which is deterministic.

Implementation:
    - Finds a .env file and loads whatever credentials it holds.
    - `PROVIDERS`: a registry mapping a short provider name to the LangChain
      integration package, chat class, default model, and the environment
      variables that provider accepts as its API key. `PROVIDER_ALIASES` folds
      alternative spellings onto those names.
    - `build_llm(...)`: imports the chosen integration lazily and constructs its
      chat class. Two environment variables select what gets built -
      `PROVENANCE_LLM_PROVIDER` (default `google`) and `PROVENANCE_LLM_MODEL`
      (default: the provider's entry in the registry).
    - `llm`: the module-level client, built once at import with temperature=0
      for determinism.
    - `message_text(message)`: normalizes a response's content to plain text
      (providers return either a string or a list of typed content parts).

Design decisions:
    - The registry is a lazy import table rather than langchain's
      `init_chat_model`. `init_chat_model` lives in the `langchain` umbrella
      package, which this project deliberately does not depend on - only
      `langchain-core` plus one integration. A dict of four entries buys the
      same provider-agnosticism without that dependency.
    - No LLM integration is a hard dependency, not even the default provider's.
      All four are optional extras (`pip install "provenance-agent[openai]"`)
      imported only when selected, so a user never installs a provider they will
      not use - which matters because `langchain-google-genai` pulls
      `google-genai` and `google-auth`, and this package is routinely installed
      into conda environments where extra transitive builds are a hazard. The
      cost is that a core install has no provider at all; a missing integration
      raises a message naming the exact install command rather than a bare
      ModuleNotFoundError, so that failure is self-correcting.
    - PaleoPAL compatibility is credentials and vendor only, never the model.
      This agent is standalone today but is expected to be integrated into
      PaleoPAL, so its configuration is readable from a PaleoPAL `.env` without
      editing. The key variables already match PaleoPAL's exactly
      (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`),
      and `resolve_provider` falls back to PaleoPAL's `DEFAULT_LLM_PROVIDER`
      when ours is unset, accepting its `grok` spelling for xAI. PaleoPAL's
      per-provider model variables (`OPENAI_MODEL`, `CLAUDE_MODEL`,
      `GOOGLE_MODEL`, `GROK_MODEL`, `OLLAMA_MODEL`) are deliberately *not*
      read: their defaults are heavyweight reasoning models chosen for
      multi-agent literature work, while the only call made here is one short
      classification that the cheap fast tier handles. Inheriting the vendor a
      user has credentials for is helpful; inheriting a model picked for a
      different task would silently cost them money and latency. All of this is
      a documented convenience, not a contract - ours always wins, and nothing
      breaks if PaleoPAL renames a variable.
    - Credentials are validated here, before the chat class is constructed.
      Pydantic-based chat classes raise a `ValidationError` about a field name
      when a key is absent, which reads as a bug in this project rather than a
      missing key. The explicit check names the variable to set instead. A
      provider with no `key_variables` (Ollama, which is local) skips the check.
    - Credentials are discovered, not packaged. The lookup is working-directory
      first (`find_dotenv(usecwd=True)`) so an installed copy picks up the
      .env beside whatever project is using it, with a source-tree fallback
      (`find_dotenv()`) so a plain `pytest` run out of this checkout still
      finds one. No .env file is ever included in the built artifact.
    - `load_dotenv` is left at its default `override=False`, so a key already
      exported in the environment beats any dotenv value. CI and shell exports
      therefore win over a stale local file.
    - Note for notebook users: python-dotenv treats a Jupyter kernel as
      interactive and resolves *both* lookups from the kernel's working
      directory, so a .env buried in src/ is invisible there. Keep the .env at
      the repository root (or export the key) for `%load_ext provenance`.
    - APA rendering has been removed. There is no `bibtex_to_apa` chain here
      anymore, because APA output is no longer produced by an LLM. `fmt` is
      still accepted by the data workflow but is ignored, so nothing in this
      module renders citation text.
"""

from __future__ import annotations

import importlib
import os
from typing import NamedTuple

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True) or find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)


class ProviderSpec(NamedTuple):
    """
    Everything needed to construct one provider's LangChain chat client.

    Attributes:
        module: the integration package to import, e.g. "langchain_openai"
        class_name: the chat class inside that module
        default_model: the model used when PROVENANCE_LLM_MODEL is unset
        key_variables: environment variables the provider accepts as its API
            key; any one of them being set is enough. Empty for local providers
            that need no credentials.
        install: the pip command that makes this provider importable
    """

    module: str
    class_name: str
    default_model: str
    key_variables: tuple[str, ...]
    install: str


# gemini-flash-latest is an alias that tracks the current Gemini Flash model, so
# a specific version being retired does not 404 us (gemini-2.5-flash was retired).
PROVIDERS: dict[str, ProviderSpec] = {
    "google": ProviderSpec(
        module="langchain_google_genai",
        class_name="ChatGoogleGenerativeAI",
        default_model="gemini-flash-latest",
        key_variables=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        install='pip install "provenance-agent[google]"',
    ),
    "openai": ProviderSpec(
        module="langchain_openai",
        class_name="ChatOpenAI",
        default_model="gpt-4o-mini",
        key_variables=("OPENAI_API_KEY",),
        install='pip install "provenance-agent[openai]"',
    ),
    "anthropic": ProviderSpec(
        module="langchain_anthropic",
        class_name="ChatAnthropic",
        default_model="claude-sonnet-5",
        key_variables=("ANTHROPIC_API_KEY",),
        install='pip install "provenance-agent[anthropic]"',
    ),
    "ollama": ProviderSpec(
        module="langchain_ollama",
        class_name="ChatOllama",
        default_model="llama3.1",
        key_variables=(),  # runs locally; no credentials at all
        install='pip install "provenance-agent[ollama]"',
    ),
    "xai": ProviderSpec(
        module="langchain_xai",
        class_name="ChatXAI",
        default_model="grok-4",
        key_variables=("XAI_API_KEY",),
        install='pip install "provenance-agent[xai]"',
    ),
}

# PaleoPAL calls xAI "grok" in DEFAULT_LLM_PROVIDER, so accept that spelling.
PROVIDER_ALIASES = {"grok": "xai"}

DEFAULT_PROVIDER = "google"

# PaleoPAL's own provider variable, read only when ours is unset. See the
# module docstring for why the model variables are deliberately not inherited.
PALEOPAL_PROVIDER_VARIABLE = "DEFAULT_LLM_PROVIDER"


def resolve_provider(provider: str | None = None) -> str:
    """
    Determines which provider to build, and checks that it is a known one.

    Precedence is explicit argument, then PROVENANCE_LLM_PROVIDER, then
    PaleoPAL's DEFAULT_LLM_PROVIDER, then DEFAULT_PROVIDER. Aliases in
    PROVIDER_ALIASES are folded onto their canonical name.

    Args:
        provider: an explicit provider name; when None, the environment is
            consulted in the order above

    Returns:
        the canonical lowercased provider name, guaranteed to be a key of
        PROVIDERS

    Raises:
        ValueError: if the name is not in PROVIDERS. The message names whichever
            variable supplied the bad value, so a stray DEFAULT_LLM_PROVIDER
            does not send the reader to the wrong setting.
    """
    name, source = provider, "the provider argument"
    if not name:
        name, source = os.getenv("PROVENANCE_LLM_PROVIDER"), "PROVENANCE_LLM_PROVIDER"
    if not name:
        name, source = os.getenv(PALEOPAL_PROVIDER_VARIABLE), PALEOPAL_PROVIDER_VARIABLE
    if not name:
        name, source = DEFAULT_PROVIDER, "the built-in default"

    name = name.strip().lower()
    name = PROVIDER_ALIASES.get(name, name)
    if name not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"Unknown LLM provider {name!r} (from {source}). Set "
            f"PROVENANCE_LLM_PROVIDER to one of: {known}."
        )
    return name


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0,
    **kwargs,
):
    """
    Constructs the chat client for the configured provider.

    The integration package is imported lazily, so selecting a provider is the
    only thing that requires having installed it.

    Args:
        provider: provider name; defaults to PROVENANCE_LLM_PROVIDER, then
            DEFAULT_PROVIDER
        model: model identifier; defaults to PROVENANCE_LLM_MODEL, then the
            provider's default_model
        temperature: sampling temperature, 0 for reproducible classification
        **kwargs: forwarded verbatim to the provider's chat class

    Returns:
        an instantiated LangChain chat model

    Raises:
        ValueError: if the provider name is unknown
        RuntimeError: if the integration package is not installed, or if none of
            the provider's API key variables are set
    """
    name = resolve_provider(provider)
    spec = PROVIDERS[name]
    model_id = model or os.getenv("PROVENANCE_LLM_MODEL") or spec.default_model

    if spec.key_variables and not any(os.getenv(v) for v in spec.key_variables):
        accepted = " or ".join(spec.key_variables)
        raise RuntimeError(
            f"No API key found for LLM provider {name!r}. Set {accepted} in your "
            f"environment or in a .env file at the root of the project you are "
            f"working in. To use a different provider, set "
            f"PROVENANCE_LLM_PROVIDER (one of: {', '.join(sorted(PROVIDERS))})."
        )

    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"LLM provider {name!r} needs the {spec.module} package, which is "
            f"not installed. Install it with: {spec.install}"
        ) from exc

    chat_class = getattr(module, spec.class_name)
    return chat_class(model=model_id, temperature=temperature, **kwargs)


llm = build_llm()


def message_text(message) -> str:
    """
    Extracts the plain text of a LangChain AI message.

    Responses arrive either as a plain string or as a list of typed content
    parts like {"type": "text", "text": ...}. Newer langchain-google-genai
    versions return the list form (with thought signatures in "extras"), and
    reasoning-capable models on other providers do the same. Text parts are
    joined; non-text parts (e.g. thinking) are dropped.

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
