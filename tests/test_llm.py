"""
Unit tests for llm.py's offline pieces.

Two areas are covered, both without a network call or a real API key:

- `message_text()`, the normalizer that turns a LangChain AI message into plain
  text. Responses arrive either as a plain string or as a list of typed content
  parts, and both shapes must normalize to the same text. Fake messages are
  built with SimpleNamespace so no model call happens.
- The provider registry: name resolution, the two environment variables, and
  the two failure messages (missing package, missing key). Constructing a real
  provider client would need credentials, so the one construction test swaps in
  a ProviderSpec pointing at `types.SimpleNamespace`, which accepts the same
  keyword arguments a chat class would and records them for inspection.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from provenance_agent import llm as llm_module
from provenance_agent.llm import (
    DEFAULT_PROVIDER,
    PALEOPAL_PROVIDER_VARIABLE,
    PROVIDER_ALIASES,
    PROVIDERS,
    ProviderSpec,
    build_llm,
    message_text,
    resolve_provider,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Removes every variable the registry reads, so tests start from nothing."""
    monkeypatch.delenv("PROVENANCE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("PROVENANCE_LLM_MODEL", raising=False)
    monkeypatch.delenv(PALEOPAL_PROVIDER_VARIABLE, raising=False)
    for spec in PROVIDERS.values():
        for variable in spec.key_variables:
            monkeypatch.delenv(variable, raising=False)
    return monkeypatch


# --- fake provider ------------------------------------------------------------

def _recording_spec(**overrides) -> ProviderSpec:
    """A ProviderSpec whose 'chat class' is SimpleNamespace, so it just records."""
    fields = {
        "module": "types",
        "class_name": "SimpleNamespace",
        "default_model": "fake-default",
        "key_variables": ("FAKE_API_KEY",),
        "install": "pip install nothing",
    }
    fields.update(overrides)
    return ProviderSpec(**fields)


def _patch_xai_client(monkeypatch):
    """Replaces the xAI integration import with an offline recording client."""
    class RecordingXAI:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

    monkeypatch.setattr(
        llm_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(ChatXAI=RecordingXAI),
    )


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


# --- provider resolution ------------------------------------------------------

def test_default_provider_is_google_and_is_registered():
    assert DEFAULT_PROVIDER == "google"
    assert DEFAULT_PROVIDER in PROVIDERS


def test_resolve_provider_falls_back_to_the_default(clean_env):
    assert resolve_provider() == DEFAULT_PROVIDER


def test_resolve_provider_reads_the_environment_variable(clean_env):
    clean_env.setenv("PROVENANCE_LLM_PROVIDER", "openai")
    assert resolve_provider() == "openai"


def test_resolve_provider_normalizes_case_and_whitespace(clean_env):
    clean_env.setenv("PROVENANCE_LLM_PROVIDER", "  Anthropic  ")
    assert resolve_provider() == "anthropic"


def test_an_explicit_argument_beats_the_environment_variable(clean_env):
    clean_env.setenv("PROVENANCE_LLM_PROVIDER", "openai")
    assert resolve_provider("ollama") == "ollama"


def test_an_unknown_provider_names_the_registered_ones(clean_env):
    with pytest.raises(ValueError) as excinfo:
        resolve_provider("bedrock")
    message = str(excinfo.value)
    assert "bedrock" in message
    assert "google" in message and "openai" in message


# --- PaleoPAL compatibility ---------------------------------------------------

def test_paleopals_provider_variable_is_honored(clean_env):
    """A PaleoPAL .env selects the same vendor here without being edited."""
    clean_env.setenv(PALEOPAL_PROVIDER_VARIABLE, "anthropic")
    assert resolve_provider() == "anthropic"


def test_our_provider_variable_beats_paleopals(clean_env):
    clean_env.setenv(PALEOPAL_PROVIDER_VARIABLE, "anthropic")
    clean_env.setenv("PROVENANCE_LLM_PROVIDER", "ollama")
    assert resolve_provider() == "ollama"


def test_paleopals_grok_spelling_resolves_to_xai(clean_env):
    clean_env.setenv(PALEOPAL_PROVIDER_VARIABLE, "grok")
    assert resolve_provider() == "xai"


def test_the_grok_alias_works_from_our_variable_too(clean_env):
    clean_env.setenv("PROVENANCE_LLM_PROVIDER", "GROK")
    assert resolve_provider() == "xai"


def test_xai_registry_default_is_grok_4_3():
    assert PROVIDERS["xai"].default_model == "grok-4.3"


def test_every_alias_points_at_a_registered_provider():
    for alias, canonical in PROVIDER_ALIASES.items():
        assert canonical in PROVIDERS, alias
        assert alias not in PROVIDERS, f"{alias} is an alias, not a provider"


def test_paleopals_model_variables_are_not_inherited(clean_env):
    """
    PaleoPAL's model defaults are heavyweight reasoning models chosen for a
    different task. Selecting its vendor must not also adopt its model.
    """
    clean_env.setitem(PROVIDERS, "fake", _recording_spec())
    clean_env.setenv("FAKE_API_KEY", "x")
    for variable in (
        "OPENAI_MODEL", "CLAUDE_MODEL", "GOOGLE_MODEL", "GROK_MODEL", "OLLAMA_MODEL"
    ):
        clean_env.setenv(variable, "paleopal-heavyweight-model")

    assert build_llm("fake").model == "fake-default"


def test_the_error_names_whichever_variable_supplied_the_bad_value(clean_env):
    clean_env.setenv(PALEOPAL_PROVIDER_VARIABLE, "bedrock")
    with pytest.raises(ValueError) as excinfo:
        resolve_provider()
    assert PALEOPAL_PROVIDER_VARIABLE in str(excinfo.value)


def test_the_key_variables_match_paleopals_names():
    """
    A PaleoPAL user's existing keys must work here untouched. These four names
    are read from paleopal/backend/config.py.
    """
    expected = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
    }
    for provider, variable in expected.items():
        assert variable in PROVIDERS[provider].key_variables, provider


# --- construction -------------------------------------------------------------

def test_build_llm_passes_the_default_model_and_temperature(clean_env):
    clean_env.setitem(PROVIDERS, "fake", _recording_spec())
    clean_env.setenv("FAKE_API_KEY", "x")

    client = build_llm("fake")

    assert client.model == "fake-default"
    assert client.temperature == 0


def test_build_llm_omits_temperature_for_anthropic(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "x")

    class NoTemperatureClient:
        def __init__(self, **kwargs):
            assert "temperature" not in kwargs
            self.model = kwargs["model"]

    fake_module = SimpleNamespace(ChatAnthropic=NoTemperatureClient)
    clean_env.setattr(
        llm_module.importlib,
        "import_module",
        lambda module_name: fake_module,
    )

    client = build_llm("anthropic")

    assert client.model == PROVIDERS["anthropic"].default_model


def test_xai_legacy_model_from_environment_is_rerouted(clean_env):
    clean_env.setenv("XAI_API_KEY", "x")
    clean_env.setenv("PROVENANCE_LLM_PROVIDER", "xai")
    clean_env.setenv("PROVENANCE_LLM_MODEL", "grok-4")
    _patch_xai_client(clean_env)

    assert build_llm().model == "grok-4.3"


def test_explicit_legacy_xai_model_is_rerouted(clean_env):
    clean_env.setenv("XAI_API_KEY", "x")
    _patch_xai_client(clean_env)

    assert build_llm("xai", model="grok-4").model == "grok-4.3"


def test_build_llm_honors_the_model_environment_variable(clean_env):
    clean_env.setitem(PROVIDERS, "fake", _recording_spec())
    clean_env.setenv("FAKE_API_KEY", "x")
    clean_env.setenv("PROVENANCE_LLM_MODEL", "from-the-environment")

    assert build_llm("fake").model == "from-the-environment"


def test_an_explicit_model_beats_the_environment_variable(clean_env):
    clean_env.setitem(PROVIDERS, "fake", _recording_spec())
    clean_env.setenv("FAKE_API_KEY", "x")
    clean_env.setenv("PROVENANCE_LLM_MODEL", "from-the-environment")

    assert build_llm("fake", model="explicit").model == "explicit"


def test_extra_keyword_arguments_reach_the_chat_class(clean_env):
    clean_env.setitem(PROVIDERS, "fake", _recording_spec())
    clean_env.setenv("FAKE_API_KEY", "x")

    assert build_llm("fake", max_retries=7).max_retries == 7


# --- failure messages ---------------------------------------------------------

def test_a_missing_key_names_the_variable_to_set(clean_env):
    clean_env.setitem(PROVIDERS, "fake", _recording_spec())

    with pytest.raises(RuntimeError) as excinfo:
        build_llm("fake")
    assert "FAKE_API_KEY" in str(excinfo.value)


def test_any_one_of_a_providers_key_variables_is_enough(clean_env):
    clean_env.setitem(
        PROVIDERS, "fake", _recording_spec(key_variables=("FIRST_KEY", "SECOND_KEY"))
    )
    clean_env.setenv("SECOND_KEY", "x")

    assert build_llm("fake").model == "fake-default"


def test_a_provider_with_no_key_variables_skips_the_check(clean_env):
    """Ollama is local, so an absent API key must not stop construction."""
    clean_env.setitem(PROVIDERS, "fake", _recording_spec(key_variables=()))

    assert build_llm("fake").model == "fake-default"


def test_a_missing_integration_package_names_the_install_command(clean_env):
    clean_env.setitem(
        PROVIDERS,
        "fake",
        _recording_spec(
            module="provenance_agent_no_such_integration",
            install='pip install "provenance-agent[fake]"',
        ),
    )
    clean_env.setenv("FAKE_API_KEY", "x")

    with pytest.raises(RuntimeError) as excinfo:
        build_llm("fake")
    message = str(excinfo.value)
    assert "provenance_agent_no_such_integration" in message
    assert 'pip install "provenance-agent[fake]"' in message


def test_the_key_check_runs_before_the_import(clean_env):
    """A user with neither a key nor the package hears about the key first."""
    clean_env.setitem(
        PROVIDERS,
        "fake",
        _recording_spec(module="provenance_agent_no_such_integration"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        build_llm("fake")
    assert "FAKE_API_KEY" in str(excinfo.value)


# --- the module-level client --------------------------------------------------

def test_importing_the_module_constructs_no_client():
    """
    The whole point of the lazy accessor: import must cost nothing and need no
    credentials. Anything already cached came from another test, not import.
    """
    source = (REPO_ROOT / "src" / "provenance_agent" / "llm.py").read_text()
    assert "\nllm = build_llm()" not in source
    assert "_CLIENT = None" in source


def test_the_client_is_built_on_first_access_and_cached(clean_env):
    """Reading `llm` builds one client through the registry and reuses it."""
    clean_env.setitem(PROVIDERS, "fake", _recording_spec())
    clean_env.setenv("FAKE_API_KEY", "x")
    clean_env.setenv("PROVENANCE_LLM_PROVIDER", "fake")
    clean_env.setattr(llm_module, "_CLIENT", None)

    first = llm_module.llm
    second = llm_module.llm

    assert first.model == "fake-default"
    assert first is second, "the client must be cached, not rebuilt per access"


def test_seeding_the_cache_avoids_construction(clean_env):
    """The supported substitution point: assign _CLIENT, never patch `llm`."""
    sentinel = SimpleNamespace(model="substituted")
    clean_env.setattr(llm_module, "_CLIENT", sentinel)

    assert llm_module.llm is sentinel


def test_an_unknown_module_attribute_still_raises_attribute_error():
    """__getattr__ must not swallow genuine typos."""
    with pytest.raises(AttributeError):
        llm_module.no_such_attribute


def test_every_registered_provider_is_fully_specified():
    for name, spec in PROVIDERS.items():
        assert spec.module and spec.class_name, name
        assert spec.default_model, name
        assert spec.install.startswith("pip install"), name
        assert isinstance(spec.key_variables, tuple), name
        assert all(v == v.upper() for v in spec.key_variables), name
        assert isinstance(spec.supports_temperature, bool), name


def test_temperature_support_is_explicit_for_every_registered_provider():
    assert PROVIDERS["google"].supports_temperature is True
    assert PROVIDERS["openai"].supports_temperature is True
    assert PROVIDERS["anthropic"].supports_temperature is False
    assert PROVIDERS["ollama"].supports_temperature is True
    assert PROVIDERS["xai"].supports_temperature is True


def test_an_explicit_provider_ignores_a_configured_one(clean_env):
    """
    Selection happens in build_llm, not at module scope, so a caller can pass
    provider= explicitly without the environment interfering.
    """
    clean_env.setenv("PROVENANCE_LLM_PROVIDER", "anthropic")
    assert resolve_provider("google") == "google"
