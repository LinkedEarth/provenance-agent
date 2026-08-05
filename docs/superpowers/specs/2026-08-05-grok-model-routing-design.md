# Grok model routing design

## Status

Approved by the user on 2026-08-05.

## Goal

Move the xAI provider away from the legacy `grok-4` model name while keeping
existing project configurations working.

## Design

The xAI provider will use `grok-4.3` as its registry default. Model resolution
will preserve the existing precedence—an explicit `build_llm(model=...)` value,
then `PROVENANCE_LLM_MODEL`, then the provider default—but will normalize an
exact legacy `grok-4` value to `grok-4.3` when the selected provider is xAI.
Other providers and other explicit xAI model identifiers will be unchanged.

The README provider table will advertise `grok-4.3` as the xAI default and add a
short migration note so users understand that older `grok-4` settings are
rerouted. No changes are needed to credentials, provider extras, or temperature
handling.

## Testing

Add regression coverage for:

1. the xAI registry default being `grok-4.3`;
2. an xAI `PROVENANCE_LLM_MODEL=grok-4` value being normalized; and
3. an explicit xAI `model="grok-4"` value being normalized.

The existing provider tests and the full test suite must remain green.

## Non-goals

- Do not change model selection for Google, OpenAI, Anthropic, or Ollama.
- Do not remove support for explicitly selected, non-legacy xAI model IDs.
- Do not change the xAI extra or API-key handling.
