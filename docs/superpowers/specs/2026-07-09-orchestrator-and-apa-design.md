# Orchestrator + APA rendering design (#20, #21)

Date: 2026-07-09
Issues: #20 (LangChain orchestrator), #21 (APA rendering for both workflows)
Related: #24 (never-ran retrieval decision, deferred)

## Goal

Expose the software and data workflows as two LangChain tools with clear
descriptions ("system prompts") that route to the correct workflow with the
correct arguments. This is the tools-first step toward the eventual natural
language agent ("@provenance agent, generate citations"); the NL-routing agent
is deferred. APA rendering (#21) is folded in as an output-format option on both
tools rather than a separate module.

## Scope

In scope:
- A new `src/orchestrator.py` with two callable tools plus their LangChain
  `StructuredTool` wrappers.
- Extending `filter_datasets` to accept a list of variable targets.
- APA rendering wired in as a `fmt` argument reusing the existing renderers.

Out of scope (deferred):
- The NL-routing LangChain agent (a later step; the tool descriptions are
  written now so the agent can be added on top).
- The never-ran retrieval contract - tracked in #24. This design assumes the
  MVP behavior (reuse loaded objects; fail clearly if a target variable is not
  defined).
- A combined "cite everything" entry point. When both software and data are
  wanted, the (future) agent calls both tools and concatenates. No third
  function is added now.

## Architecture

Two tools, both thin wrappers over functions that already exist. No new citation
logic is introduced - the orchestrator only routes and shapes arguments.

```
orchestrator.py
  cite_software(notebook_path, libraries=None, citation_types=None, fmt="apa")
      -> notebook_parser.parse_notebook            (imported libraries)
      -> bibliography.collect_library_entries      (BibTeX, deduped, filtered)
      -> bibliography.render_apa                    (when fmt == "apa")

  cite_data(notebook_path, targets=None, fmt="apa")
      -> dataset_detection.detect_datasets         (LLM: [variable, tool] pairs)
      -> data_workflow.filter_datasets             (narrow by targets)
      -> data_workflow.inject_retrieval_cells      (retrieval cell per dataset)
         (the injected cell reuses live kernel objects and prints BibTeX,
          or renders BibTeX -> APA in-cell when fmt == "apa")

  cite_software_tool / cite_data_tool
      -> LangChain StructuredTool wrappers with routing descriptions
```

### Tool 1: cite_software (in-process)

Signature: `cite_software(notebook_path, libraries=None, citation_types=None, fmt="apa")`

- `libraries`: `None` (all imported libraries) | `str` | `list[str]`. A single
  string is normalized to a one-element list.
- `citation_types`: `None` (all) | list subset of `["paper", "software"]`.
- `fmt`: `"apa"` (default) or `"bibtex"`.

Flow: `parse_notebook` returns the sorted library list. When `libraries` is
given, validate against that list (via `validate_libraries`) so a typo or an
un-imported library surfaces instead of silently returning nothing.
`collect_library_entries` returns deduped BibTeX; `fmt="apa"` renders it with
`render_apa`. Returns the citation text directly - no kernel required.

### Tool 2: cite_data (cell output)

Signature: `cite_data(notebook_path, targets=None, fmt="apa")`

- `targets`: `None` (all detected datasets) | `str` | `list[str]` of variable
  names to cite.
- `fmt`: `"apa"` (default) or `"bibtex"`.

Flow: `detect_datasets` (LLM, static over source - works even if nothing ran)
returns `[variable, tool]` pairs. `filter_datasets` narrows to `targets`.
`inject_retrieval_cells` appends one retrieval cell per dataset; each cell
reuses the live kernel object and prints its BibTeX. When `fmt="apa"`, the
injected cell additionally renders the BibTeX to APA via Gemini so APA also
lands as cell output. The LiPDGraph endpoint is lifted from the notebook
(`extract_lipdgraph_endpoint`).

The data citations exist as **cell output**, not a Python return value, because
retrieval needs the live kernel objects. The tool returns the injected
`[variable, tool]` pairs (what it acted on).

### #21 APA rendering

Not a new module - the `fmt` argument selects it, reusing `bibliography.py`:
- Software: `render_apa(BibliographyData)` in-process.
- Data: the injected cell renders its BibTeX strings to APA
  (`render_bibtex_strings_to_apa` logic) so it appears as cell output.

`fmt="apa"` is the default (the finished product is a human-readable
bibliography); `fmt="bibtex"` skips the Gemini call for the deterministic raw
artifact.

### Supporting change

`filter_datasets(pairs, tool=None, variable=None)` gains list support on the
variable filter: `variable` may be a `str` or `list[str]`. `None` still means
no filter.

## Data flow summary

- Software: source -> libraries -> BibTeX -> (APA). Fully in-process.
- Data: source -> detection -> filtered pairs -> injected retrieval cells ->
  (run in kernel) -> BibTeX/APA as cell output.

## Error handling

- `cite_software` with an unknown/un-imported library: report it via
  `validate_libraries` (found/not_found) rather than silently omitting.
- `cite_data` retrieval when a target variable is not in memory: the injected
  cell fails with an actionable message ("`<var>` is not defined - run the
  notebook's data cells first"). Final contract deferred to #24.
- Unsupported detected tool: `build_retrieval_cell` already raises `ValueError`.
- `fmt` other than "apa"/"bibtex": raise `ValueError`.

## Testing

- Unit tests (no LLM/network): argument normalization (str -> list), `fmt`
  validation, `filter_datasets` list support, and that each tool routes to the
  expected underlying calls (monkeypatched). LangChain tool wrappers expose the
  right name/description/signature.
- Manual/dev: `notebooks/workflow.ipynb` (software) and
  `notebooks/testing/data_workflow.ipynb` (data) exercise the tools; the
  PaleoPCAlite LiPDGraph path is the end-to-end data check.

## Open questions

- #24: behavior when the notebook was never run (assumed reuse-or-fail here).
- Whether the eventual NL agent lives in `orchestrator.py` or its own module -
  decided when that step is picked up.
