# Future Integration into PaleoPAL

> **Status: notes, not a plan.** Nothing here is scheduled or committed. This
> records what an integration would actually touch, which seams already exist,
> and which assumptions this agent makes that PaleoPAL does not satisfy, so that
> whoever does the work does not have to rediscover it.
>
> **Claims about PaleoPAL are a snapshot taken 2026-08-05** against a local
> checkout of `LinkedEarth/paleopal`. It is a sibling repository that moves
> independently, so verify any file path or class name below before relying on
> it. Claims about *this* repository are current and enforced by its tests.

See also [design-decisions.md](design-decisions.md) for why this agent is built
the way it is, and [documentation-draft.md](documentation-draft.md) for what it
does today.

---

## Contents

1. [The standing constraint](#1-the-standing-constraint)
2. [What already fits](#2-what-already-fits)
3. [The two assumptions PaleoPAL does not satisfy](#3-the-two-assumptions-paleopal-does-not-satisfy)
4. [Three candidate host surfaces](#4-three-candidate-host-surfaces)
5. [Mechanical friction](#5-mechanical-friction)
6. [What integration should not change](#6-what-integration-should-not-change)
7. [Open questions](#7-open-questions)

---

## 1. The standing constraint

**It must work both ways.** Standing alone is the requirement today; integration
is the expected destination. The rule that follows from holding both at once:

- Nothing in this package may *require* PaleoPAL to be present.
- Nothing in this package may make dropping it into PaleoPAL awkward.

In practice that means matching PaleoPAL's conventions where they are free to
match, and never importing PaleoPAL code. The dependency arrow points one way
only: PaleoPAL may depend on `provenance-agent`, never the reverse. A change
that inverts that arrow is the one change to refuse.

This also rules out the shortcut of delegating citation retrieval to PaleoPAL's
Code or SPARQL agents. Retrieval is hardcoded here and stays hardcoded, because
delegation would make the standalone mode impossible.

---

## 2. What already fits

Three seams were built with integration in mind and need no further work.

### 2.1 Model injection is the intended entry point

`run()` takes an optional `model` argument that is any LangChain `Runnable`
returning a message:

```python
from provenance_agent.agent import run

result = run("cite the software", "notebook.ipynb", model=my_chat_client)
```

PaleoPAL's `LLMProviderFactory.get_langchain_model()`
(`backend/services/llm_providers.py`) returns a `LangChainWrapper`, which
subclasses `BaseChatModel` and is therefore a `Runnable`. The wiring is one
expression:

```python
from services.llm_providers import LLMProviderFactory
from provenance_agent.agent import run

model = LLMProviderFactory.get_langchain_model(provider_type=..., model_name=...)
result = run(user_input, notebook_path, model=model)
```

This is why `llm.py`'s provider registry is a standalone-mode *default* rather
than a competing abstraction. An embedding host passes its own client and the
registry never runs. Nothing has to be deleted on integration.

Two caveats on PaleoPAL's wrapper:

- `get_langchain_model()` silently falls back through a hardcoded list of other
  providers and models when the requested one is unavailable. A host that cares
  which model classified the request should check what it got back.
- `LangChainWrapper` exists partly to strip reasoning-model preamble
  (`clean_reasoning_response`, `extract_json_from_response`). That matters here:
  the classification stage parses a Pydantic-validated JSON decision, and
  PaleoPAL's Ollama default is `deepseek-r1`, a reasoning model that wraps its
  JSON. Route through the wrapper rather than around it.

### 2.2 Credentials and vendor selection already carry over

Deliberately compatible, and already true today:

| Concern | Status |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY` | identical names in both projects, so an existing PaleoPAL key works untouched |
| `DEFAULT_LLM_PROVIDER` | read as a fallback when `PROVENANCE_LLM_PROVIDER` is unset, including PaleoPAL's `grok` spelling for xAI |
| `OPENAI_MODEL`, `CLAUDE_MODEL`, `GOOGLE_MODEL`, `GROK_MODEL`, `OLLAMA_MODEL` | deliberately **not** read |

The model variables are excluded on purpose. PaleoPAL's defaults are heavyweight
reasoning models chosen for multi-agent literature work (`gpt-5.5`,
`gemini-2.5-pro`, `deepseek-r1`); the only model call here is one short
classification. Inheriting the vendor a user already has credentials for is
helpful. Inheriting a model picked for a different task would silently cost them
money and latency. Do not "finish the job" by wiring these up.

All of this is a **convenience, not a contract.** Ours always wins, and nothing
breaks if PaleoPAL renames a variable.

### 2.3 The envelope maps onto `AgentResponse` almost directly

`run()` returns a JSON-serializable dict, which is what `AgentResponse.result`
wants:

| `run()` envelope | `AgentResponse` field |
|---|---|
| `status: "ok"` | `status=AgentStatus.SUCCESS` |
| `status: "warning"` | `AgentStatus.SUCCESS` with the warning in `message`, or `NEEDS_CLARIFICATION` if the host wants to re-prompt |
| `decision`, `dispatch` | `result` |
| `verification` | `metadata` |
| `warning` | `message` |

Note that `status: "warning"` is not an error. It means the request was
understood but nothing was injected, and the notebook was deliberately left
unmutated. Mapping it to `AgentStatus.ERROR` would misreport a working safety
behavior as a failure.

---

## 3. The two assumptions PaleoPAL does not satisfy

This is the substance of the integration problem, and it is worth stating
plainly before discussing where to attach.

**This agent assumes there is an `.ipynb` on disk, and assumes it cannot reach a
kernel. PaleoPAL is the exact inverse: it has no notebook on disk during a
conversation, and it does have a kernel.**

### 3.1 There is no notebook file to read

Every entry point here takes a path and reads the notebook from disk with
`nbformat`. In PaleoPAL, a conversation is a sequence of messages; the notebook
does not exist until `NotebookExportService.export_conversation_to_notebook(conversation_id)`
(`backend/services/notebook_export_service.py`) builds one from the message
history at the end.

There is a `notebook_context` field on `AgentRequest` carrying "notebook
variables and cell context", but it is a `Dict[str, Any]` populated from the
request body, not a path to a file. Uploaded notebooks
(`POST /document_extraction/notebook`) are written to a `NamedTemporaryFile` and
extracted, not retained as a working document.

So an integration must supply a notebook by one of: materializing the
conversation to a temporary `.ipynb` first, reaching the VSCode extension's live
notebook document, or hooking export.

### 3.2 There *is* a kernel, which changes the design

The injected-cell contract exists for exactly one reason: this agent cannot
reach the user's kernel, and dataset citations require calling `get_bibtex()` on
a LiPD object that lives there. So it writes retrieval code into the notebook
and hands it back.

PaleoPAL removes that constraint. `ExecutionClient.execute_code(code, conversation_id)`
(`backend/services/execution_client.py`) runs code against a per-conversation
isolated execution service that already holds the loaded objects.

**Inside PaleoPAL, the injection step could collapse into direct execution.**
Build the retrieval code, execute it, return the DataFrame in the response.

This is the strongest argument for a structural property the package already
has: **the cell builders are public and separate from injection.** `data.py`
exposes `build_retrieval_cell()` and `build_dataset_cell()` independently of
`inject_retrieval_cells()`, and `software.py` exposes `build_metadata_cell()`
independently of `inject_metadata_cell()`. A host with a kernel uses the
builders and skips the injectors. Keep that separation intact.

### 3.3 Two hazards in the execution service, if that path is taken

Read `backend/services/isolated_execution_service/isolated_execution_service.py`
before designing around it. State is persisted between executions by pickling
variables into SQLite, and the filter has two consequences that hit this agent
specifically:

**Underscore-prefixed names are dropped.** The variable-extraction loop keeps
only names where `not name.startswith('_')`. Every result our generated cells
bind is underscore-prefixed by convention: `_bib_{variable}`, `_meta_{variable}`,
and the intermediate `_names_*` / `_lipd_*` bindings. **None of them would
survive into the next execution.** Either read the results out of the same
execution's response, or rename the bindings for that host. Do not assume they
persist.

**Non-picklable objects are silently skipped.** Each variable is round-tripped
through `pickle.dumps()` in a bare `try/except`, and anything that fails is
dropped without a warning. Untargeted dataset retrieval depends on reusing the
already-loaded `LiPD` / `PangaeaDataset` / `NOAADataset` object from a previous
cell. If those objects do not pickle, they are not there on the next execution
and retrieval fails with a `NameError` that points at nothing obvious. **Verify
picklability of the PyLiPD and PyleoTUPS objects before relying on cross-execution
reuse.** The targeted path, which constructs a fresh `LiPD` object inside the
generated code, is immune to this and may be the safer default in that host.

The service does preload `pylipd` into the execution namespace, which is a point
in favor of this pathway.

---

## 4. Three candidate host surfaces

Each solves the notebook-source problem differently. They are not mutually
exclusive.

### 4.1 A backend agent under `agents/provenance/`

Follows the existing pattern most closely. PaleoPAL's agents subclass
`BaseAgent` (`backend/agents/base_agent.py`), declare `AgentCapability` entries,
and are registered in `initialize_agents()` in `backend/routers/agents.py`
alongside the SPARQL, Code, and Workflow agents. `AgentRegistry` is a singleton
keyed by `agent_type`.

Sketch:

```python
class ProvenanceAgent(BaseAgent):
    def __init__(self):
        super().__init__("provenance", "Provenance Agent",
                         "Generates citations for the data and software in a notebook")
        self.register_capability(AgentCapability(name="cite_software", ...))
        self.register_capability(AgentCapability(name="cite_data", ...))

    async def handle_request(self, request: AgentRequest) -> AgentResponse:
        ...
```

- **Fits:** registry, routing, capability discovery (`GET /agents`), streaming
  scaffolding, and conversation plumbing all come free.
- **Costs:** must solve [3.1](#31-there-is-no-notebook-file-to-read) itself.
- **Note:** `handle_request` is `async`; see [5.1](#51-sync-versus-async).

Registration is one call added to `initialize_agents()`. That function runs at
module import of `routers/agents.py` and re-raises on failure, so a provenance
agent that throws at construction would take down agent routing for the whole
backend. Construct nothing expensive in `__init__`.

### 4.2 The VSCode extension

`vscode-extension/src/notebook.ts` is the only place in the PaleoPAL codebase
where a real, live notebook document exists, and it already does exactly the
operation this agent performs: inserting cells, via `insertCodeCellAt()`,
`insertMarkdownCellAt()`, and `vscode.NotebookEdit.replaceCells`.

- **Fits:** the user has a notebook open and a kernel running. This is the
  closest match to the experience `%provenance` delivers today.
- **Costs:** the extension is TypeScript, so it would call the backend over HTTP
  rather than the Python API. And the document it creates is
  `untitled:PaleoPal.ipynb`, which has no `fsPath` until saved, so a
  path-based API still cannot read it. Cell content would have to be sent to the
  backend and the returned cell source inserted client-side.

That shape is workable and arguably the cleanest: the backend becomes a pure
function from notebook content to cell source, with no file I/O and no
mutation. It would need a content-based entry point next to the path-based one.

### 4.3 An export-time hook

Run `cite_software()` and `cite_data()` on the notebook that
`NotebookExportService` has just written, before it reaches the user.

- **Fits:** a real file on disk, which is precisely the current API's
  assumption. Near-zero integration code.
- **Costs:** the injected cells arrive un-run, so the user sees retrieval code
  rather than citations, and the data cell may fail because the exported
  notebook's own cells have not executed in their kernel. It also cites only
  what the conversation happened to produce.

Reasonable as a first increment because it is nearly free; not the destination.

### Summary

| | Notebook source | Kernel available | Integration cost |
|---|---|---|---|
| Backend agent | must be materialized | yes, via `ExecutionClient` | medium |
| VSCode extension | live document, unsaved | yes, the user's own | medium, spans two languages |
| Export hook | the exported file | no | low |

---

## 5. Mechanical friction

Small, real, and each one will otherwise cost an afternoon.

### 5.1 Sync versus async

`BaseAgent.handle_request` is `async`. Every public function here is synchronous
and does blocking work: file reads, `nbformat` writes, and a model call. Calling
them directly from a coroutine blocks the event loop.

Wrap them: `await asyncio.to_thread(run, user_input, path, model=model)`.
`routers/agents.py` already keeps a `ThreadPoolExecutor(max_workers=4)` for this
purpose.

### 5.2 Environment and dependencies

- PaleoPAL's backend runs in `backend/.venv`; this project is developed in the
  `lang` conda env. `pylipd` and `pyleotups` are hard dependencies here, because
  the generated retrieval cells import them, so they must be installed wherever
  the agent runs.
- The analysis libraries (`pyleoclim`, `xarray`, `eofs`, `cfr`) are deliberately
  **not** dependencies. They belong to the analyzed notebook's environment.
- No LLM integration is a core dependency; all five providers are extras. A host
  that always passes `model=` needs none of them. A host that relies on
  `PROVENANCE_LLM_PROVIDER` must install the matching extra.

### 5.3 The dotenv collision

PaleoPAL's `backend/config.py` calls a bare `load_dotenv()` at import, resolving
relative to the process working directory. This project calls
`find_dotenv(usecwd=True)` first with a source-tree fallback. In one FastAPI
process both run, and both leave `override=False`, so an already-exported
variable wins over either file. That is the desired outcome, but it means
whichever file loads first wins between the two files.

Concretely: PaleoPAL keeps keys in `backend/.env`. Our lookup finds it only if
the process working directory is `backend/`. An embedding host should either
export its keys or pass `model=` and bypass credential discovery here entirely.

Note also that `DEFAULT_LLM_PROVIDER` defaults to `openai` in PaleoPAL. A
PaleoPAL environment with `PROVENANCE_LLM_PROVIDER` unset therefore selects
OpenAI here, which requires `pip install "provenance-agent[openai]"`. Without it
the failure is a `RuntimeError` naming the exact install command.

### 5.4 The magic layer does not travel

`%load_ext provenance`, `%provenance`, `%provenance_notebook`, and the
`ipynbname` auto-detection are notebook-session concepts with no meaning in a
FastAPI backend. An integration targets `agent.run()` or the two direct
functions. `magic.py` and `src/provenance.py` stay behind, unused, and cost
nothing.

---

## 6. What integration should not change

Guardrails, phrased as things a future change should be measured against.

1. **No PaleoPAL import, ever.** Standalone use must keep working with PaleoPAL
   absent. `tests/test_import_hygiene.py` and `tests/test_packaging.py` are the
   backstop.
2. **Citation retrieval stays hardcoded here.** Not delegated to PaleoPAL's Code
   or SPARQL agents. Delegation would make standalone mode impossible.
3. **The builder/injector separation stays.** It is what lets a host with a
   kernel skip injection. See [3.2](#32-there-is-a-kernel-which-changes-the-design).
4. **Nothing constructs an LLM client at import.** The lazy `_CLIENT` / `_CHAIN`
   caches are what let a host pass its own model without this package ever
   building one, and what let the offline test suite run without credentials.
5. **Model configuration compatibility stays a convenience.** Ours wins; nothing
   verifies PaleoPAL's variable names; a rename there must not break anything
   here.
6. **Detection stays deterministic.** The deprecated LLM detector in
   `dataset_detection.py` is a rollback path, not a design direction to revisit
   because a host happens to have a model handy.

---

## 7. Open questions

Unresolved, and each needs a decision before implementation rather than during
it.

- **Which host surface?** [4.1](#41-a-backend-agent-under-agentsprovenance),
  [4.2](#42-the-vscode-extension), or [4.3](#43-an-export-time-hook). They imply
  different APIs.
- **Does the package need a content-based entry point?** Every public function
  takes a path. A backend or an extension would rather pass notebook JSON and
  receive cell source, with no file I/O. That is an additive API, not a
  breaking one, but it is a real design decision about what this package's
  contract is.
- **Do the PyLiPD and PyleoTUPS objects pickle?** Determines whether the
  execution-service pathway can reuse loaded objects at all. See
  [3.3](#33-two-hazards-in-the-execution-service-if-that-path-is-taken).
- **Should citations become a first-class conversation artifact?** PaleoPAL has
  conversations, messages, and export. Citations could be a message type rather
  than a notebook cell. That would be a genuinely different product and is out of
  scope for a first integration, but it is the direction the ecosystem points.
- **Where does the "prompt the user when no citation is found" behavior live?**
  The README names it as future work for this agent. PaleoPAL already has
  clarification machinery (`clarification_questions` on `AgentResponse`,
  `enable_clarification` config). Building it here would duplicate that; building
  it there would put agent logic outside the agent.
