# Integrating `provenance-agent` into PaleoPAL

This document is connecting the provenance workflows to a
PaleoPAL request path. It assumes that installation, standalone usage, and the
general behavior of this package are already understood from the README, so please read that if not done so already. The
integration problem is narrower: register provenance as one of PaleoPAL's
agents, translate its requests into this package's interfaces, and carry
generated-cell results back to the user.

> **Snapshot notice.** Everything below about PaleoPAL describes the
> `LinkedEarth/PaleoPAL` backend as it stood on 2026-08-06 (commit `0ddcf29`).
> That is a sibling repository on its own release schedule, so treat the file
> paths and class names as a starting point to verify rather than as a stable
> contract. Claims about `provenance-agent` itself are current.

## Recommended integration

PaleoPAL is a collection of agents, and provenance should become one of them.
Today the backend registers three: `SparqlGenerationAgent`,
`CodeGenerationAgent`, and `WorkflowGenerationAgent`. A provenance agent joins
them as a fourth, and calls this package for the provenance-specific work:
static import and dataset detection, curated software citation lookup, and
provider-specific dataset citation retrieval.

The reason for this arrangement is consistency. Dataset lineage rules and
provider-specific retrieval code should have one implementation; duplicating
them in another PaleoPAL agent would eventually produce different target
selection or different citation metadata. The model can classify a request,
but it should not be the authority for whether a variable is a dataset. That
decision is made by the deterministic detector, and the actual data citation
still runs against the objects in the execution session.

### The agent contract

An agent is a class, not a function call. The contract lives in
`backend/agents/base_agent.py`:

- Subclass `BaseAgent` and pass `agent_type`, `name`, and `description` to
  `super().__init__`.
- Register one or more `AgentCapability` objects, each carrying a JSON
  `input_schema` and `output_schema`.
- Implement `async def handle_request(self, request: AgentRequest) ->
  AgentResponse`.
- Register the instance in `initialize_agents()` in
  `backend/routers/agents.py`, beside the three existing agents.

The three current agents subclass `BaseLangGraphAgent` instead, which adds a
LangGraph state machine plus the conversation, message, and job services, and
requires implementing `_build_graph()` and `_create_agent_config()`. That
machinery exists to support multi-turn clarification. A provenance request does
not need it: this package already owns its own LCEL classification chain, so
subclassing `BaseAgent` directly is the smaller and more honest fit. Reach for
`BaseLangGraphAgent` only if provenance requests turn out to need PaleoPAL's
clarification dialogue.

Two capabilities cover the workflows cleanly, for example `cite_software` and
`cite_data`, or a single `cite_provenance` capability that accepts a kind. The
registry routes on `agent_type` and `capability` only, so whichever split you
choose becomes the public surface other parts of PaleoPAL call.

`handle_request` is `async`, while every entry point in this package is
synchronous. Detection and cell generation are CPU-bound and fast, so calling
them directly is acceptable; if a notebook is large enough for that to block
the event loop, wrap the call in `asyncio.to_thread`.

## Choose the integration surface

### Live execution: build the cells, let PaleoPAL run them

This is the best fit for an interactive PaleoPAL session. The agent can build
the source, submit it to the execution service, and return the output
immediately, without first rewriting an `.ipynb` file.

The basic sequence is:

```python
from provenance_agent.data import build_dataset_cell
from provenance_agent.dataset_detection import detect_datasets
from provenance_agent.notebook_io import parse_notebook
from provenance_agent.software import build_metadata_cell

libraries = parse_notebook(notebook_path)
pairs = detect_datasets(notebook_path)

if request_software:
    software_source = build_metadata_cell(libraries)
    # Submit software_source through PaleoPAL's execution client.

if request_data and pairs:
    data_source = build_dataset_cell(pairs)
    # Submit data_source through the same execution client.
```

The builders return Python source; they do not execute it. PaleoPAL executes
code through `ExecutionClient.execute_code(code, conversation_id)` in
`backend/services/execution_client.py`, which forwards to a separate isolated
execution service (`http://localhost:8001` by default) that keeps state per
conversation. There is no local Jupyter kernel client to hand a cell to.

That matters for the data workflow, because untargeted retrieval calls methods
on the live objects the detected variables name. **The unit of "the same
kernel" is the `conversation_id`, not the notebook.** A data cell must be
submitted under the same `conversation_id` that already ran the notebook's
data-loading and filtering cells, or the objects will not exist. The service
exposes `get_conversation_variables(conversation_id)` if the agent needs to
confirm what that session currently holds.

For a selected dataset, pass the dataset names to
`build_dataset_cell(..., dataset_names=names)` after applying the same target
rules as `cite_data()`.

For a LiPDGraph notebook that uses a non-default endpoint, pass the endpoint
that the notebook queried when building the data cell. The generated cell must
retrieve from the same graph repository; otherwise the citation request can be
silently directed at a different data source.

### Saved notebook: use the path-based workflows

For a backend that has a materialized notebook file but does not need to run a
cell immediately, use:

```python
from provenance_agent import cite_data, cite_software

software_libraries = cite_software(notebook_path)
dataset_pairs = cite_data(notebook_path, targets=targets)
```

These calls analyze the file and add generated cells to it. They return the
inputs used to build those cells, not citation DataFrames. A later execution
run is still required for the data citation output. This path is useful for an
export or file-based workflow, and pairs naturally with
`backend/services/notebook_export_service.py`. It is less suitable when the
agent already holds notebook content and an active conversation.

### Natural-language routing: `run()` is the classifier inside the agent

PaleoPAL's registry routes on `request.agent_type` and `request.capability`. It
decides *which agent* handles a request; it does not decide whether a
provenance request concerns software, data, or both. That sub-decision still
belongs to the provenance agent, and `run()` is exactly the piece that makes
it:

```python
from provenance_agent.agent import run

result = run(
    request.user_input,
    notebook_path=notebook_path,
    model=paleopal_model,
)
```

Get `paleopal_model` from `LLMProviderFactory.get_langchain_model()` in
`backend/services/llm_providers.py`, which returns a LangChain `BaseChatModel`.
Do not pass a bare `LLMProvider`: its `generate_response()` returns a plain
string and is not a Runnable, so the LCEL chain cannot use it. Passing `model=`
keeps provider selection and credentials in the host and prevents
`provenance-agent` from constructing a second model client.

`run()` uses that model only for request classification; the workflows still
perform static detection and inject the citation cells. It does not execute
those cells, so the agent must handle the execution step and surface its
result.

Call the direct functions or builders instead of `run()` only when the
capability itself already fixes the answer, for example a `cite_software`
capability whose schema names the libraries. In that case there is nothing left
to classify.

## Translate requests and targets

The package has two different target vocabularies:

| User/PaleoPAL target | Package meaning |
| --- | --- |
| `pandas`, `xarray` | An imported software library for `cite_software()` or `run()`. |
| `TR04EVLI`, a study name, or a dataset ID | A data target passed through `targets=`. |
| `df_filtered`, `D`, or another notebook variable | Internal detector output, not a user-facing data target. |

For data, detection returns pairs such as `["df_filtered", "LiPDGraph"]`.
Those pairs tell the generated cell which live variable and provider to use;
they are not names that should be shown as dataset choices in the PaleoPAL
interface.

Specific PyLiPD and LiPDGraph names can be passed into targeted retrieval. A
specific PyleoTUPS study cannot be validated before the notebook's provider
object runs, so the current workflow warns and leaves the notebook unchanged
for that request. An integration should expose that as a limitation or offer
the user an all-datasets request rather than treating the notebook variable as
the study name.

## Return values and execution results

The static and runtime parts have deliberately different contracts:

| Interface | Returns | Does not return |
| --- | --- | --- |
| `build_metadata_cell(...)` | Python source for a software citation cell | Citation metadata or execution output |
| `build_dataset_cell(...)` | Python source for a dataset retrieval cell | Retrieved citations before the source is run |
| `cite_software(...)` | Imported library names included in the cell | The software DataFrame |
| `cite_data(...)` | `[variable, tool]` pairs included in the cell | The data metadata frames |
| `run(...)` | Classification, dispatch, and static-verification envelope | Proof that a generated cell executed successfully |

The agent should therefore treat cell generation and cell execution as two
separate stages. A successful `run()` or `cite_data()` call means that the
notebook was analyzed and a cell was prepared; it does not mean that a remote
dataset was reached or that a citation was retrieved.

For interactive use, capture the display and error output from the same
execution that ran the generated source, and put it in the `AgentResponse`
`result` while that context is still available. The generated `_bib_*` and
`_meta_*` bindings are useful inside the execution session, but they are not a
reliable transport format across restarts or persisted-state boundaries.

Warnings are part of the normal contract. An unclear request, an unsupported
target, or a missing dataset lineage can result in a warning and no notebook
mutation. Those map onto `AgentStatus.SUCCESS` with an explanatory `message`,
or `AgentStatus.NEEDS_CLARIFICATION` when the user could usefully rephrase.
Runtime failures from a remote retrieval or a missing dependency occur later,
when the generated cell executes, and should be reported as
`AgentStatus.ERROR` rather than confused with classification warnings.

## The notebook-context gap

`AgentRequest` carries `notebook_context: Dict[str, Any]`, documented as
"Notebook variables and cell context". It never carries a notebook path. This
package's public detector and path-based workflows accept a path, not notebook
JSON held in memory.

This is not an edge case for unsaved notebooks; it is the normal path for every
provenance request. The smallest current adapter is to materialize
`notebook_context` into a temporary `.ipynb` for the static analysis, and keep
the conversation's execution session for running the generated source. A future
content-based entry point in this package would remove that temporary-file
boundary; it is an API improvement, not a reason to move dataset detection into
PaleoPAL.

The software cell builder can work from a library list directly, so a request
that already names its libraries can skip materialization. Dataset detection
still needs the notebook path today.

## Integration acceptance tests

An integration is ready when these cases have been exercised:

1. The provenance agent is registered in `initialize_agents()` and appears in
   `AgentRegistry.list_agents()` with its capabilities and schemas.
2. A software request produces a software cell, the host executes it, and the
   resulting metadata output reaches the PaleoPAL response path.
3. A data request is submitted under the same `conversation_id` that ran the
   notebook's data-loading cells, and returns the retrieval output or a clear
   runtime error.
4. A selected software library is matched against imports, while a selected
   dataset is matched by dataset name or ID rather than notebook variable name.
5. An unsupported or unclear request produces a warning without changing the
   notebook, and maps onto the right `AgentStatus`.
6. If `run()` is used, the host model is injected through `model=` using
   `LLMProviderFactory.get_langchain_model()`, and the agent handles the
   separate cell-execution step.
7. The `notebook_context` materialization path is tested with unsaved changes.
