# Integrating `provenance-agent` with PaleoPAL

This guide describes the current interfaces and the work required to embed
`provenance-agent` in a PaleoPAL agent, extension, or notebook service.

## Integration boundary

`provenance-agent` is a standalone package. It must work without importing
PaleoPAL, and PaleoPAL may depend on it without becoming a dependency of the
package.

The package owns:

- detecting imported software and dataset references;
- looking up software citations and retrieving dataset metadata;
- generating the notebook cells that display those citations;
- optional natural-language routing through an LLM.

PaleoPAL should own the conversation, user interface, notebook session, and
kernel execution. Dataset retrieval remains in `provenance-agent`; the
integration should not delegate it to a separate PaleoPAL agent.

## Public interfaces

| Interface | What it does | Important behavior |
| --- | --- | --- |
| `cite_software(notebook_path, ...)` | Finds imported libraries and adds a software-citation cell. | Detection is static. The function returns the library names used to build the cell, not the citation DataFrame. |
| `cite_data(notebook_path, ...)` | Detects datasets and adds a dataset-retrieval cell. | Detection is static, but retrieval happens only when the generated cell runs in a kernel. The function returns `[variable, tool]` pairs, not the retrieved citations. |
| `run(request, notebook_path, model=None)` | Classifies a natural-language request and dispatches the appropriate workflow. | The notebook is modified in place. It returns a structured result and performs static verification; it does not execute the new cells. |
| `build_metadata_cell(...)` | Returns the source for a software-citation cell. | Use this when the host already controls cell execution. |
| `build_dataset_cell(...)` | Returns the source for one dataset-citation cell. | Use this with a live kernel. The cell displays the metadata frames after retrieval. |

The direct functions are defined in `provenance_agent.software` and
`provenance_agent.data`. The router is in `provenance_agent.agent`:

```python
from provenance_agent.agent import run

result = run(
    "cite the software and datasets",
    notebook_path="analysis.ipynb",
    model=model,  # optional PaleoPAL LangChain-compatible model
)
```

The `%provenance` magic is a notebook convenience layer. A backend or service
should call the Python interfaces instead of depending on IPython magic.

## Choose an integration strategy

### Host has a live notebook kernel

Use the cell builders. Detect or resolve the targets, build the source, and
execute it through PaleoPAL's existing execution client in the same kernel as
the notebook. This keeps loaded `pylipd`, `pyleotups`, and LiPDGraph objects
available for retrieval.

```python
from provenance_agent.data import build_dataset_cell
from provenance_agent.software import build_metadata_cell

software_source = build_metadata_cell(libraries)
data_source = build_dataset_cell(detected_pairs, endpoint=endpoint)

# Pseudocode: use the execution method provided by PaleoPAL.
await execution_client.run_cell(software_source)
await execution_client.run_cell(data_source)
```

`build_retrieval_cell()` returns a fragment for one dataset; use
`build_dataset_cell()` when a complete cell is needed. The exact execution
client method depends on the PaleoPAL host.

This approach is the best fit for an interactive PaleoPAL notebook because it
does not require an unsaved notebook to exist on disk. The host can insert the
generated source into the notebook and display or return the execution output.

### Host has a notebook file but no live kernel

Use `cite_software()` or `cite_data()` with the path to an `.ipynb` file. Both
functions add generated cells to that file, replacing the package's previous
generated cells on a repeat run.

This works well for an exported notebook or a backend that has a materialized
copy. It does not produce dataset citation output until the generated data
cell is later run in a notebook kernel.

### Host has notebook content but no path

The current direct APIs read and write `.ipynb` paths with `nbformat`. A
PaleoPAL conversation or an unsaved VS Code notebook may have notebook JSON in
memory without a stable path. An integration must either:

1. materialize that content as a temporary or working `.ipynb` file; or
2. add a content-based entry point that accepts notebook JSON and returns the
   modified notebook or generated cells.

The second option is the better long-term interface for live editors. It keeps
file handling and cell insertion with the host while preserving the existing
path-based API for standalone use.

## Supplying PaleoPAL's model

Pass PaleoPAL's existing LangChain-compatible chat model to `run()`:

```python
from provenance_agent.agent import run

model = LLMProviderFactory.get_langchain_model(
    provider_type=provider_type,
    model_name=model_name,
)

result = run(user_request, notebook_path, model=model)
```

Use PaleoPAL's current import path for `LLMProviderFactory`. The `model`
argument is the integration point: it avoids constructing the standalone
provider client in `provenance-agent` and avoids requiring that package's
provider extra.

The model should return the structured classification expected by the router.
PaleoPAL's current wrapper is responsible for normalizing provider-specific
reasoning preambles and JSON responses before the router parses them.

The result has this shape:

```python
{
    "status": "ok" or "warning",
    "decision": {...},
    "dispatch": {...},
    "verification": {...},
    "warning": str | None,
}
```

- `status="ok"` means the request was classified and a workflow ran.
- `status="warning"` means the request was unclear or unsupported. It is a
  safe no-op, not an exception, and the notebook is not changed.
- `dispatch` describes the requested workflow.
- `verification` reports the static cell changes. It does not prove that a
  generated cell ran successfully.

If `model` is omitted, the package uses its own lazy provider registry and
environment configuration. PaleoPAL can use compatible environment variables
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and
`XAI_API_KEY`), but passing its model is less ambiguous when both projects
load `.env` files.

## Execution and dependency requirements

The direct APIs are synchronous. If they are called from an asynchronous
PaleoPAL request handler, run them off the event loop:

```python
import asyncio

result = await asyncio.to_thread(
    run,
    user_request,
    notebook_path,
    model,
)
```

The process that executes generated dataset cells needs the dataset retrieval
dependencies, `pylipd` and `pyleotups`. These are not core dependencies: they
are the opt-in `data` extra (`pip install "provenance-agent[data]"`), because
the package itself never imports them - only the generated cell source does.
Install them wherever cells are executed, not necessarily where the package
generates them. Note that a missing one surfaces as a plain
`ModuleNotFoundError` raised inside the executed cell, not as a friendly
install message.

The notebook's own scientific packages, such as `pyleoclim` and `xarray`, still
need to be installed in the notebook environment when its cells require them.
Passing a model to `run()` does not install or load any provider integration.

PaleoPAL should also account for its execution service's state rules:

- Generated citation and metadata bindings use underscore-prefixed names such
  as `_bib_<variable>` and `_meta_<variable>`.
- If persisted kernel state drops underscore-prefixed variables, those names
  cannot be relied on in a later execution request.
- Non-picklable objects may be skipped when state is persisted. Consume the
  result during the same execution, return it through the execution response,
  or change the host's persistence policy rather than assuming the object will
  be available later.
- Untargeted retrieval may reuse objects already loaded by the notebook.
  Targeted retrieval can load a fresh LiPD record and is safer when persistence
  across execution calls is uncertain.

## Keep these boundaries

An integration should preserve the following behavior:

1. Do not add a PaleoPAL import to the standalone package.
2. Keep dataset detection and retrieval in `provenance-agent`.
3. Keep cell generation separate from notebook mutation. Hosts with a live
   kernel should use builders and manage insertion themselves.
4. Keep LLM clients lazy. Importing the package and loading `%provenance`
   should not require credentials.
5. Keep dataset detection deterministic. The LLM routes the request; it does
   not decide whether a notebook variable is a dataset.
6. Treat warnings as non-mutating responses and map them to PaleoPAL's normal
   clarification or warning flow.

## Integration checklist

Before shipping an integration, verify that it:

1. chooses a host surface: live notebook, backend agent, or export hook;
2. provides notebook content or a valid `.ipynb` path;
3. decides whether the host will inject cells or execute builders directly;
4. passes PaleoPAL's model to `run()` when using natural-language routing;
5. runs synchronous calls outside the async event loop;
6. installs retrieval dependencies in the environment that runs the cells;
7. returns execution output instead of depending on underscore-prefixed state;
8. tests both an already-saved notebook and an unsaved or in-memory notebook;
9. tests a live data retrieval, not only static cell generation.
