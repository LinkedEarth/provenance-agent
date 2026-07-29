"""
Inspectable LCEL routing pipeline for software and dataset provenance.

Purpose:
    Accept one natural-language citation request, classify its typed targets,
    dispatch the existing notebook workflows in place, and return a structured
    verification envelope describing what was injected. The pipeline replaces
    hidden one-shot model tool-calling with named Runnable stages composed by
    the LCEL ``|`` operator.

Implementation:
    ``prepare_context`` reads imported library names from the notebook;
    ``classify`` parses a Pydantic-validated JSON decision from the model;
    ``resolve_targets`` validates imported software and runs dataset detection
    only when data is requested; ``dispatch`` calls the existing orchestrator
    functions sequentially; and ``verify`` compares the notebook before and
    after dispatch, without executing injected cells. ``run`` returns the
    final JSON-serializable envelope, while the IPython magic formats that
    envelope for human readers.

Design decisions:
    - Multiple named targets are explicitly typed as ``software`` or ``data``
      so dispatch never guesses a target's category after classification.
    - An unclear request, missing requested software import, or data request
      with no detected data variables returns a warning and does not mutate the
      notebook. An unmatched target is allowed for PyLiPD/LiPDGraph data-name
      loading, but a specific PyleoTUPS study-name request warns and does not
      dispatch or mutate the notebook because its available names are only
      known inside the live provider object.
    - The existing StructuredTool wrappers remain public for direct callers,
      but they are not bound to the classification model.
    - ``build_chain(model=...)`` supports fake Runnable models in offline tests;
      the module-level ``chain`` uses the configured Gemini client.
"""

from __future__ import annotations

from collections import Counter
import json
import sys
from typing import Literal

import nbformat
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field

from llm import llm
from notebook_parser import PROVENANCE_CELL_MARKER
from orchestrator import (
    cite_data,
    cite_data_tool,
    cite_software,
    cite_software_tool,
)


class TypedTarget(BaseModel):
    """A user-named target with an explicit software or data category."""

    kind: Literal["software", "data"]
    name: str = Field(min_length=1)


class RouteDecision(BaseModel):
    """The validated JSON decision produced by the classification stage."""

    action: Literal["cite", "warning"]
    scope: Literal["all", "selected"]
    targets: list[TypedTarget] = Field(default_factory=list)
    kinds: list[Literal["software", "data"]] = Field(default_factory=list)
    fmt: Literal["apa", "bibtex"] = "apa"
    warning: str | None = None


_CLASSIFIER_PARSER = PydanticOutputParser(pydantic_object=RouteDecision)

SYSTEM_PROMPT = (
    "Classify a provenance request for a Jupyter notebook. The request can "
    "concern software, data, both, or be unclear.\n\n"
    "Software means an imported Python library. Data means a dataset or "
    "dataset variable.\n\n"
    "The notebook path is `{notebook_path}`.\n\n"
    "For 'cite everything', return action 'cite', scope 'all', and kinds "
    "['software', 'data']. For software-only or data-only requests, return "
    "scope 'all' with the corresponding kind.\n\n"
    "For specifically named requests, return scope 'selected' and one typed "
    "target per named item: {{\"kind\": \"software\"|\"data\", \"name\": \"...\"}}.\n\n"
    "If a requested software library is not in the imported-library list, "
    "return action 'warning' and explain that it is not imported. If the "
    "request is unclear or unrelated, return action 'warning' and explain why.\n\n"
    "The following dataset names are examples only: TR04EVLI, SP13CHPE, "
    "IC13THQU, SP08VBPE, SP12LAMX, SP12KEBZ, LS09BAUM, LS11BIPU, "
    "CuevadelTigrePerdido.vanBreukelen.2008, Huagapocave.Kanner.2013, "
    "ElCondorcave.Cheng.2013, MacalChasm.Akers.2016, "
    "Ocn-RedSea.Felis.2000.\n\n"
    "Return only JSON matching the provided schema. Do not provide reasoning "
    "or additional text."
)

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        SYSTEM_PROMPT
        + "\nImported libraries: {imports}\n\n{format_instructions}",
    ),
    ("human", "{request}"),
]).partial(format_instructions=_CLASSIFIER_PARSER.get_format_instructions())


_TOOLS = [cite_software_tool, cite_data_tool]
_TOOLS_BY_NAME = {tool.name: tool for tool in _TOOLS}


def build_messages(request: str, notebook_path: str) -> list:
    """
    Builds the legacy two-message representation for callers that use it.

    Args:
        request: natural-language citation request
        notebook_path: path to the notebook being analyzed

    Returns:
        a two-element list containing a system and human message
    """
    system = SYSTEM_PROMPT.replace("{notebook_path}", notebook_path)
    return [SystemMessage(system), HumanMessage(request)]


def _prepare_context(state: dict) -> dict:
    """Adds imported library names and a pre-dispatch notebook snapshot."""
    from notebook_parser import parse_notebook

    notebook_path = state["notebook_path"]
    return {
        **state,
        "imports": parse_notebook(notebook_path),
        "before_snapshot": _snapshot_cells(notebook_path),
    }


def _detect_dataset_pairs(notebook_path: str) -> list[list[str]]:
    """Runs the existing dataset detector for a data-bearing request."""
    from dataset_detection import detect_datasets
    from notebook_parser import read_notebook_code

    return detect_datasets(read_notebook_code(notebook_path))


def _warning_state(state: dict, message: str) -> dict:
    """Marks a pipeline state as a no-op warning before dispatch."""
    return {
        **state,
        "status": "warning",
        "warning": message,
        "resolved": None,
        "dispatch": [],
    }


def _resolve_targets(state: dict) -> dict:
    """Validates the model decision and resolves data detection if needed."""
    decision = state["decision"]
    decision_dict = decision.model_dump()
    state = {**state, "decision": decision_dict, "status": "ok"}

    if decision.action == "warning":
        return _warning_state(
            state,
            decision.warning or "The request could not be classified clearly.",
        )

    if decision.scope == "all":
        requested_kinds = set(decision.kinds)
        if not requested_kinds:
            return _warning_state(state, "No software or data target was specified.")
        software_targets = None
        data_targets = None
        software_requested = "software" in requested_kinds
        data_requested = "data" in requested_kinds
    else:
        software_targets = [
            target.name for target in decision.targets if target.kind == "software"
        ]
        data_targets = [
            target.name for target in decision.targets if target.kind == "data"
        ]
        if not decision.targets:
            return _warning_state(state, "No citation targets were specified.")
        software_requested = bool(software_targets)
        data_requested = bool(data_targets)

    imports = state["imports"]
    imports_by_lower = {name.casefold(): name for name in imports}
    if software_requested and software_targets is not None:
        missing = [
            name for name in software_targets
            if name.casefold() not in imports_by_lower
        ]
        if missing:
            return _warning_state(
                state,
                "These software targets are not imported: " + ", ".join(missing),
            )
        software_targets = [imports_by_lower[name.casefold()] for name in software_targets]
    elif software_requested and not imports:
        return _warning_state(state, "No imported software libraries were found.")

    detected_pairs: list[list[str]] = []
    runtime_unverified = False
    if data_requested:
        detected_pairs = _detect_dataset_pairs(state["notebook_path"])
        if not detected_pairs:
            return _warning_state(state, "No datasets were detected in the notebook.")
        from data_workflow import pyleotups_target_warning

        target_warning = pyleotups_target_warning(
            detected_pairs,
            data_targets,
        )
        if target_warning:
            return _warning_state(state, target_warning)
        if data_targets is not None:
            detected_variables = {pair[0].casefold() for pair in detected_pairs}
            runtime_unverified = any(
                target.casefold() not in detected_variables for target in data_targets
            )

    return {
        **state,
        "resolved": {
            "software_requested": software_requested,
            "software_targets": software_targets,
            "data_requested": data_requested,
            "data_targets": data_targets,
            "detected_pairs": detected_pairs,
            "runtime_unverified": runtime_unverified,
        },
    }


def _dispatch(state: dict) -> dict:
    """Runs the selected workflows sequentially against the same notebook."""
    if state.get("status") == "warning":
        return {**state, "dispatch": []}

    resolved = state["resolved"]
    notebook_path = state["notebook_path"]
    fmt = state["decision"]["fmt"]
    calls = []

    if resolved["software_requested"]:
        libraries = resolved["software_targets"]
        result = cite_software(notebook_path, libraries=libraries)
        args = {"notebook_path": notebook_path}
        if libraries is not None:
            args["libraries"] = libraries
        calls.append({"name": "cite_software", "args": args, "result": result})

    if resolved["data_requested"]:
        targets = resolved["data_targets"]
        result = cite_data(
            notebook_path,
            targets=targets,
            fmt=fmt,
            detected_pairs=resolved["detected_pairs"],
        )
        args = {"notebook_path": notebook_path, "fmt": fmt}
        if targets is not None:
            args["targets"] = targets
        calls.append({"name": "cite_data", "args": args, "result": result})

    return {**state, "dispatch": calls}


def _snapshot_cells(notebook_path: str) -> Counter:
    """Returns a multiset of cell types and source for static diffing."""
    with open(notebook_path) as handle:
        notebook = nbformat.read(handle, as_version=4)
    return Counter((cell.cell_type, cell.source) for cell in notebook.cells)


def _new_cells(before: Counter, notebook) -> list:
    """Returns cell instances added since a prior source multiset snapshot."""
    remaining = before.copy()
    added = []
    for cell in notebook.cells:
        key = (cell.cell_type, cell.source)
        if remaining[key]:
            remaining[key] -= 1
        else:
            added.append(cell)
    return added


def _cell_tool(source: str) -> str:
    """Classifies an injected cell by its software/data marker."""
    if "provenance_software" in source:
        return "software"
    if "provenance_datasets" in source:
        return "data"
    if PROVENANCE_CELL_MARKER in source:
        return "data"
    return "unknown"


def _verify(state: dict) -> dict:
    """Statically verifies the notebook changes and returns the public envelope."""
    notebook_path = state["notebook_path"]
    with open(notebook_path) as handle:
        notebook = nbformat.read(handle, as_version=4)

    if state.get("status") == "warning":
        mutated = _snapshot_cells(notebook_path) != state["before_snapshot"]
        verification = {
            "cells": [],
            "mutated": mutated,
            "runtime_unverified": False,
        }
        if mutated:
            state["warning"] += " The notebook changed unexpectedly."
        return _public_result(state, verification)

    added = _new_cells(state["before_snapshot"], notebook)
    summaries = [
        {
            "tool": _cell_tool(cell.source),
            "source_preview": cell.source[:240],
            "injected": True,
        }
        for cell in added
    ]
    expected = []
    if state["resolved"]["software_requested"]:
        expected.append("software")
    if state["resolved"]["data_requested"]:
        expected.append("data")
    # Presence in the final notebook, not newness. A re-run whose inputs have
    # not changed rewrites a byte-identical cell, so the added-cell diff is
    # empty even though the notebook is exactly as it should be.
    present = {_cell_tool(cell.source) for cell in notebook.cells}
    missing = [tool for tool in expected if tool not in present]
    verification = {
        "cells": summaries,
        "present": sorted(tool for tool in present if tool != "unknown"),
        "mutated": bool(added),
        "runtime_unverified": state["resolved"]["runtime_unverified"],
    }
    if missing:
        state = {
            **state,
            "status": "warning",
            "warning": (
                "Verification failed: missing injected "
                + ", ".join(missing) + " cell"
            ),
        }
    return _public_result(state, verification)


def _public_result(state: dict, verification: dict) -> dict:
    """Builds the JSON-serializable result returned by the public API."""
    result = {
        "status": state.get("status", "ok"),
        "decision": state.get("decision"),
        "dispatch": state.get("dispatch", []),
        "verification": verification,
    }
    if state.get("warning"):
        result["warning"] = state["warning"]
    return result


def build_chain(model: Runnable | None = None) -> Runnable:
    """
    Builds the inspectable LCEL provenance pipeline.

    Args:
        model: optional LangChain Runnable used for classification; omitted uses
            the configured Gemini client.

    Returns:
        a Runnable composed as prepare_context | classify | resolve_targets |
        dispatch | verify
    """
    classifier_model = model if model is not None else llm
    classify = _CLASSIFIER_PROMPT | classifier_model | _CLASSIFIER_PARSER
    prepare = RunnableLambda(_prepare_context, name="prepare_context")
    classify_assign = RunnablePassthrough.assign(decision=classify)
    classify_state = RunnableLambda(
        classify_assign.invoke,
        name="classify",
    )
    resolve = RunnableLambda(_resolve_targets, name="resolve_targets")
    dispatch = RunnableLambda(_dispatch, name="dispatch")
    verify = RunnableLambda(_verify, name="verify")
    return prepare | classify_state | resolve | dispatch | verify


chain = build_chain()


def _parser_warning_envelope(message: str) -> dict:
    """Returns a warning result for malformed classifier output."""
    return {
        "status": "warning",
        "decision": None,
        "dispatch": [],
        "verification": {
            "cells": [],
            "mutated": False,
            "runtime_unverified": False,
        },
        "warning": message,
    }


def run(request: str, notebook_path: str) -> dict:
    """
    Runs the LCEL pipeline and returns its structured JSON envelope.

    Args:
        request: natural-language citation request
        notebook_path: notebook to inspect and mutate in place

    Returns:
        a JSON-serializable dictionary containing classification, dispatch, and
        static verification results
    """
    try:
        return chain.invoke({
            "request": request,
            "notebook_path": notebook_path,
        })
    except OutputParserException as error:
        return _parser_warning_envelope(
            "Could not classify the request clearly: " + str(error)
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python agent.py <notebook.ipynb> "<request>"')
        sys.exit(1)
    print(json.dumps(run(sys.argv[2], sys.argv[1]), indent=2))
