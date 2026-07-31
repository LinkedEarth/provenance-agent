"""
Deterministic, source-oriented dataset detection for Jupyter notebooks.

Purpose:
    Identify external dataset sources whose values reach a recognized scientific
    analysis operation, without executing the notebook or calling an LLM. The
    public entry point accepts only a notebook path and returns the repository's
    existing list-of-pairs contract, such as
    [["filtered_df2", "LiPDGraph"]].

Implementation:
    The scanner parses code cells in notebook order and builds a small,
    versioned data-flow graph. Every assigned value records its dependencies,
    source groups, object family, and source position. Recognizers attach source
    groups to LiPD, PyleoTUPS, LiPDGraph, xarray, and pandas loader patterns.
    Analysis calls are recorded as sinks. Results are produced by walking each
    sink's dependency closure and selecting the nearest source-family boundary.

Design decisions:
    - This module is standalone. It does not replace or import the active LLM
      detector, data workflow, agent, or orchestrator.
    - The scanner is conservative: syntax errors, opaque helper functions, and
      dynamic imports do not create guessed source lineage.
    - Source objects are not reported merely because they were loaded or
      searched. Their lineage must reach a configured analysis sink.
    - Notebook variables are internal result labels. User-facing target
      selection remains dataset-name based in the separate workflow layer.
    - Analysis and inspection registries are centralized so extending the
      supported scientific vocabulary does not require changing graph logic.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import nbformat

from notebook_parser import is_generated_cell, strip_ipython_directives


__all__ = ("detect_datasets_in_notebook",)


_GRAPH_ENDPOINT_PREFIX = (
    "https://linkedearth.graphdb.mint.isi.edu/repositories/"
)

# These sets are intentionally module-level registries. They are the only
# vocabulary decisions the graph needs to make; users do not configure them.
ANALYSIS_METHODS = frozenset({
    "anova",
    "bootstrap",
    "classify",
    "cluster",
    "clustering",
    "coherence",
    "correlation",
    "corr",
    "cov",
    "covariance",
    "crossvalidate",
    "decompose",
    "decomposition",
    "eof",
    "eofs",
    "eofsascovariance",
    "eofscorrelation",
    "fft",
    "fit",
    "fittransform",
    "forecast",
    "f_oneway",
    "infer",
    "kmeans",
    "kendalltau",
    "linregress",
    "mannwhitney",
    "minimize",
    "optimize",
    "pca",
    "pearsonr",
    "predict",
    "regress",
    "regression",
    "score",
    "spearmanr",
    "spectral",
    "spectrum",
    "ssa",
    "svd",
    "t_test",
    "ttest",
    "train",
    "transform",
    "wavelet",
    "wilcoxon",
})

INSPECTION_METHODS = frozenset({
    "columns",
    "display",
    "get_all_dataset_names",
    "get_publications",
    "head",
    "info",
    "len",
    "map",
    "modeplot",
    "nunique",
    "plot",
    "print",
    "screeplot",
    "search_studies",
    "shape",
    "show",
    "stackplot",
    "type",
    "unique",
})

_PYLEO_LOAD_METHODS = frozenset({
    "load",
    "load_from_dir",
    "load_datasets",
    "load_remote_datasets",
    "search_studies",
})

_PYLEO_DATA_METHODS = frozenset({
    "get_data",
    "get_datasets",
    "get_timeseries",
    "get_timeseries_essentials",
})

_PYLEO_PUBLICATION_METHODS = frozenset({
    "get_bibtex",
    "get_publications",
})

_XARRAY_LOADERS = frozenset({
    "load_dataset",
    "open_dataarray",
    "open_dataset",
    "open_mfdataset",
    "open_zarr",
})

_PANDAS_READERS = frozenset({
    "read_csv",
    "read_excel",
    "read_fwf",
    "read_hdf",
    "read_json",
    "read_parquet",
    "read_pickle",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_table",
})

_XARRAY_DATASET_METHODS = frozenset({
    "assign",
    "assign_coords",
    "chunk",
    "compute",
    "drop",
    "drop_vars",
    "fillna",
    "load",
    "mean",
    "max",
    "min",
    "persist",
    "rename",
    "resample",
    "sel",
    "sortby",
    "sum",
    "transpose",
    "where",
})

_XARRAY_INTERMEDIATE_METHODS = frozenset({
    "groupby",
    "resample",
    "weighted",
})

_TABLE_METHODS = frozenset({
    "assign",
    "copy",
    "drop",
    "drop_duplicates",
    "dropna",
    "fillna",
    "filter",
    "join",
    "merge",
    "query",
    "reset_index",
    "sort_values",
    "astype",
})


# Static extraction algorithm:
# 1. Parse each non-generated code cell in notebook order; skip cells that
#    cannot be parsed rather than executing or guessing through them.
# 2. Version every assigned name. Each version stores dependencies and the
#    source groups inherited from those dependencies.
# 3. Recognize external source constructors/loaders and object-loading methods,
#    then attach a canonical tool to the resulting value lineage.
# 4. Record only calls from ANALYSIS_METHODS. Calls in INSPECTION_METHODS do not
#    become sinks, even when they receive a source-backed value.
# 5. For each sink, walk dependencies backward and choose a stable boundary:
#    the latest table, xarray, or source-object value for each source group.
# 6. Deduplicate pairs and sort them by source position and variable name.


@dataclass
class _Source:
    """One recognized external source lineage."""

    source_id: int
    tool: str
    family: str
    order: tuple[int, int, int, int]
    name: str | None = None
    active: bool = False


@dataclass
class _Value:
    """One versioned assigned value in the static dependency graph."""

    value_id: int
    name: str
    order: tuple[int, int, int, int]
    deps: frozenset[int]
    source_ids: frozenset[int]
    family: str = "unknown"
    kind: str = "unknown"
    literal: Any = None


@dataclass(frozen=True)
class _Eval:
    """Static information computed for an expression before assignment."""

    deps: frozenset[int] = frozenset()
    source_ids: frozenset[int] = frozenset()
    family: str = "unknown"
    kind: str = "unknown"
    literal: Any = None
    created_sources: frozenset[int] = frozenset()


@dataclass
class _Sink:
    """A recognized analysis call and the values it consumes."""

    name: str
    order: tuple[int, int, int, int]
    deps: frozenset[int]
    source_ids: frozenset[int]
    prefer_dataset: bool = False


def _simple_name(node: ast.AST | None) -> str | None:
    """Returns a direct variable name, or None for a compound expression."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _qualified_name(node: ast.AST | None) -> str:
    """Builds a dotted static name from Name/Attribute syntax."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _last_name(name: str) -> str:
    """Returns the final component of a dotted or underscored callable name."""
    return name.rsplit(".", 1)[-1]


def _normalized_name(name: str) -> str:
    """Normalizes a method name for registry matching."""
    return name.casefold().replace("-", "").replace("_", "")


def _is_analysis_sink(name: str) -> bool:
    """Reports whether a callable name belongs to the analysis registry."""
    normalized = _normalized_name(_last_name(name))
    if normalized in {
        _normalized_name(item) for item in INSPECTION_METHODS
    }:
        return False
    return (
        normalized in {_normalized_name(item) for item in ANALYSIS_METHODS}
        or normalized.startswith("eof")
        or normalized.endswith("regression")
    )


def _constant_value(node: ast.AST) -> Any:
    """Returns a literal represented by an AST node, or None if unresolved."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            if not isinstance(value, ast.Constant):
                return None
            pieces.append(str(value.value))
        return "".join(pieces)
    return None


def _choose_family(evals: Iterable[_Eval], fallback: str = "unknown") -> str:
    """Chooses a non-unknown family while combining expression dependencies."""
    families = [item.family for item in evals if item.family != "unknown"]
    if not families:
        return fallback
    first = families[0]
    return first if all(item == first for item in families) else first


def _combine_evals(
    evals: Iterable[_Eval],
    family: str | None = None,
    kind: str | None = None,
) -> _Eval:
    """Unions static expression information from several child expressions."""
    items = list(evals)
    literals = [item.literal for item in items if item.literal is not None]
    return _Eval(
        deps=frozenset(dep for item in items for dep in item.deps),
        source_ids=frozenset(
            source_id for item in items for source_id in item.source_ids
        ),
        family=family or _choose_family(items),
        kind=kind or (items[0].kind if items else "unknown"),
        literal=literals[0] if len(literals) == 1 else None,
        created_sources=frozenset(
            source_id
            for item in items
            for source_id in item.created_sources
        ),
    )


class _NotebookGraph(ast.NodeVisitor):
    """Builds a conservative versioned graph for one notebook."""

    def __init__(self) -> None:
        self.env: dict[str, _Value] = {}
        self.values: dict[int, _Value] = {}
        self.sources: dict[int, _Source] = {}
        self.sinks: list[_Sink] = []
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.cell_index = 0
        self._sequence = 0
        self._next_value_id = 0
        self._next_source_id = 0

    def process(self, tree: ast.AST, cell_index: int) -> None:
        """Visits one parsed notebook cell in source order."""
        self.cell_index = cell_index
        self.visit(tree)

    def _position(self, node: ast.AST) -> tuple[int, int, int, int]:
        """Returns a stable source position for a node."""
        self._sequence += 1
        return (
            self.cell_index,
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
            self._sequence,
        )

    def _new_source(
        self,
        tool: str,
        family: str,
        node: ast.AST,
        active: bool,
    ) -> int:
        """Creates a source group and returns its stable ID."""
        source_id = self._next_source_id
        self._next_source_id += 1
        self.sources[source_id] = _Source(
            source_id=source_id,
            tool=tool,
            family=family,
            order=self._position(node),
            active=active,
        )
        return source_id

    def _new_value(
        self,
        name: str,
        evaluation: _Eval,
        node: ast.AST,
    ) -> _Value:
        """Adds a versioned value and assigns source constructor names."""
        for source_id in evaluation.created_sources:
            source = self.sources[source_id]
            if source.name is None:
                source.name = name
        value = _Value(
            value_id=self._next_value_id,
            name=name,
            order=self._position(node),
            deps=evaluation.deps,
            source_ids=evaluation.source_ids,
            family=evaluation.family,
            kind=evaluation.kind,
            literal=evaluation.literal,
        )
        self._next_value_id += 1
        self.values[value.value_id] = value
        return value

    def _assign_name(self, name: str, evaluation: _Eval, node: ast.AST) -> _Value:
        """Creates a value version and makes it the current binding."""
        value = self._new_value(name, evaluation, node)
        self.env[name] = value
        return value

    def _eval_value(self, value: _Value) -> _Eval:
        """Converts a graph value into expression information."""
        return _Eval(
            deps=frozenset({value.value_id}),
            source_ids=value.source_ids,
            family=value.family,
            kind=value.kind,
            literal=value.literal,
        )

    def _new_source_eval(
        self,
        tool: str,
        family: str,
        kind: str,
        node: ast.AST,
        active: bool,
    ) -> _Eval:
        """Creates expression information for a new external source."""
        source_id = self._new_source(tool, family, node, active)
        return _Eval(
            family=family,
            kind=kind,
            source_ids=frozenset({source_id}),
            created_sources=frozenset({source_id}),
        )

    def _activate_receiver(
        self,
        receiver_node: ast.AST,
        receiver: _Eval,
        arguments: Iterable[_Eval],
        method: str,
        node: ast.AST,
    ) -> _Eval:
        """Marks a source object as loaded and versions a direct receiver."""
        argument_list = list(arguments)
        argument_sources = frozenset(
            source_id
            for item in argument_list
            for source_id in item.source_ids
        )
        source_ids = receiver.source_ids | argument_sources
        if method in _PYLEO_LOAD_METHODS:
            for source_id in source_ids:
                self.sources[source_id].active = True

        receiver_name = _simple_name(receiver_node)
        if receiver_name and (
            receiver.source_ids
            or argument_sources
        ):
            current = self.env.get(receiver_name)
            if current is not None:
                updated = _Eval(
                    deps=frozenset(
                        {current.value_id}
                        | {
                            dep
                            for item in argument_list
                            for dep in item.deps
                        }
                    ),
                    source_ids=source_ids,
                    family=current.family,
                    kind=current.kind,
                    literal=current.literal,
                )
                self._assign_name(receiver_name, updated, node)
                return self._eval_value(self.env[receiver_name])
        return _Eval(
            deps=receiver.deps
            | frozenset(
                dep for item in argument_list for dep in item.deps
            ),
            source_ids=source_ids,
            family=receiver.family,
            kind=receiver.kind,
        )

    def _mutate_collection(
        self,
        receiver_node: ast.AST,
        receiver: _Eval,
        arguments: Iterable[_Eval],
        node: ast.AST,
    ) -> _Eval:
        """Propagates source lineage through list append/extend operations."""
        argument_list = list(arguments)
        receiver_name = _simple_name(receiver_node)
        source_ids = receiver.source_ids | frozenset(
            source_id
            for item in argument_list
            for source_id in item.source_ids
        )
        if receiver_name and receiver_name in self.env:
            current = self.env[receiver_name]
            updated = _Eval(
                deps=frozenset({current.value_id})
                | frozenset(
                    dep for item in argument_list for dep in item.deps
                ),
                source_ids=source_ids,
                family="list",
                kind="list",
            )
            self._assign_name(receiver_name, updated, node)
            return self._eval_value(self.env[receiver_name])
        return _Eval(
            deps=receiver.deps
            | frozenset(dep for item in argument_list for dep in item.deps),
            source_ids=source_ids,
            family="list",
            kind="list",
        )

    # ------------------------------------------------------------------ AST

    def visit_Import(self, node: ast.Import) -> None:
        """Ignores imports; recognizers use stable call spelling."""

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Ignores imports; recognizers use stable call spelling."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Stores function definitions without executing opaque bodies."""
        self.functions[node.name] = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Stores async definitions without executing opaque bodies."""
        self.functions[node.name] = node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Skips class bodies because dynamic methods are not resolved."""

    def visit_Assign(self, node: ast.Assign) -> None:
        """Versions every target in a regular assignment."""
        evaluation = self._eval(node.value)
        for target in node.targets:
            self._assign_target(target, evaluation, node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Versions an annotated assignment when it has a value."""
        if node.value is not None:
            self._assign_target(node.target, self._eval(node.value), node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Preserves lineage through in-place assignments."""
        target_eval = self._eval(node.target)
        value_eval = self._eval(node.value)
        evaluation = _combine_evals(
            [target_eval, value_eval],
            family=target_eval.family,
            kind=target_eval.kind,
        )
        self._assign_target(node.target, evaluation, node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> _Eval:
        """Tracks a walrus binding as well as its expression value."""
        evaluation = self._eval(node.value)
        self._assign_target(node.target, evaluation, node)
        return evaluation

    def visit_Expr(self, node: ast.Expr) -> None:
        """Evaluates calls used only for their side effects."""
        self._eval(node.value)

    def visit_For(self, node: ast.For) -> None:
        """Analyzes a loop body once and propagates iterable lineage."""
        iterable = self._eval(node.iter)
        loop_value = _Eval(
            deps=iterable.deps,
            source_ids=iterable.source_ids,
            family=iterable.family,
            kind="loop_item",
        )
        self._assign_target(node.target, loop_value, node)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Analyzes an async loop body with the same conservative rules."""
        iterable = self._eval(node.iter)
        loop_value = _Eval(
            deps=iterable.deps,
            source_ids=iterable.source_ids,
            family=iterable.family,
            kind="loop_item",
        )
        self._assign_target(node.target, loop_value, node)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_While(self, node: ast.While) -> None:
        """Visits a while body once; iteration count is intentionally static."""
        self._eval(node.test)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_If(self, node: ast.If) -> None:
        """Analyzes both branches and merges their visible bindings."""
        self._eval(node.test)
        before = dict(self.env)

        self.env = dict(before)
        for statement in node.body:
            self.visit(statement)
        body_env = dict(self.env)

        self.env = dict(before)
        for statement in node.orelse:
            self.visit(statement)
        else_env = dict(self.env)

        self.env = dict(before)
        self._merge_environments([body_env, else_env], node)

    def visit_Try(self, node: ast.Try) -> None:
        """Analyzes try and handler paths, then merges their bindings."""
        before = dict(self.env)
        branch_envs = []

        self.env = dict(before)
        for statement in node.body:
            self.visit(statement)
        branch_envs.append(dict(self.env))

        for handler in node.handlers:
            self.env = dict(before)
            for statement in handler.body:
                self.visit(statement)
            branch_envs.append(dict(self.env))

        self.env = dict(before)
        self._merge_environments(branch_envs, node)
        for statement in node.orelse:
            self.visit(statement)
        for statement in node.finalbody:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        """Visits context expressions and their body without executing them."""
        for item in node.items:
            self._eval(item.context_expr)
            if item.optional_vars is not None:
                self._assign_target(
                    item.optional_vars,
                    self._eval(item.context_expr),
                    node,
                )
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Analyzes async context bodies conservatively."""
        for item in node.items:
            self._eval(item.context_expr)
            if item.optional_vars is not None:
                self._assign_target(
                    item.optional_vars,
                    self._eval(item.context_expr),
                    node,
                )
        for statement in node.body:
            self.visit(statement)

    def _merge_environments(
        self,
        environments: Iterable[dict[str, _Value]],
        node: ast.AST,
    ) -> None:
        """Creates a union value for names that differ across branches."""
        environments = list(environments)
        names = set().union(*(environment.keys() for environment in environments))
        for name in sorted(names):
            candidates = [
                environment[name]
                for environment in environments
                if name in environment
            ]
            if not candidates:
                continue
            if len(candidates) == 1:
                self.env[name] = candidates[0]
                continue
            evaluation = _Eval(
                deps=frozenset(value.value_id for value in candidates),
                source_ids=frozenset(
                    source_id
                    for value in candidates
                    for source_id in value.source_ids
                ),
                family=candidates[0].family,
                kind=candidates[0].kind,
                literal=(
                    candidates[0].literal
                    if all(value.literal == candidates[0].literal for value in candidates)
                    else None
                ),
            )
            self._assign_name(name, evaluation, node)

    def _assign_target(
        self,
        target: ast.AST,
        evaluation: _Eval,
        node: ast.AST,
    ) -> None:
        """Assigns an expression to names, destructuring targets, or cells."""
        if isinstance(target, ast.Name):
            self._assign_name(target.id, evaluation, node)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._assign_target(item, evaluation, node)
            return
        if isinstance(target, ast.Starred):
            self._assign_target(target.value, evaluation, node)
            return
        if isinstance(target, ast.Subscript):
            base_name = _simple_name(target.value)
            if base_name and base_name in self.env:
                current = self.env[base_name]
                combined = _Eval(
                    deps=frozenset({current.value_id}) | evaluation.deps,
                    source_ids=current.source_ids | evaluation.source_ids,
                    family=current.family,
                    kind=current.kind,
                    literal=current.literal,
                )
                self._assign_name(base_name, combined, node)
            return
        if isinstance(target, ast.Attribute):
            base_name = _simple_name(target.value)
            if base_name and base_name in self.env:
                current = self.env[base_name]
                combined = _Eval(
                    deps=frozenset({current.value_id}) | evaluation.deps,
                    source_ids=current.source_ids | evaluation.source_ids,
                    family=current.family,
                    kind=current.kind,
                    literal=current.literal,
                )
                self._assign_name(base_name, combined, node)

    def _merge_subscript_family(
        self,
        base: _Eval,
        target: ast.Subscript,
    ) -> tuple[str, str]:
        """Infers a family/kind for a subscripted dataset value."""
        if base.family == "xarray":
            if base.kind == "xarray_collection":
                return "xarray", "xarray_dataset"
            return "xarray", "xarray_dataarray"
        if base.family == "table":
            index = target.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                return "table", "table_cell"
            return "table", "table"
        return base.family, base.kind

    def _eval(self, node: ast.AST | None) -> _Eval:
        """Evaluates an expression into static dependency information."""
        if node is None:
            return _Eval()
        if isinstance(node, ast.Name):
            value = self.env.get(node.id)
            return self._eval_value(value) if value is not None else _Eval()
        if isinstance(node, ast.Constant):
            return _Eval(literal=node.value)
        if isinstance(node, ast.JoinedStr):
            return _Eval(literal=_constant_value(node))
        if isinstance(node, ast.Attribute):
            base = self._eval(node.value)
            if base.family == "xarray" and node.attr in {"data", "values"}:
                return _Eval(
                    deps=base.deps,
                    source_ids=base.source_ids,
                    family="xarray",
                    kind="xarray_dataarray",
                )
            if base.family == "response":
                return _Eval(
                    deps=base.deps,
                    source_ids=base.source_ids,
                    family="response",
                    kind="response_text",
                )
            return base
        if isinstance(node, ast.Subscript):
            base = self._eval(node.value)
            index = self._eval(node.slice)
            family, kind = self._merge_subscript_family(base, node)
            return _combine_evals(
                [base, index],
                family=family,
                kind=kind,
            )
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [self._eval(item) for item in node.elts]
            return _combine_evals(
                items,
                family="list" if isinstance(node, ast.List) else "tuple",
                kind="list" if isinstance(node, ast.List) else "tuple",
            )
        if isinstance(node, ast.Dict):
            items = [
                self._eval(item)
                for item in [*node.keys, *node.values]
                if item is not None
            ]
            return _combine_evals(items, family="collection", kind="dict")
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            return _combine_evals(
                [left, right],
                family=_choose_family([left, right]),
            )
        if isinstance(node, ast.BoolOp):
            return _combine_evals([self._eval(item) for item in node.values])
        if isinstance(node, ast.Compare):
            return _combine_evals(
                [self._eval(node.left)]
                + [self._eval(item) for item in node.comparators],
                family="boolean",
                kind="boolean",
            )
        if isinstance(node, ast.UnaryOp):
            return self._eval(node.operand)
        if isinstance(node, ast.IfExp):
            return _combine_evals([
                self._eval(node.test),
                self._eval(node.body),
                self._eval(node.orelse),
            ])
        if isinstance(node, ast.Starred):
            return self._eval(node.value)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            items = [self._eval(node.elt)]
            for generator in node.generators:
                items.append(self._eval(generator.iter))
                if generator.ifs:
                    items.extend(self._eval(item) for item in generator.ifs)
            return _combine_evals(items, family="list", kind="list")
        if isinstance(node, ast.DictComp):
            return _combine_evals(
                [self._eval(node.key), self._eval(node.value)]
                + [self._eval(generator.iter) for generator in node.generators],
                family="collection",
                kind="dict",
            )
        if isinstance(node, ast.NamedExpr):
            return self.visit_NamedExpr(node)
        return _Eval()

    # ------------------------------------------------------------- call rules

    def _call_url(self, node: ast.Call, evaluations: list[_Eval]) -> str | None:
        """Resolves a statically known URL argument from a call."""
        candidates = list(node.args)
        candidates.extend(keyword.value for keyword in node.keywords)
        for argument, evaluation in zip(candidates, evaluations):
            literal = evaluation.literal
            if isinstance(literal, str) and literal.startswith("http"):
                return literal
            literal = _constant_value(argument)
            if isinstance(literal, str) and literal.startswith("http"):
                return literal
        return None

    def _is_graph_post(self, name: str, url: str | None) -> bool:
        """Reports whether a call is a LinkedEarth graph request."""
        return (
            _last_name(name).casefold() == "post"
            and "requests" in name.casefold()
            and bool(url and _GRAPH_ENDPOINT_PREFIX in url)
        )

    def _is_pandas_reader(self, name: str) -> bool:
        """Reports whether a callable is a configured pandas reader."""
        last = _last_name(name).casefold()
        return (
            last in _PANDAS_READERS
            and (
                name.casefold().startswith(("pd.", "pandas."))
                or "." not in name
            )
        )

    def _constructor_tool(self, name: str) -> tuple[str, str, str] | None:
        """Maps a recognized dataset constructor to tool/family/kind."""
        last = _last_name(name).casefold()
        if last == "lipd":
            return "PyLiPD", "object", "pylipd_object"
        if last in {"pangaeadataset", "noaadataset"}:
            return "PyleoTUPS", "object", "pyleotups_object"
        return None

    def _is_xarray_loader(self, name: str) -> bool:
        """Reports whether a callable is a configured xarray loader."""
        last = _last_name(name).casefold()
        return last in _XARRAY_LOADERS and (
            name.casefold().startswith(("xr.", "xarray."))
            or "." not in name
        )

    def _record_sink(
        self,
        name: str,
        evaluation: _Eval,
        node: ast.Call,
    ) -> None:
        """Records a source-backed analysis call."""
        if not evaluation.source_ids or not _is_analysis_sink(name):
            return
        self.sinks.append(
            _Sink(
                name=_last_name(name),
                order=self._position(node),
                deps=evaluation.deps,
                source_ids=evaluation.source_ids,
                prefer_dataset=_normalized_name(_last_name(name)) in {
                    "eof",
                    "eofs",
                    "eofsascovariance",
                    "eofscorrelation",
                },
            )
        )

    def _eval_call(self, node: ast.Call) -> _Eval:
        """Recognizes source loaders, object methods, and analysis sinks."""
        name = _qualified_name(node.func)
        receiver_node = node.func.value if isinstance(node.func, ast.Attribute) else None
        receiver = self._eval(receiver_node)
        arguments = [self._eval(argument) for argument in node.args]
        arguments.extend(self._eval(keyword.value) for keyword in node.keywords)
        combined = _combine_evals([receiver, *arguments])
        method = _last_name(name).casefold()

        url = self._call_url(node, arguments)
        if self._is_graph_post(name, url):
            source_id = self._new_source("LiPDGraph", "table", node, True)
            evaluation = _Eval(
                deps=combined.deps,
                source_ids=frozenset({source_id}),
                family="response",
                kind="response",
                created_sources=frozenset({source_id}),
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        constructor = self._constructor_tool(name)
        if constructor is not None:
            tool, family, kind = constructor
            evaluation = self._new_source_eval(
                tool, family, kind, node, active=False
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if self._is_xarray_loader(name):
            evaluation = self._new_source_eval(
                "xarray",
                "xarray",
                "xarray_dataarray"
                if method == "open_dataarray"
                else "xarray_dataset",
                node,
                active=True,
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if method == "open_esm_datastore" and "intake" in name.casefold():
            evaluation = self._new_source_eval(
                "xarray", "xarray", "xarray_catalog", node, active=True
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if method == "to_dataset_dict" and receiver.source_ids:
            evaluation = _Eval(
                deps=combined.deps,
                source_ids=combined.source_ids,
                family="xarray",
                kind="xarray_collection",
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if self._is_pandas_reader(name):
            source_ids = combined.source_ids
            created_sources = frozenset()
            if not source_ids:
                source_id = self._new_source("pandas", "table", node, True)
                source_ids = frozenset({source_id})
                created_sources = source_ids
            evaluation = _Eval(
                deps=combined.deps,
                source_ids=source_ids,
                family="table",
                kind="table",
                created_sources=created_sources,
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if receiver.source_ids and method in _PYLEO_LOAD_METHODS:
            receiver = self._activate_receiver(
                receiver_node, receiver, arguments, method, node
            )
            combined = _combine_evals([receiver, *arguments])

        if method in {"append", "extend", "insert"}:
            evaluation = self._mutate_collection(
                receiver_node, receiver, arguments, node
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if receiver.source_ids and method in _PYLEO_PUBLICATION_METHODS:
            return _Eval(deps=combined.deps, family="citation", kind="citation")

        if receiver.source_ids and method in _PYLEO_DATA_METHODS:
            family = "table" if method != "get_datasets" else "collection"
            kind = "table" if family == "table" else "list"
            evaluation = _Eval(
                deps=combined.deps,
                source_ids=combined.source_ids,
                family=family,
                kind=kind,
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if receiver.source_ids and method == "iterrows":
            return _Eval(
                deps=combined.deps,
                source_ids=combined.source_ids,
                family="table",
                kind="table_iterator",
            )

        if receiver.family == "xarray":
            if method in _XARRAY_INTERMEDIATE_METHODS:
                kind = "xarray_intermediate"
            elif receiver.kind == "xarray_dataarray":
                kind = "xarray_dataarray"
            else:
                kind = "xarray_dataset"
            evaluation = _Eval(
                deps=combined.deps,
                source_ids=combined.source_ids,
                family="xarray",
                kind=kind,
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if receiver.family == "table":
            kind = "table" if method in _TABLE_METHODS else receiver.kind
            evaluation = _Eval(
                deps=combined.deps,
                source_ids=combined.source_ids,
                family="table",
                kind=kind,
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        if receiver.family in {"object", "pyleo", "domain"}:
            evaluation = _Eval(
                deps=combined.deps,
                source_ids=combined.source_ids,
                family=receiver.family,
                kind=receiver.kind,
            )
            self._record_sink(name, evaluation, node)
            return evaluation

        evaluation = _Eval(
            deps=combined.deps,
            source_ids=combined.source_ids,
            family=combined.family,
            kind=combined.kind,
        )
        self._record_sink(name, evaluation, node)
        return evaluation

    # ------------------------------------------------------------- results

    def _closure(self, value_ids: Iterable[int]) -> set[int]:
        """Returns the transitive dependency closure of graph values."""
        pending = list(value_ids)
        seen: set[int] = set()
        while pending:
            value_id = pending.pop()
            if value_id in seen or value_id not in self.values:
                continue
            seen.add(value_id)
            pending.extend(self.values[value_id].deps)
        return seen

    def _candidate_for_source(
        self,
        source: _Source,
        sink: _Sink,
        closure: set[int],
    ) -> _Value | None:
        """Selects the nearest analysis-boundary value for one source."""
        candidates = [
            self.values[value_id]
            for value_id in closure
            if source.source_id in self.values[value_id].source_ids
        ]
        if source.family == "object":
            candidates = [
                value
                for value in candidates
                if value.kind in {"pylipd_object", "pyleotups_object"}
            ]
        elif source.family == "table":
            candidates = [
                value
                for value in candidates
                if value.kind == "table"
            ]
        elif source.family == "xarray":
            candidates = [
                value
                for value in candidates
                if value.kind in {
                    "xarray_dataset",
                    "xarray_dataarray",
                }
            ]
            if sink.prefer_dataset:
                datasets = [
                    value
                    for value in candidates
                    if value.kind == "xarray_dataset"
                ]
                if datasets:
                    candidates = datasets
        if not candidates:
            return None
        return max(candidates, key=lambda value: value.order)

    def results(self) -> list[list[str]]:
        """Returns deterministic, deduplicated source/tool pairs."""
        pairs: dict[
            tuple[str, str],
            tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
        ] = {}
        for sink in self.sinks:
            closure = self._closure(sink.deps)
            for source_id in sorted(sink.source_ids):
                source = self.sources.get(source_id)
                if source is None or not source.active:
                    continue
                candidate = self._candidate_for_source(source, sink, closure)
                if candidate is None:
                    continue
                pair = (candidate.name, source.tool)
                if pair[0] is None:
                    continue
                position = (source.order, candidate.order)
                previous = pairs.get(pair)
                if previous is None or position < previous:
                    pairs[pair] = position

        ordered = sorted(
            pairs.items(),
            key=lambda item: (item[1][0], item[1][1], item[0][0], item[0][1]),
        )
        return [[variable, tool] for (variable, tool), _ in ordered]


def _parse_code_cells(notebook_path: str) -> list[tuple[int, str]]:
    """Reads code cells while preserving notebook order and skipping generated cells."""
    with open(notebook_path) as handle:
        notebook = nbformat.read(handle, as_version=4)
    cells = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or is_generated_cell(cell.source):
            continue
        source = strip_ipython_directives(cell.source)
        if source.strip():
            cells.append((index, source))
    return cells


def _parse_tree(source: str) -> ast.AST | None:
    """Parses source while skipping syntax errors and harmless warnings."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(source)
    except SyntaxError:
        return None


def _detect_datasets_from_code(code: str) -> list[list[str]]:
    """
    Detects analysis-used dataset sources from one Python source string.

    This private helper is useful for small internal checks. The public
    notebook API remains detect_datasets_in_notebook(), which accepts only a
    notebook path.

    Args:
        code: Python source containing one or more notebook cells

    Returns:
        deterministic [variable, tool] pairs
    """
    graph = _NotebookGraph()
    tree = _parse_tree(strip_ipython_directives(code))
    if tree is None:
        return []
    graph.process(tree, 0)
    return graph.results()


def detect_datasets_in_notebook(notebook_path: str) -> list[list[str]]:
    """
    Detects external dataset sources that reach scientific analysis in a notebook.

    Args:
        notebook_path: path to the target .ipynb file

    Returns:
        deterministic [variable, tool] pairs, ordered by source position
    """
    graph = _NotebookGraph()
    for cell_index, source in _parse_code_cells(notebook_path):
        tree = _parse_tree(source)
        if tree is None:
            continue
        graph.process(tree, cell_index)
    return graph.results()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python deterministic_dataset_detection.py <path_to_notebook.ipynb>")
        sys.exit(1)
    for variable, tool in detect_datasets_in_notebook(sys.argv[1]):
        print(f"{variable}\t{tool}")
