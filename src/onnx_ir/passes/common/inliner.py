# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Implementation of an inliner for onnx_ir."""

from __future__ import annotations

import dataclasses
import graphlib
from collections.abc import Callable

__all__ = ["InlinePass", "InlinePassResult"]

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import onnx_ir as ir
from onnx_ir import _cloner
from onnx_ir.passes.common import unused_removal as _unused_removal

# A replacement for a node specifies a list of nodes that replaces the original node,
# and a list of values that replaces the original node's outputs.

NodeReplacement = tuple[Sequence[ir.Node], Sequence[ir.Value]]

# A call stack is a list of identifiers of call sites, where the first element is the
# outermost call site, and the last element is the innermost call site. This is used
# primarily for generating unique names for values in the inlined functions.
CallSiteId = str
CallStack = list[CallSiteId]


logger = logging.getLogger(__name__)


def _make_unique_name(name: str, callstack: CallStack, used_names: set[str]) -> str:  # pylint: disable=unused-argument
    """Generate a unique name from a name, calling-context, and set of used names.

    If there is a name clash, we add a numeric suffix to the name to make
    it unique. We use the same strategy to make node names unique.

    TODO: We can use the callstack in generating a name for a value X in a function
    that is inlined into a graph. This is not yet implemented. Using the full callstack
    leads to very long and hard to read names. Some investigation is needed to find
    a good naming strategy that will produce useful names for debugging.
    """
    candidate = name
    i = 1
    while candidate in used_names:
        i += 1
        candidate = f"{name}_{i}"
    used_names.add(candidate)
    return candidate


def _abbreviate(
    function_ids: Iterable[ir.OperatorIdentifier],
) -> dict[ir.OperatorIdentifier, str]:
    """Create a short unambiguous abbreviation for all function ids."""

    def id_abbreviation(id: ir.OperatorIdentifier) -> str:
        """Create a short unambiguous abbreviation for a function id."""
        domain, name, overload = id
        # Omit the domain, if it remains unambiguous after omitting it.
        if any(x[0] != domain and x[1] == name and x[2] == overload for x in function_ids):
            short_domain = domain + "_"
        else:
            short_domain = ""
        if overload != "":
            return short_domain + name + "_" + overload
        return short_domain + name

    return {id: id_abbreviation(id) for id in function_ids}


def _topological_sort_functions(model: ir.Model) -> None:
    """Sort functions so that callees come before callers.

    This ensures that when we inline function A which calls function B,
    function B has already been inlined into A's body.

    Raises:
        graphlib.CycleError: If there is a cyclic dependency between functions.
    """
    # Build dependency graph: function_id -> set of function_ids it calls
    dependencies: dict[ir.OperatorIdentifier, set[ir.OperatorIdentifier]] = {}

    for func_id, function in model.functions.items():
        for node in function.all_nodes():
            op_id = node.op_identifier()
            if op_id in model.functions:
                dependencies.setdefault(func_id, set()).add(op_id)

    sorter = graphlib.TopologicalSorter(dependencies)

    # Perform the sort to detect cycles.
    _ = tuple(sorter.static_order())
    return


@dataclasses.dataclass
class InlinePassResult(ir.passes.PassResult):
    id_count: dict[ir.OperatorIdentifier, int]


class InlinePass(ir.passes.InPlacePass):
    """Inline model local functions to the main graph and functions and remove unused functions.

    When a node calls a function defined in the model and when ``criteria`` is True or
    unspecified, the function body is inlined into the graph in place of the call node.

    .. warning::
        The local functions must not contain cycles (i.e., recursive calls). If there are cycles,
        the pass will not terminate.

    .. versionadded:: 0.1.16
        The ``criteria`` parameter.

    Attributes:
        criteria: Optional function that takes a node and returns True if the node
            should be inlined. If None, all function calls are inlined.
    """

    def __init__(self, criteria: Callable[[ir.Node], bool] | None = None) -> None:
        super().__init__()
        self.criteria = criteria
        self._function_removal_pass = _unused_removal.RemoveUnusedFunctionsPass()

        # Internal states
        self._functions: dict[ir.OperatorIdentifier, ir.Function] = {}
        self._function_id_abbreviations: dict[ir.OperatorIdentifier, str] = {}
        self._opset_imports: dict[str, int] = {}
        self._used_value_names: set[str] = set()
        self._used_node_names: set[str] = set()
        self._node_context: dict[ir.Node, CallStack] = {}

    def _reset(self, model: ir.Model) -> None:
        self._functions = model.functions
        self._function_id_abbreviations = _abbreviate(self._functions.keys())
        self._opset_imports = model.opset_imports
        self._used_value_names = set()
        self._used_node_names = set()
        self._node_context = {}

    def requires(self, model: ir.Model) -> None:
        self._reset(model)
        # No cyclic dependencies allowed in functions
        try:
            _topological_sort_functions(model)
        except graphlib.CycleError as e:
            raise ir.passes.PreconditionError(
                "Cyclic dependency detected between functions in model"
            ) from e

    def call(self, model: ir.Model) -> InlinePassResult:
        self._reset(model)
        id_count: dict[ir.OperatorIdentifier, int] = {}

        # Inline calls in the main graph
        main_id_count, total_inlined = self._inline_calls_in(model.graph)
        for k, v in main_id_count.items():
            id_count[k] = id_count.get(k, 0) + v

        self._function_removal_pass(model)

        # Inline local functions left in the model because some functions may need to be
        # preserved due to the criteria. These functions may themselves contain calls to other
        # functions that can be inlined.
        for function in model.functions.values():
            inner_id_count, inlined = self._inline_calls_in(function.graph)
            total_inlined += inlined
            for k, v in inner_id_count.items():
                id_count[k] = id_count.get(k, 0) + v

        return InlinePassResult(model, modified=bool(total_inlined), id_count=id_count)

    def _instantiate_call(self, node: ir.Node, call_site_id: CallSiteId) -> NodeReplacement:
        id = node.op_identifier()
        function = self._functions[id]

        # check opset compatibility and update the opset imports
        for key, value in function.opset_imports.items():
            if key not in self._opset_imports:
                self._opset_imports[key] = value
            elif self._opset_imports[key] != value:
                raise ValueError(
                    f"Opset mismatch: {key} {self._opset_imports[key]} != {value}"
                )

        # Identify substitutions for both inputs and attributes of the function:
        attributes: Mapping[str, ir.Attr] = node.attributes
        default_attr_values = {
            attr.name: attr
            for attr in function.attributes.values()
            if attr.name not in attributes and attr.value is not None
        }
        if default_attr_values:
            attributes = {**attributes, **default_attr_values}
        if any(
            attr.type in {ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS}
            for attr in attributes.values()
        ):
            raise ValueError(
                "Inliner does not support graph attribute parameters to functions"
            )

        if len(node.inputs) > len(function.inputs):
            raise ValueError(f"Input mismatch: {len(node.inputs)} > {len(function.inputs)}")
        value_map = {}
        for i, input in enumerate(node.inputs):
            value_map[function.inputs[i]] = input
        for i in range(len(node.inputs), len(function.inputs)):
            value_map[function.inputs[i]] = None

        # Identify call-stack for node, used to generate unique names.
        call_stack = self._node_context.get(node, [])
        new_call_stack = [*call_stack, call_site_id]

        def rename(node: ir.Node) -> None:
            """Rename node/values in inlined node to ensure uniqueness in the inlined context."""
            node_name = node.name or "node"
            node.name = _make_unique_name(node_name, new_call_stack, self._used_node_names)
            for output in node.outputs:
                if output is not None:
                    output_name = output.name or "val"
                    output.name = _make_unique_name(
                        output_name, new_call_stack, self._used_value_names
                    )
            # Update context in case the new node is itself a call node that will be inlined.
            self._node_context[node] = new_call_stack

        cloner = _cloner.Cloner(
            attr_map=attributes,
            value_map=value_map,
            metadata_props=node.metadata_props,
            post_process=rename,
            resolve_ref_attrs=True,
        )

        # iterate over the nodes in the function, creating a copy of each node
        # and replacing inputs with the corresponding values in the value map.
        # Update the value map with the new values.

        nodes = [cloner.clone_node(node) for node in function]
        output_values = [value_map[output] for output in function.outputs]
        return nodes, output_values  # type: ignore[return-value]

    def _inline_calls_in(
        self, graph: ir.Graph
    ) -> tuple[dict[ir.OperatorIdentifier, int], int]:
        """Inline function calls in a graph.

        Returns:
            A tuple of (id_count, inlined_count) where:
            - id_count: A dict mapping function ids to the number of calls in the graph
              (used for naming disambiguation).
            - inlined_count: The number of nodes that were actually inlined.
        """
        for input in graph.inputs:
            if input.name is not None:
                self._used_value_names.add(input.name)
        for initializer in graph.initializers:
            self._used_value_names.add(initializer)

        # Pre-processing:
        # * Count the number of times each function is called in the graph.
        #   This is used for disambiguating names of values in the inlined functions.
        # * And identify names of values that are used in the graph.
        id_count: dict[ir.OperatorIdentifier, int] = defaultdict(int)
        for node in graph:
            if node.name:
                self._used_node_names.add(node.name)
            op_id = node.op_identifier()
            if op_id in self._functions:
                id_count[op_id] += 1
            for output in node.outputs:
                if output.name is not None:
                    self._used_value_names.add(output.name)

        next_id: dict[ir.OperatorIdentifier, int] = defaultdict(int)
        inlined_count = 0
        for node in graph:
            op_id = node.op_identifier()
            if op_id in self._functions:
                if self.criteria is not None and not self.criteria(node):
                    logger.debug("Skipping inlining for node '%r' due to criteria", node)
                    continue
                # If there are multiple calls to same function, we use a prefix to disambiguate
                # the different call-sites:
                if id_count[op_id] > 1:
                    call_site_prefix = f"_{next_id[op_id]}"
                    next_id[op_id] += 1
                else:
                    call_site_prefix = ""
                call_site = node.name or (
                    self._function_id_abbreviations[op_id] + call_site_prefix
                )
                nodes, values = self._instantiate_call(node, call_site)
                ir.convenience.replace_nodes_and_values(
                    graph,
                    insertion_point=node,
                    old_nodes=[node],
                    new_nodes=nodes,
                    old_values=node.outputs,
                    new_values=values,
                )
                inlined_count += 1
            else:
                for attr in node.attributes.values():
                    if attr.type == ir.AttributeType.GRAPH:
                        _, sub_inlined = self._inline_calls_in(attr.as_graph())
                        inlined_count += sub_inlined
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for g in attr.as_graphs():
                            _, sub_inlined = self._inline_calls_in(g)
                            inlined_count += sub_inlined
        return id_count, inlined_count
