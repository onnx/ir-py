# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for extracting subgraphs from a graph."""

from __future__ import annotations

import itertools
from collections.abc import Collection, Sequence
from typing import Union

import onnx_ir as ir

GraphLike = Union["ir.Graph", "ir.Function", "ir.GraphView"]


def _collect_all_external_values(parent_graph: ir.Graph, graph: ir.Graph) -> set[ir.Value]:
    """Collects all values in the given graph-like object.

    Args:
        parent_graph: The parent graph to which collected values must belong.
        graph: The graph-like object to collect values from.

    Returns:
        A set of :class:`~onnx_ir.Value` objects belonging to ``parent_graph``.
    """
    values: set[ir.Value] = set()
    for node in ir.traversal.RecursiveGraphIterator(graph):
        for val in node.inputs:
            if val is None:
                continue
            if val.graph is parent_graph:
                values.add(val)
    return values


def _find_subgraph_bounded_by_values(
    graph: GraphLike,
    inputs: Collection[ir.Value],
    outputs: Collection[ir.Value],
    parent_graph: ir.Graph,
) -> tuple[list[ir.Node], Collection[ir.Value]]:
    """Finds the subgraph bounded by the given inputs and outputs.

    Args:
        graph: The graph to search.
        inputs: The inputs to the subgraph.
        outputs: The outputs of the subgraph.
        parent_graph: The parent graph of the subgraph.

    Returns:
        A list of nodes in the subgraph and the initializers used.

    Raises:
        ValueError: If the subgraph is not properly bounded by the given inputs and outputs.
    """
    if isinstance(graph, ir.Function):
        initialized_values: set[ir.Value] = set()
    else:
        initialized_values = {val for val in inputs if val.is_initializer()}
    node_index = {node: idx for idx, node in enumerate(graph)}
    all_nodes = []
    value_stack: list[ir.Value] = [*outputs]
    visited_nodes: set[ir.Node] = set()
    visited_values: set[ir.Value] = set(inputs)

    while value_stack:
        value = value_stack.pop()
        if value in visited_values:
            continue
        if value.is_initializer():
            # Record the initializer
            initialized_values.add(value)

        visited_values.add(value)

        if (node := value.producer()) is not None:
            if node not in visited_nodes:
                visited_nodes.add(node)
                all_nodes.append(node)
                for input in node.inputs:
                    if input not in visited_values and input is not None:
                        value_stack.append(input)
                for attr in node.attributes.values():
                    if attr.type == ir.AttributeType.GRAPH:
                        values = _collect_all_external_values(parent_graph, attr.as_graph())
                        for val in values:
                            if val not in visited_values:
                                value_stack.append(val)
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for g in attr.as_graphs():
                            values = _collect_all_external_values(parent_graph, g)
                            for val in values:
                                if val not in visited_values:
                                    value_stack.append(val)

    # Validate that the subgraph is properly bounded
    # Collect all values at the input frontier (used by subgraph but not produced by it)
    # The frontier can only contain graph inputs or initializers (values with no producer)
    input_frontier: set[ir.Value] = set()
    for node in visited_nodes:
        for input_val in node.inputs:
            if input_val is None:
                continue
            # If this value is not produced by any node in the subgraph
            producer = input_val.producer()
            if producer is None or producer not in visited_nodes:
                input_frontier.add(input_val)

    # Check for graph inputs that weren't specified in the inputs parameter
    # (initializers are allowed, but unspecified graph inputs mean the subgraph is unbounded)
    unspecified_graph_inputs: list[ir.Value] = []
    inputs_set = set(inputs)
    for val in sorted(input_frontier, key=lambda v: v.name or ""):
        if val not in inputs_set and not val.is_initializer():
            unspecified_graph_inputs.append(val)

    if unspecified_graph_inputs:
        value_names = [val.name or "<None>" for val in unspecified_graph_inputs]
        raise ValueError(
            f"The subgraph is not properly bounded by the specified inputs and outputs. "
            f"The following graph inputs are required but not provided: {', '.join(value_names)}"
        )

    # Preserve the original order
    all_nodes.sort(key=lambda n: node_index[n])
    return all_nodes, initialized_values


def extract(
    graph_like: GraphLike,
    /,
    inputs: Sequence[ir.Value | str],
    outputs: Sequence[ir.Value | str],
) -> ir.Graph:
    """Extracts a subgraph from the given graph-like object.

    .. versionadded:: 0.1.14

    Args:
        graph_like: The graph-like object to extract from.
        inputs: The inputs to the subgraph. Can be Value objects or their names.
        outputs: The outputs of the subgraph. Can be Value objects or their names.

    Returns:
        The extracted subgraph as a new :class:`~onnx_ir.Graph` object.

    Raises:
        ValueError: If any of the inputs or outputs are not found in the graph.
        ValueError: If the subgraph is not properly bounded by the given inputs and outputs.
    """
    if isinstance(graph_like, ir.Function):
        graph: ir.Graph | ir.GraphView = graph_like.graph
    else:
        graph = graph_like
    values = ir.convenience.create_value_mapping(graph, include_subgraphs=False)
    is_graph_view = isinstance(graph_like, ir.GraphView)
    for val in itertools.chain(inputs, outputs):
        if isinstance(val, ir.Value):
            if not is_graph_view and val.graph is not graph:
                graph_name = graph.name if graph.name is not None else "unnamed graph"
                raise ValueError(
                    f"Value '{val}' does not belong to the given "
                    f"{graph_like.__class__.__name__} ({graph_name})."
                )
        else:
            if val not in values:
                raise ValueError(f"Value with name '{val}' not found in the graph.")

    input_vals = [values[val] if isinstance(val, str) else val for val in inputs]
    output_vals = [values[val] if isinstance(val, str) else val for val in outputs]
    # Find the owning graph of the outputs to set as the parent graph
    if not output_vals:
        raise ValueError("At least one output must be provided to extract a subgraph.")
    parent_graph = output_vals[0].graph
    assert parent_graph is not None
    extracted_nodes, initialized_values = _find_subgraph_bounded_by_values(
        graph_like, input_vals, output_vals, parent_graph=parent_graph
    )

    graph_view = ir.GraphView(
        input_vals,
        output_vals,
        nodes=extracted_nodes,
        initializers=tuple(initialized_values),
        doc_string=graph_like.doc_string,
        opset_imports=graph_like.opset_imports,
        name=graph_like.name,
        metadata_props=graph_like.metadata_props,
    )

    return graph_view.clone()
