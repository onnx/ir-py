# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for extracting subgraphs from a graph."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Collection, Sequence
from typing import Union

import onnx_ir as ir

logger = logging.getLogger(__name__)


GraphLike = Union["ir.Graph", "ir.Function", "ir.GraphView"]


def _find_subgraph_bounded_by_values(
    graph: GraphLike, inputs: Collection[ir.Value], outputs: Collection[ir.Value]
) -> tuple[list[ir.Node], Collection[ir.Value]]:
    """Finds the subgraph bounded by the given inputs and outputs.

    Args:
        graph: The graph to search.
        inputs: The inputs to the subgraph.
        outputs: The outputs of the subgraph.

    Returns:
        A list of nodes in the subgraph and the initializers used.
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
        ValueError: If any of the input or output are not found in the graph.
    """
    if isinstance(graph_like, ir.Function):
        graph = graph_like.graph
    else:
        graph = graph_like
    values = ir.convenience.create_value_mapping(graph)
    is_graph_view = isinstance(graph_like, ir.GraphView)
    for val in itertools.chain(inputs, outputs):
        if isinstance(val, ir.Value):
            if not is_graph_view and val.graph is not graph:
                raise ValueError(
                    f"Value '{val}' does not belong to the given "
                    f"{graph_like.__class__.__name__} ({graph.name})."
                )
        else:
            if val not in values:
                raise ValueError(f"Value with name '{val}' not found in the graph.")

    inputs = [values[val] if isinstance(val, str) else val for val in inputs]
    outputs = [values[val] if isinstance(val, str) else val for val in outputs]
    extracted_nodes, initialized_values = _find_subgraph_bounded_by_values(
        graph_like, inputs, outputs
    )

    graph_view = ir.GraphView(
        inputs,
        outputs,
        nodes=extracted_nodes,
        initializers=tuple(initialized_values),
        doc_string=graph_like.doc_string,
        opset_imports=graph_like.opset_imports,
        name=graph_like.name,
        metadata_props=graph_like.metadata_props,
    )

    return graph_view.clone()
