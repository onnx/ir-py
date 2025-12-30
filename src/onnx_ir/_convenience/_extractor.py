# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for extracting subgraphs from a graph."""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from typing import Union

import onnx_ir as ir

logger = logging.getLogger(__name__)


GraphLike = Union[ir.Graph, ir.Function, ir.GraphView]


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
    node_index = {node: idx for idx, node in enumerate(graph)}
    all_nodes = []
    node_stack: list[ir.Node] = [
        producer for value in outputs if (producer := value.producer()) is not None
    ]
    visited_nodes: set[ir.Node] = set()
    if isinstance(graph, ir.Function):
        initialized_values: set[ir.Value] = set()
    else:
        initialized_values = {val for val in inputs if val.is_initializer()}
    while node_stack:
        node = node_stack.pop()
        if node in visited_nodes:
            continue
        if not isinstance(graph, ir.Function):
            # Record the initializer
            for input in node.inputs:
                if input is not None and input.is_initializer():
                    initialized_values.add(input)
        visited_nodes.add(node)
        all_nodes.append(node)
        for predecessor in node.predecessors():
            if predecessor not in visited_nodes:
                node_stack.append(predecessor)
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

    Args:
        graph_like: The graph-like object to extract from.
        inputs: The inputs to the subgraph. Can be Value objects or their names.
        outputs: The outputs of the subgraph. Can be Value objects or their names.

    Returns:
        The extracted subgraph as a new :class:`~onnx_ir.Graph` object.
    """
    values = ir.convenience.create_value_mapping(graph_like)
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
