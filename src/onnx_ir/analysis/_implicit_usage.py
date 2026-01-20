# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Identify implicit uses of values in ONNX sub-graphs."""

from __future__ import annotations

__all__ = [
    "analyze_implicit_usage",
]

import onnx_ir as ir


def analyze_implicit_usage(graph: ir.Graph) -> dict[ir.Graph, set[ir.Value]]:
    """Analyze implicit usage of values in sub-graphs.

    This function returns a mapping from each sub-graph to a set of
    :class:`~onnx_ir.Value`s that are captured from outer scopes (i.e., not defined
    within the sub-graph itself).

    Args:
        graph: The graph to analyze.

    Returns:
        A dictionary mapping sub-graphs to sets of implicitly used values.
    """
    graph_stack: list[ir.Graph] = [graph]
    implicit_usages: dict[ir.Graph, set[ir.Value]] = {}
    for node in graph:
        _process_node(node, implicit_usages, graph_stack)
    return implicit_usages


def _collect_implicit_usages(
    node: ir.Node,
    subgraph: ir.Graph,
    graph_stack: list[ir.Graph],
    implicit_usages: dict[ir.Graph, set[ir.Value]],
) -> None:
    for inp in node.inputs:
        if inp is None or inp.graph is subgraph:
            continue
        # This is a closed variable, add to implicit usages of all graphs that enclose it
        for g in reversed(graph_stack):
            if g is inp.graph:
                break
            implicit_usages[g].add(inp)


def _process_node(
    node: ir.Node,
    implicit_usages: dict[ir.Graph, set[ir.Value]],
    graph_stack: list[ir.Graph],
) -> None:
    """Perform a DFS to find all implicit usages in subgraphs."""
    for attr in node.attributes.values():
        if attr.type == ir.AttributeType.GRAPH:
            subgraph = attr.as_graph()
            graph_stack.append(subgraph)
            if subgraph not in implicit_usages:
                implicit_usages[subgraph] = set()
            for node in subgraph:
                _collect_implicit_usages(node, subgraph, graph_stack, implicit_usages)
                _process_node(node, implicit_usages, graph_stack)
            graph_stack.pop()
        elif attr.type == ir.AttributeType.GRAPHS:
            for subgraph in attr.as_graphs():
                graph_stack.append(subgraph)
                if subgraph not in implicit_usages:
                    implicit_usages[subgraph] = set()
                for node in subgraph:
                    _collect_implicit_usages(node, subgraph, graph_stack, implicit_usages)
                    _process_node(node, implicit_usages, graph_stack)
                graph_stack.pop()
