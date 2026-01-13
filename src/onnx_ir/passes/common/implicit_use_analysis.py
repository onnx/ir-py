# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Identify implicit uses of values in ONNX sub-graphs."""

from __future__ import annotations

__all__ = [
    "ImplicitUseAnalysisPass",
]

import onnx_ir as ir


class ImplicitUseAnalysisPass(ir.passes.InPlacePass):
    """Find all values that are implicitly used by the sub-graphs.

    This pass will store in each sub-graph's ``.meta`` (not ``metadata_props``) a
    list of :class:`~onnx_ir.Value`s that are captured from outer scopes (i.e., not defined
    within the sub-graph itself). The list is stored under the key defined by
    :attr:`onnx_ir.passes.common.ImplicitUseAnalysisPassMETADATA_KEY`.
    """

    METADATA_KEY = "pkg.onnx_ir.ImplicitUseAnalysisPass.values"

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        graph_stack: list[ir.Graph] = []
        implicit_usages: dict[ir.Graph, list[ir.Value]] = {}
        for node in model.graph:
            _iterate_subgraphs(node, implicit_usages, graph_stack)

        for graph, used_values in implicit_usages.items():
            # Remove duplicates while preserving order
            seen = set()
            captured_values = []
            for val in used_values:
                if val not in seen:
                    seen.add(val)
                    captured_values.append(val)
            graph.meta[ImplicitUseAnalysisPass.METADATA_KEY] = captured_values
            modified = True
        return ir.passes.PassResult(model, modified=modified)


def _iterate_subgraphs(
    node: ir.Node,
    implicit_usages: dict[ir.Graph, list[ir.Value]],
    graph_stack: list[ir.Graph],
) -> None:
    """Perform a DFS to find all implicit usages in subgraphs."""

    def process_node(node: ir.Node, subgraph: ir.Graph):
        for inp in node.inputs:
            if inp is not None and inp.graph is not subgraph:
                # This is a closed variable, add to implicit usages of all graphs that enclose it
                for g in reversed(graph_stack):
                    if g is inp.graph:
                        break
                    implicit_usages[g].append(inp)

    for attr in node.attributes.values():
        if attr.type == ir.AttributeType.GRAPH:
            subgraph = attr.as_graph()
            graph_stack.append(subgraph)
            if subgraph not in implicit_usages:
                implicit_usages[subgraph] = []
            for node in subgraph:
                process_node(node, subgraph)
                _iterate_subgraphs(node, implicit_usages, graph_stack)
            graph_stack.pop()
        elif attr.type == ir.AttributeType.GRAPHS:
            for subgraph in attr.as_graphs():
                graph_stack.append(subgraph)
                if subgraph not in implicit_usages:
                    implicit_usages[subgraph] = []
                for node in subgraph:
                    process_node(node, subgraph)
                    _iterate_subgraphs(node, implicit_usages, graph_stack)
                graph_stack.pop()
