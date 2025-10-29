# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass for topologically sorting the graphs."""

from __future__ import annotations

__all__ = [
    "TopologicalSortPass",
]


import onnx_ir as ir


class TopologicalSortPass(ir.passes.InPlacePass):
    """Topologically sort graphs and functions in a model.

    The sort is stable, preserving the relative order of nodes that are not
    dependent on each other. Read more at :meth:`onnx_ir.Graph.sort`.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        original_nodes = list(model.graph)
        model.graph.sort()
        sorted_nodes = list(model.graph)
        for function in model.functions.values():
            original_nodes.extend(function)
            function.sort()
            sorted_nodes.extend(function)

        # Compare node orders to determine if any changes were made
        modified = False
        for node, new_node in zip(original_nodes, sorted_nodes):
            if node is not new_node:
                modified = True
                break
        return ir.passes.PassResult(model=model, modified=modified)
