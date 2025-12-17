# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Output fix pass for adding Identity nodes.

- Graph inputs are directly used as outputs (without any intermediate nodes).
- A value is used multiple times as a graph output (ensuring each output is unique).

This ensures compliance with the ONNX specification for valid output configurations.
"""

from __future__ import annotations

__all__ = [
    "OutputFixPass",
]

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


class OutputFixPass(ir.passes.InPlacePass):
    """Pass for adding Identity nodes to fix invalid output configurations.

    This pass adds Identity nodes according to the following rules:

    - If a graph input is directly used as a graph output (without any intermediate nodes),
      insert an Identity node between them. The ONNX specification does not allow a graph
      input to be directly used as a graph output without any processing nodes in between.
    - If a value is used multiple times as graph outputs, insert Identity nodes for each
      duplicate usage (keeping the first usage unchanged). This ensures each output value
      is unique, as required by the ONNX specification.

    This pass processes both the main graph and all subgraphs (e.g., in control flow operators).

    Example transformations:
        Direct input-to-output:
            Before: input -> (direct connection) -> output
            After:  input -> Identity -> output

        Duplicate outputs:
            Before: value -> [output1, output2]
            After:  value -> output1, value -> Identity -> output2
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Main entry point for the output fix pass."""
        modified = False

        # Process the main graph
        if _alias_multi_used_outputs(model.graph):
            modified = True
        if _alias_direct_outputs(model.graph):
            modified = True

        # Process functions
        for function in model.functions.values():
            if _alias_multi_used_outputs(function):
                modified = True
            if _alias_direct_outputs(function):
                modified = True

        return ir.passes.PassResult(model, modified=modified)


def _alias_multi_used_outputs(graph_like: ir.Graph | ir.Function) -> bool:
    """Insert Identity nodes for values that appear in the graph output list multiple times."""
    modified = False

    for graph in (graph_like, *graph_like.subgraphs()):
        # Count usage of each output
        seen: set[ir.Value] = set()

        # Add Identity nodes for outputs used multiple times
        for i, output in enumerate(graph.outputs):
            if output not in seen:
                # Skip the first occurrence
                seen.add(output)
                continue

            # Create an Identity node
            identity_node = ir.node("Identity", inputs=[output])
            identity_output = identity_node.outputs[0]

            # Copy metadata from the original output
            # TODO: Use a better unique naming strategy if needed
            identity_output.name = f"{output.name}_alias_{i}"
            identity_output.shape = output.shape
            identity_output.type = output.type
            identity_output.metadata_props.update(output.metadata_props)
            identity_output.doc_string = output.doc_string

            # Add the node to the graph
            graph.append(identity_node)
            graph.outputs[i] = identity_output
            logger.debug(
                "Added Identity node for graph output '%s' used multiple times", output
            )
            modified = True
    return modified


def _alias_direct_outputs(graph_like: ir.Graph | ir.Function) -> bool:
    """Insert Identity nodes for graph inputs used directly as outputs."""
    modified = False

    for graph in (graph_like, *graph_like.subgraphs()):
        # Check each output to see if it's directly a graph input
        outputs_to_fix: list[tuple[ir.Value, int]] = []
        for i, output in enumerate(graph.outputs):
            if output.is_graph_input():
                outputs_to_fix.append((output, i))

        # Add Identity nodes for each output that needs fixing
        for output, index in outputs_to_fix:
            # Create an Identity node
            identity_node = ir.node("Identity", inputs=[output])
            identity_output = identity_node.outputs[0]

            # Copy metadata from the original output
            # Preserve the original output name
            identity_output.name = output.name
            identity_output.shape = output.shape
            identity_output.type = output.type
            identity_output.metadata_props.update(output.metadata_props)
            identity_output.doc_string = output.doc_string

            # Create a new name for the old output
            # TODO: Use a better unique naming strategy if needed
            output.name = f"{output.name}_orig"

            # Add the node to the graph
            graph.append(identity_node)
            graph.outputs[index] = identity_output

            logger.debug("Added Identity node for graph input '%s' used as output", output)
            modified = True

    return modified
