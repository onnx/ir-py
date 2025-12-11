# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Identity fix pass for adding Identity nodes when graph inputs are directly used as outputs."""

from __future__ import annotations

__all__ = [
    "IdentityFixPass",
]

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


class IdentityFixPass(ir.passes.InPlacePass):
    """Pass for adding Identity nodes when graph inputs are directly used as outputs.

    This pass adds Identity nodes according to the following rule:

    If a graph input is directly used as a graph output (without any intermediate nodes),
    insert an Identity node between them. This turns an invalid ONNX graph into a valid one.

    Example transformation:
        Before: input -> (direct connection) -> output
        After:  input -> Identity -> output

    This is required because ONNX specification does not allow a graph input to be
    directly used as a graph output without any processing nodes in between.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Main entry point for the identity fix pass."""
        modified = False

        # Process the main graph
        if self._process_graph(model.graph):
            modified = True

        # Process functions
        for function in model.functions.values():
            if self._process_graph(function):
                modified = True

        if modified:
            logger.info("Identity fix pass modified the model")

        return ir.passes.PassResult(model, modified=modified)

    def _process_graph(self, graph_like: ir.Graph | ir.Function) -> bool:
        """Process a single graph or function, returning True if modified."""
        modified = False

        # Check each output to see if it's directly a graph input
        outputs_to_fix = []
        for output in graph_like.outputs:
            if output.is_graph_input():
                outputs_to_fix.append(output)

        # Add Identity nodes for each output that needs fixing
        for output in outputs_to_fix:
            # Create an Identity node
            identity_node = ir.Node("", "Identity", inputs=[output])
            identity_output = identity_node.outputs[0]
            
            # Copy metadata from the original output
            identity_output.name = output.name
            identity_output.shape = output.shape
            identity_output.type = output.type
            if output.metadata_props:
                identity_output.metadata_props.update(output.metadata_props)
            identity_output.doc_string = output.doc_string
            
            # Add the node to the graph
            graph_like.append(identity_node)
            
            # Replace the output with the Identity node's output
            # Find the index of the output in the graph outputs
            output_index = graph_like.outputs.index(output)
            graph_like.outputs[output_index] = identity_output
            
            logger.debug(
                "Added Identity node for graph input '%s' used as output", output.name
            )
            modified = True

        # Process subgraphs in nodes
        for node in graph_like:
            for attr in node.attributes.values():
                if isinstance(attr, ir.Attr):
                    if attr.type == ir.AttributeType.GRAPH and attr.value is not None:
                        if self._process_graph(attr.value):
                            modified = True
                    elif attr.type == ir.AttributeType.GRAPHS and attr.value is not None:
                        for subgraph in attr.value:
                            if subgraph is not None and self._process_graph(subgraph):
                                modified = True

        return modified
