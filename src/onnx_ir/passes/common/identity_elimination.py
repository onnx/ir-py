# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Identity elimination pass for removing redundant Identity nodes."""

from __future__ import annotations

__all__ = [
    "IdentityEliminationPass",
]

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


class IdentityEliminationPass(ir.passes.InPlacePass):
    """Pass for eliminating redundant Identity nodes.
    
    This pass removes Identity nodes according to the following rules:
    1. For any node of the form `y = Identity(x)`, where `y` is not an output 
       of any graph, replace all uses of `y` with a use of `x`, and remove the node.
    2. If `y` is an output of a graph, and `x` is not an input of any graph, 
       we can still do the elimination, but the value `x` should be renamed to be `y`.
    3. If `y` is a graph-output and `x` is a graph-input, we cannot eliminate 
       the node. It should be retained.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Main entry point for the identity elimination pass."""
        modified = False
        modified |= self._eliminate_identities_in_graph(model.graph)
        
        # Process subgraphs in functions
        for function in model.functions.values():
            modified |= self._eliminate_identities_in_function(function)
            
        if modified:
            logger.info("Identity elimination pass modified the model")
        
        return ir.passes.PassResult(model, modified=modified)
    
    def _eliminate_identities_in_graph(self, graph: ir.Graph) -> bool:
        """Eliminate identity nodes in a graph."""
        return self._eliminate_identities_in_graph_like(graph)
    
    def _eliminate_identities_in_function(self, function: ir.Function) -> bool:
        """Eliminate identity nodes in a function."""
        return self._eliminate_identities_in_graph_like(function)
    
    def _eliminate_identities_in_graph_like(self, graph_like: ir.Graph | ir.Function) -> bool:
        """Eliminate identity nodes in a graph-like object (Graph or Function)."""
        modified = False
        
        # Collect identity nodes to process (to avoid modifying while iterating)
        identity_nodes = []
        for node in graph_like:
            if node.op_type == "Identity":
                identity_nodes.append(node)
        
        for node in identity_nodes:
            if self._try_eliminate_identity_node(graph_like, node):
                modified = True
        
        # Process subgraphs in node attributes
        for node in graph_like:
            for attr in node.attributes.values():
                if not isinstance(attr, ir.Attr):
                    continue
                if attr.type == ir.AttributeType.GRAPH:
                    modified |= self._eliminate_identities_in_graph_like(attr.as_graph())
                elif attr.type == ir.AttributeType.GRAPHS:
                    for subgraph in attr.as_graphs():
                        modified |= self._eliminate_identities_in_graph_like(subgraph)
        
        return modified
    
    def _try_eliminate_identity_node(self, graph_like: ir.Graph | ir.Function, node: ir.Node) -> bool:
        """Try to eliminate a single identity node. Returns True if modified."""
        if node.op_type != "Identity":
            return False
        
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            # Invalid Identity node, skip
            return False
        
        input_value = node.inputs[0]
        output_value = node.outputs[0]
        
        if input_value is None:
            # Cannot eliminate if input is None
            return False
        
        output_is_graph_output = output_value.is_graph_output()
        input_is_graph_input = input_value.is_graph_input()
        
        # Case 3: Both output is graph output and input is graph input - keep the node
        if output_is_graph_output and input_is_graph_input:
            return False
        
        # Case 1: Output is not a graph output - replace all uses and remove node
        if not output_is_graph_output:
            ir.convenience.replace_all_uses_with(output_value, input_value)
            graph_like.remove(node, safe=True)
            logger.debug("Eliminated identity node: %s", node)
            return True
        
        # Case 2: Output is graph output but input is not graph input
        # Rename input to output name and update graph outputs
        if output_is_graph_output and not input_is_graph_input:
            # Store the original output name
            original_output_name = output_value.name
            
            # Replace all uses of output with input
            ir.convenience.replace_all_uses_with(output_value, input_value)
            
            # Update the input value to have the output's name
            input_value.name = original_output_name
            
            # Update graph outputs to point to the input value
            for idx, graph_output in enumerate(graph_like.outputs):
                if graph_output is output_value:
                    graph_like.outputs[idx] = input_value
                    break
            
            # Remove the identity node
            graph_like.remove(node, safe=True)
            logger.debug("Eliminated identity node with output renaming: %s", node)
            return True
        
        return False