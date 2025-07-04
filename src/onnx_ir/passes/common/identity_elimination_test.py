# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the identity elimination pass."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common import identity_elimination


class TestIdentityEliminationPass(unittest.TestCase):
    """Test cases for IdentityEliminationPass."""

    def test_case_1_eliminate_identity_not_graph_output(self):
        """Test Case 1: y = Identity(x) where y is not a graph output."""
        # Create a simple model: input -> Identity -> Add -> output
        input_value = ir.Input("input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create Identity node
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = input_value.shape
        identity_node.outputs[0].type = input_value.type
        
        # Create Add node that uses the Identity output
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], input_value])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type
        
        graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],  # Identity output is NOT a graph output
            nodes=[identity_node, add_node],
            name="test_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass was applied
        self.assertTrue(result.modified)
        
        # Verify Identity node was removed
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)  # Only Add node should remain
        self.assertEqual(remaining_nodes[0].op_type, "Add")
        
        # Verify Add node now uses input_value directly
        add_node_after = remaining_nodes[0]
        self.assertIs(add_node_after.inputs[0], input_value)
        self.assertIs(add_node_after.inputs[1], input_value)

    def test_case_2_eliminate_identity_with_output_renaming(self):
        """Test Case 2: y = Identity(x) where y is graph output but x is not graph input."""
        # Create intermediate value (not a graph input)
        intermediate_value = ir.Value("intermediate", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create a Constant node that produces the intermediate value
        const_node = ir.Node("", "Constant", inputs=[])
        const_node.outputs[0].name = "intermediate"
        const_node.outputs[0].shape = intermediate_value.shape
        const_node.outputs[0].type = intermediate_value.type
        intermediate_value = const_node.outputs[0]  # Use the actual output
        
        # Create Identity node
        identity_node = ir.Node("", "Identity", inputs=[intermediate_value])
        identity_node.outputs[0].name = "final_output"
        identity_node.outputs[0].shape = intermediate_value.shape
        identity_node.outputs[0].type = intermediate_value.type
        
        graph = ir.Graph(
            inputs=[],
            outputs=[identity_node.outputs[0]],  # Identity output IS a graph output
            nodes=[const_node, identity_node],
            name="test_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Store original output name
        original_output_name = identity_node.outputs[0].name
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass was applied
        self.assertTrue(result.modified)
        
        # Verify Identity node was removed
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)  # Only Constant node should remain
        self.assertEqual(remaining_nodes[0].op_type, "Constant")
        
        # Verify the intermediate value was renamed and is now the graph output
        graph_output = result.model.graph.outputs[0]
        self.assertEqual(graph_output.name, original_output_name)
        self.assertIs(graph_output, intermediate_value)

    def test_case_3_keep_identity_when_both_input_and_output_are_graph_boundaries(self):
        """Test Case 3: y = Identity(x) where y is graph output AND x is graph input."""
        # Create graph input
        input_value = ir.Input("graph_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create Identity node
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "graph_output"
        identity_node.outputs[0].shape = input_value.shape
        identity_node.outputs[0].type = input_value.type
        
        graph = ir.Graph(
            inputs=[input_value],  # x IS a graph input
            outputs=[identity_node.outputs[0]],  # y IS a graph output
            nodes=[identity_node],
            name="test_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass did NOT modify the model
        self.assertFalse(result.modified)
        
        # Verify Identity node was kept
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)
        self.assertEqual(remaining_nodes[0].op_type, "Identity")
        
        # Verify structure is unchanged
        self.assertEqual(len(result.model.graph.inputs), 1)
        self.assertEqual(len(result.model.graph.outputs), 1)
        self.assertIs(result.model.graph.inputs[0], input_value)
        self.assertIs(result.model.graph.outputs[0], identity_node.outputs[0])

    def test_multiple_identity_nodes(self):
        """Test elimination of multiple Identity nodes in different scenarios."""
        # Create graph input
        input_value = ir.Input("input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create first Identity node (Case 3: should be kept)
        identity1 = ir.Node("", "Identity", inputs=[input_value])
        identity1.outputs[0].name = "identity1_out"
        identity1.outputs[0].shape = input_value.shape
        identity1.outputs[0].type = input_value.type
        
        # Create second Identity node (Case 1: should be eliminated)
        identity2 = ir.Node("", "Identity", inputs=[identity1.outputs[0]])
        identity2.outputs[0].name = "identity2_out"
        identity2.outputs[0].shape = input_value.shape
        identity2.outputs[0].type = input_value.type
        
        # Create a final Add node
        add_node = ir.Node("", "Add", inputs=[identity2.outputs[0], input_value])
        add_node.outputs[0].name = "final_output"
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type
        
        graph = ir.Graph(
            inputs=[input_value],
            outputs=[identity1.outputs[0], add_node.outputs[0]],  # identity1 is graph output
            nodes=[identity1, identity2, add_node],
            name="test_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass was applied
        self.assertTrue(result.modified)
        
        # Verify only identity2 was removed, identity1 was kept
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 2)  # identity1 and add_node
        
        remaining_ops = [node.op_type for node in remaining_nodes]
        self.assertIn("Identity", remaining_ops)  # identity1 kept
        self.assertIn("Add", remaining_ops)
        
        # Verify add_node now uses identity1 output directly (not identity2)
        add_node_after = next(node for node in remaining_nodes if node.op_type == "Add")
        identity1_after = next(node for node in remaining_nodes if node.op_type == "Identity")
        self.assertIs(add_node_after.inputs[0], identity1_after.outputs[0])

    def test_invalid_identity_node_skipped(self):
        """Test that invalid Identity nodes are skipped."""
        input_value = ir.Input("input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create an Identity node with no inputs (invalid)
        invalid_identity = ir.Node("", "Identity", inputs=[])
        invalid_identity.outputs[0].name = "invalid_out"
        
        graph = ir.Graph(
            inputs=[input_value],
            outputs=[invalid_identity.outputs[0]],
            nodes=[invalid_identity],
            name="test_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass did not modify the model (invalid node skipped)
        self.assertFalse(result.modified)
        
        # Verify invalid Identity node was kept
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)
        self.assertEqual(remaining_nodes[0].op_type, "Identity")

    def test_identity_with_none_input_skipped(self):
        """Test that Identity nodes with None input are skipped."""
        # Create Identity node with None input
        identity_node = ir.Node("", "Identity", inputs=[None])
        identity_node.outputs[0].name = "identity_out"
        
        graph = ir.Graph(
            inputs=[],
            outputs=[identity_node.outputs[0]],
            nodes=[identity_node],
            name="test_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass did not modify the model
        self.assertFalse(result.modified)
        
        # Verify Identity node was kept
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)
        self.assertEqual(remaining_nodes[0].op_type, "Identity")

    def test_no_identity_nodes(self):
        """Test pass on a graph with no Identity nodes."""
        input_value = ir.Input("input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create Add node only
        add_node = ir.Node("", "Add", inputs=[input_value, input_value])
        add_node.outputs[0].name = "output"
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type
        
        graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="test_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass did not modify the model
        self.assertFalse(result.modified)
        
        # Verify structure is unchanged
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)
        self.assertEqual(remaining_nodes[0].op_type, "Add")

    def test_empty_graph(self):
        """Test pass on an empty graph."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            name="empty_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass did not modify the model
        self.assertFalse(result.modified)
        
        # Verify structure is unchanged
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 0)

    def test_chain_of_identities(self):
        """Test elimination of a chain of Identity nodes."""
        input_value = ir.Input("input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create a chain: input -> Identity1 -> Identity2 -> Identity3 -> output
        identity1 = ir.Node("", "Identity", inputs=[input_value])
        identity1.outputs[0].name = "id1_out"
        identity1.outputs[0].shape = input_value.shape
        identity1.outputs[0].type = input_value.type
        
        identity2 = ir.Node("", "Identity", inputs=[identity1.outputs[0]])
        identity2.outputs[0].name = "id2_out"
        identity2.outputs[0].shape = input_value.shape
        identity2.outputs[0].type = input_value.type
        
        identity3 = ir.Node("", "Identity", inputs=[identity2.outputs[0]])
        identity3.outputs[0].name = "final_output"
        identity3.outputs[0].shape = input_value.shape
        identity3.outputs[0].type = input_value.type
        
        graph = ir.Graph(
            inputs=[input_value],
            outputs=[identity3.outputs[0]],  # Only the last identity is graph output
            nodes=[identity1, identity2, identity3],
            name="test_graph"
        )
        
        model = ir.Model(graph, ir_version=10)
        
        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)
        
        # Verify the pass was applied
        self.assertTrue(result.modified)
        
        # Should only have one identity left (the last one can't be eliminated 
        # because it's a graph output and input is graph input)
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)
        self.assertEqual(remaining_nodes[0].op_type, "Identity")
        
        # The remaining identity should directly use the graph input
        remaining_identity = remaining_nodes[0]
        self.assertIs(remaining_identity.inputs[0], input_value)
        
        # The output should have the final name
        self.assertEqual(remaining_identity.outputs[0].name, "final_output")


if __name__ == "__main__":
    unittest.main()