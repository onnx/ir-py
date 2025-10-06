# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the identity elimination pass."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common import identity_elimination


class TestIdentityEliminationPass(unittest.TestCase):
    """Test cases for IdentityEliminationPass."""

    def test_eliminate_identity_not_graph_output(self):
        """Test: y = Identity(x) where y is not a graph output."""
        # Create a simple model: input -> Identity -> Add -> output
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

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
            name="test_graph",
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

    def test_eliminate_identity_with_output_renaming(self):
        """Test: y = Identity(x) where y is graph output but x is not graph input."""
        # Create intermediate value (not a graph input)
        intermediate_value = ir.Value(
            "intermediate", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

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
            name="test_graph",
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

    def test_keep_identity_when_both_input_and_output_are_graph_boundaries(self):
        """Test: y = Identity(x) where y is graph output AND x is graph input."""
        # Create graph input
        input_value = ir.val(
            "graph_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Identity node
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "graph_output"
        identity_node.outputs[0].shape = input_value.shape
        identity_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],  # x IS a graph input
            outputs=[identity_node.outputs[0]],  # y IS a graph output
            nodes=[identity_node],
            name="test_graph",
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
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

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
            name="test_graph",
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
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create an Identity node with no inputs (invalid)
        invalid_identity = ir.Node("", "Identity", inputs=[])
        invalid_identity.outputs[0].name = "invalid_out"

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[invalid_identity.outputs[0]],
            nodes=[invalid_identity],
            name="test_graph",
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
            name="test_graph",
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
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Add node only
        add_node = ir.Node("", "Add", inputs=[input_value, input_value])
        add_node.outputs[0].name = "output"
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="test_graph",
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
        graph = ir.Graph(inputs=[], outputs=[], nodes=[], name="empty_graph")

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
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

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
            name="test_graph",
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

    def test_non_standard_domain_identity_skipped(self):
        """Test that Identity nodes with non-standard domain are skipped."""
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Identity node with custom domain
        identity_node = ir.Node("custom.domain", "Identity", inputs=[input_value])
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
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass did not modify the model
        self.assertFalse(result.modified)

        # Verify Identity node with custom domain was not eliminated
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 2)

        identity_nodes = [n for n in remaining_nodes if n.op_type == "Identity"]
        self.assertEqual(len(identity_nodes), 1)
        self.assertEqual(identity_nodes[0].domain, "custom.domain")

    def test_non_identity_node_skipped(self):
        """Test that non-Identity nodes are skipped (covers line 55)."""
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Add node (not Identity)
        add_node = ir.Node("", "Add", inputs=[input_value, input_value])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass did not modify the model
        self.assertFalse(result.modified)

        # Verify Add node was not eliminated
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)
        self.assertEqual(remaining_nodes[0].op_type, "Add")

    def test_function_with_identity_elimination(self):
        """Test that Identity nodes in functions are processed."""
        # Create function with Identity node
        func_input = ir.val(
            "func_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Identity node in function
        identity_node = ir.Node("", "Identity", inputs=[func_input])
        identity_node.outputs[0].name = "func_identity_output"
        identity_node.outputs[0].shape = func_input.shape
        identity_node.outputs[0].type = func_input.type

        # Create Add node that uses the Identity output
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], func_input])
        add_node.outputs[0].name = "func_add_output"
        add_node.outputs[0].shape = func_input.shape
        add_node.outputs[0].type = func_input.type

        # Create function graph
        func_graph = ir.Graph(
            inputs=[func_input],
            outputs=[add_node.outputs[0]],  # Identity output is NOT a function output
            nodes=[identity_node, add_node],
            name="test_function_graph",
        )

        function = ir.Function(
            domain="test_domain",
            name="test_function",
            graph=func_graph,
            attributes=[],
        )

        # Create a main graph
        main_input = ir.val(
            "main_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Call the function
        call_node = ir.Node("test_domain", "test_function", inputs=[main_input])
        call_node.outputs[0].name = "main_output"
        call_node.outputs[0].shape = main_input.shape
        call_node.outputs[0].type = main_input.type

        main_graph = ir.Graph(
            inputs=[main_input],
            outputs=[call_node.outputs[0]],
            nodes=[call_node],
            name="main_graph",
        )

        model = ir.Model(main_graph, ir_version=10, functions=[function])

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify Identity node in function was removed
        remaining_func_nodes = list(result.model.functions[function.identifier()])
        self.assertEqual(len(remaining_func_nodes), 1)  # Only Add node should remain
        self.assertEqual(remaining_func_nodes[0].op_type, "Add")

        # Verify Add node now uses func_input directly
        add_node_after = remaining_func_nodes[0]
        self.assertIs(add_node_after.inputs[0], func_input)
        self.assertIs(add_node_after.inputs[1], func_input)

    def test_multiple_graph_outputs_with_identity(self):
        """Test case where graph has multiple outputs and only one is the Identity output."""
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create intermediate value (not a graph input)
        const_node = ir.Node("", "Constant", inputs=[])
        const_node.outputs[0].name = "intermediate"
        const_node.outputs[0].shape = input_value.shape
        const_node.outputs[0].type = input_value.type

        # Create Identity node
        identity_node = ir.Node("", "Identity", inputs=[const_node.outputs[0]])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = input_value.shape
        identity_node.outputs[0].type = input_value.type

        # Create another output (not related to Identity)
        add_node = ir.Node("", "Add", inputs=[input_value, input_value])
        add_node.outputs[0].name = "other_output"
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[identity_node.outputs[0], add_node.outputs[0]],  # Multiple outputs
            nodes=[const_node, identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Store original output name
        original_identity_output_name = identity_node.outputs[0].name

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify Identity node was removed
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 2)  # Constant and Add nodes should remain

        # Verify we still have 2 outputs
        self.assertEqual(len(result.model.graph.outputs), 2)

        # Verify the identity output was replaced with the intermediate value
        outputs = result.model.graph.outputs
        identity_output = None
        other_output = None

        for output in outputs:
            if output.name == original_identity_output_name:
                identity_output = output
            elif output.name == "other_output":
                other_output = output

        self.assertIsNotNone(identity_output)
        self.assertIsNotNone(other_output)

        # The identity output should now be the intermediate value (renamed)
        self.assertEqual(identity_output.name, original_identity_output_name)
        self.assertIs(identity_output, const_node.outputs[0])

        # The other output should be unchanged
        self.assertEqual(other_output.name, "other_output")
        self.assertIs(other_output, add_node.outputs[0])

    def test_shape_merging_input_has_no_shape(self):
        """Test shape merging when input value has no shape but output has shape info."""
        # Create input value with no shape
        input_value = ir.Value(
            producer=None, name="input", shape=None, type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Identity node with output that has shape info
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = ir.Shape([2, 3, 4])
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Create Add node that uses the Identity output
        add_input = ir.val(
            "add_input", shape=ir.Shape([2, 3, 4]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], add_input])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = ir.Shape([2, 3, 4])
        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_value, add_input],
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify Identity node was removed
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)
        self.assertEqual(remaining_nodes[0].op_type, "Add")

        # Verify the input value now has the shape from the identity output
        self.assertIsNotNone(input_value.shape)
        self.assertEqual(input_value.shape, ir.Shape([2, 3, 4]))

    def test_shape_merging_output_has_no_shape(self):
        """Test shape merging when output value has no shape but input has shape info."""
        # Create input value with shape
        input_value = ir.val(
            "input", shape=ir.Shape([5, 6]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Identity node with output that has no shape info
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = None
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Create Add node that uses the Identity output
        add_input = ir.val(
            "add_input", shape=ir.Shape([5, 6]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], add_input])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = ir.Shape([5, 6])
        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_value, add_input],
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify the input value kept its original shape
        self.assertEqual(input_value.shape, ir.Shape([5, 6]))

    def test_shape_merging_identical_shapes(self):
        """Test shape merging when input and output have identical shape information."""
        # Create input value with shape
        input_value = ir.val(
            "input", shape=ir.Shape([3, 4, 5]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Identity node with identical shape
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = ir.Shape([3, 4, 5])
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Create Add node that uses the Identity output
        add_input = ir.val(
            "add_input", shape=ir.Shape([3, 4, 5]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], add_input])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = ir.Shape([3, 4, 5])
        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_value, add_input],
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify the input value kept its shape unchanged
        self.assertEqual(input_value.shape, ir.Shape([3, 4, 5]))

    def test_shape_merging_int_vs_symbolic_dims(self):
        """Test shape merging where one shape has int dims and other has symbolic dims."""
        # Create symbolic dimensions
        sym_dim = ir.SymbolicDim("batch")

        # Create input value with symbolic dimensions
        input_value = ir.Value(
            producer=None,
            name="input",
            shape=ir.Shape([sym_dim, 10]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        # Create Identity node with int dimensions in the same positions
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = ir.Shape([32, 10])  # int dims
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Create Add node that uses the Identity output
        add_input = ir.val(
            "add_input", shape=ir.Shape([32, 10]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], add_input])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = ir.Shape([32, 10])
        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_value, add_input],
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify the input value now has the int dimensions (preferred over symbolic)
        expected_shape = ir.Shape([32, 10])
        self.assertEqual(input_value.shape, expected_shape)

    def test_shape_merging_different_ranks_raises_error(self):
        """Test that merging shapes with different ranks raises an error."""
        # Create input value with 2D shape
        input_value = ir.Value(
            producer=None,
            name="input",
            shape=ir.Shape([2, 3]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        # Create Identity node with 3D shape (different rank)
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = ir.Shape([2, 3, 4])  # Different rank
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Create Add node
        add_input = ir.val(
            "add_input", shape=ir.Shape([2, 3, 4]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], add_input])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = ir.Shape([2, 3, 4])
        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_value, add_input],
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()

        # Should raise ValueError due to different ranks
        with self.assertRaises(ValueError) as context:
            pass_instance(model)

        self.assertIn("Shapes must have the same rank", str(context.exception))

    def test_type_copying_from_output_to_input(self):
        """Test that type is copied from output to input when input has no type."""
        # Create input value with no type
        input_value = ir.Value(producer=None, name="input", shape=ir.Shape([2, 2]), type=None)

        # Create Identity node with type info
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = ir.Shape([2, 2])
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.INT32)

        # Create Add node that uses the Identity output
        add_input = ir.val(
            "add_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.INT32)
        )
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], add_input])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = ir.Shape([2, 2])
        add_node.outputs[0].type = ir.TensorType(ir.DataType.INT32)

        graph = ir.Graph(
            inputs=[input_value, add_input],
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify the input value now has the type from the identity output
        self.assertIsNotNone(input_value.type)
        self.assertEqual(input_value.type, ir.TensorType(ir.DataType.INT32))

    def test_type_preservation_when_input_already_has_type(self):
        """Test that existing input type is preserved when both input and output have types."""
        # Create input value with existing type
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Identity node with different type
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = ir.Shape([2, 2])
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.INT32)  # Different type

        # Create Add node that uses the Identity output
        add_input = ir.val(
            "add_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.INT32)
        )
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], add_input])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = ir.Shape([2, 2])
        add_node.outputs[0].type = ir.TensorType(ir.DataType.INT32)

        graph = ir.Graph(
            inputs=[input_value, add_input],
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify the input value kept its original type (not overwritten)
        self.assertEqual(input_value.type, ir.TensorType(ir.DataType.FLOAT))

    def test_symbolic_dim_merging_edge_cases(self):
        """Test various symbolic dimension merging scenarios."""
        # Test case where one symbolic dim has value None, other has value
        sym_dim_no_value = ir.SymbolicDim(None)
        sym_dim_with_value = ir.SymbolicDim("batch")  # different name

        # Create input value with symbolic dim that has no value
        input_value = ir.Value(
            producer=None,
            name="input",
            shape=ir.Shape([sym_dim_no_value, 10]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        # Create Identity node with symbolic dim that has a value
        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = ir.Shape([sym_dim_with_value, 10])
        identity_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Create Add node that uses the Identity output
        add_input = ir.val(
            "add_input", shape=ir.Shape([16, 10]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        add_node = ir.Node("", "Add", inputs=[identity_node.outputs[0], add_input])
        add_node.outputs[0].name = "add_output"
        add_node.outputs[0].shape = ir.Shape([16, 10])
        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_value, add_input],
            outputs=[add_node.outputs[0]],
            nodes=[identity_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify the input value now has the merged shape
        self.assertIsNotNone(input_value.shape)
        assert input_value.shape is not None  # For type checker
        self.assertEqual(input_value.shape[0], sym_dim_with_value)
        self.assertEqual(input_value.shape[1], 10)

    def test_duplicate_identity_output_in_graph_outputs(self):
        """Test case where the same Identity output appears multiple times in graph outputs."""
        # Create intermediate value (not a graph input)
        const_node = ir.Node("", "Constant", inputs=[])
        const_node.outputs[0].name = "intermediate"
        const_node.outputs[0].shape = ir.Shape([2, 2])
        const_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Create Identity node
        identity_node = ir.Node("", "Identity", inputs=[const_node.outputs[0]])
        identity_node.outputs[0].name = "identity_output"
        identity_node.outputs[0].shape = const_node.outputs[0].shape
        identity_node.outputs[0].type = const_node.outputs[0].type

        # Create graph with duplicate outputs (same output appears twice)
        graph = ir.Graph(
            inputs=[],
            outputs=[identity_node.outputs[0], identity_node.outputs[0]],  # Same output twice
            nodes=[const_node, identity_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Store original output name
        original_identity_output_name = identity_node.outputs[0].name

        # Run the pass
        pass_instance = identity_elimination.IdentityEliminationPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify Identity node was removed
        remaining_nodes = list(result.model.graph)
        self.assertEqual(len(remaining_nodes), 1)  # Only Constant node should remain

        # Verify we still have 2 outputs
        self.assertEqual(len(result.model.graph.outputs), 2)

        # Verify BOTH outputs were replaced with the intermediate value
        outputs = result.model.graph.outputs
        for output in outputs:
            self.assertEqual(output.name, original_identity_output_name)
            self.assertIs(output, const_node.outputs[0])


if __name__ == "__main__":
    unittest.main()
