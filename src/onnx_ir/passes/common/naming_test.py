# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the name fix pass."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common import naming


class TestNameFixPass(unittest.TestCase):
    """Test cases for NameFixPass."""

    def test_assign_names_to_unnamed_values(self):
        """Test ensuring all values have names even if IR auto-assigned them."""
        # Create a simple model with auto-assigned names
        input_value = ir.Input(
            None, shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )  # Will get auto-assigned name when added to graph

        # Create Add node
        add_node = ir.Node("", "Add", inputs=[input_value, input_value])

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Verify IR has auto-assigned names
        self.assertIsNotNone(input_value.name)
        self.assertIsNotNone(add_node.outputs[0].name)

        # Store original names
        original_input_name = input_value.name
        original_output_name = add_node.outputs[0].name

        # Run the pass
        result = naming.NameFixPass()(model)

        # Verify the pass didn't modify anything (names were already assigned and unique)
        self.assertFalse(result.modified)

        # Verify names remain the same
        self.assertEqual(input_value.name, original_input_name)
        self.assertEqual(add_node.outputs[0].name, original_output_name)

    def test_assign_names_to_unnamed_nodes(self):
        """Test ensuring all nodes have names even if IR auto-assigned them."""
        # Create a simple model
        input_value = ir.Input(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create Add node - IR will auto-assign name when added to graph
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

        # Verify IR has auto-assigned node name
        self.assertIsNotNone(add_node.name)
        original_node_name = add_node.name

        # Run the pass
        result = naming.NameFixPass()(model)

        # Verify the pass didn't modify anything (node already had unique name)
        self.assertFalse(result.modified)

        # Verify node name remains the same
        self.assertEqual(add_node.name, original_node_name)

    def test_assigns_names_when_truly_unnamed(self):
        """Test that the pass assigns names when values/nodes are created without names and manually cleared."""
        # Create a model and manually clear names to test assignment
        input_value = ir.Input(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

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

        # Manually clear some names to test assignment
        add_node.name = None
        add_node.outputs[0].name = ""

        # Run the pass
        result = naming.NameFixPass()(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify names were assigned
        self.assertIsNotNone(add_node.name)
        self.assertIsNotNone(add_node.outputs[0].name)
        self.assertNotEqual(add_node.outputs[0].name, "")

    def test_handles_global_uniqueness_across_subgraphs(self):
        """Test that names are unique globally, including across subgraphs."""
        # Create main graph input
        main_input = ir.Input(
            "main_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create a simple subgraph for an If node
        # Subgraph input and output (with potential name conflicts)
        sub_input = ir.Input(
            "main_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )  # Same name as main input - should cause conflict

        sub_add_node = ir.Node("", "Add", inputs=[sub_input, sub_input])
        sub_add_node.outputs[0].name = "main_input"  # Another conflict
        sub_add_node.outputs[0].shape = sub_input.shape
        sub_add_node.outputs[0].type = sub_input.type

        subgraph = ir.Graph(
            inputs=[sub_input],
            outputs=[sub_add_node.outputs[0]],
            nodes=[sub_add_node],
            name="subgraph",
        )

        # Create condition input for If node
        condition_input = ir.Input(
            "condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
        )

        # Create If node with subgraph
        if_node = ir.Node(
            "",
            "If",
            inputs=[condition_input],
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, subgraph)
            },
        )
        if_node.outputs[0].name = "if_output"
        if_node.outputs[0].shape = main_input.shape
        if_node.outputs[0].type = main_input.type

        # Create main graph
        main_graph = ir.Graph(
            inputs=[main_input, condition_input],
            outputs=[if_node.outputs[0]],
            nodes=[if_node],
            name="main_graph",
        )

        model = ir.Model(main_graph, ir_version=10)

        # Run the pass
        result = naming.NameFixPass()(model)

        # Verify the pass was applied (should fix duplicates)
        self.assertTrue(result.modified)

        # Collect all value names to verify uniqueness
        all_value_names = set()

        # Main graph values
        for input_val in main_graph.inputs:
            self.assertIsNotNone(input_val.name)
            self.assertNotIn(
                input_val.name, all_value_names, f"Duplicate value name: {input_val.name}"
            )
            all_value_names.add(input_val.name)

        for output_val in main_graph.outputs:
            self.assertIsNotNone(output_val.name)
            if output_val.name not in all_value_names:  # Could be same as input
                all_value_names.add(output_val.name)

        # Node values in main graph
        for node in main_graph:
            for input_val in node.inputs:
                if input_val is not None:
                    if input_val.name not in all_value_names:
                        all_value_names.add(input_val.name)
            for output_val in node.outputs:
                if output_val.name not in all_value_names:
                    all_value_names.add(output_val.name)

        # Subgraph values
        for input_val in subgraph.inputs:
            self.assertIsNotNone(input_val.name)
            self.assertNotIn(
                input_val.name,
                all_value_names,
                f"Duplicate value name in subgraph: {input_val.name}",
            )
            all_value_names.add(input_val.name)

        for output_val in subgraph.outputs:
            if output_val.name not in all_value_names:  # Could be same as input
                all_value_names.add(output_val.name)

        # Node values in subgraph
        for node in subgraph:
            for input_val in node.inputs:
                if input_val is not None:
                    if input_val.name not in all_value_names:
                        all_value_names.add(input_val.name)
            for output_val in node.outputs:
                if output_val.name not in all_value_names:
                    all_value_names.add(output_val.name)

        # Verify main_input keeps its name (has precedence as graph input)
        self.assertEqual(main_input.name, "main_input")

    def test_handle_duplicate_value_names(self):
        """Test handling duplicate value names by making them unique."""
        # Create values with duplicate names
        input1 = ir.Input(
            "duplicate_name", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        input2 = ir.Input(
            "duplicate_name", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        add_node = ir.Node("", "Add", inputs=[input1, input2])
        add_node.outputs[0].name = "output"
        add_node.outputs[0].shape = input1.shape
        add_node.outputs[0].type = input1.type

        graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Verify both inputs have the same name initially
        self.assertEqual(input1.name, "duplicate_name")
        self.assertEqual(input2.name, "duplicate_name")

        # Run the pass
        result = naming.NameFixPass()(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify names are now unique
        self.assertNotEqual(input1.name, input2.name)
        # One should keep the original name, the other should have a suffix
        names = {input1.name, input2.name}
        self.assertIn("duplicate_name", names)
        self.assertTrue("duplicate_name_1" in names, f"Expected 'duplicate_name_1' in {names}")

    def test_handle_duplicate_node_names(self):
        """Test handling duplicate node names by making them unique."""
        input_value = ir.Input(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create nodes with duplicate names
        add_node1 = ir.Node("", "Add", inputs=[input_value, input_value])
        add_node1.name = "duplicate_node"
        add_node1.outputs[0].name = "output1"
        add_node1.outputs[0].shape = input_value.shape
        add_node1.outputs[0].type = input_value.type

        add_node2 = ir.Node("", "Add", inputs=[input_value, add_node1.outputs[0]])
        add_node2.name = "duplicate_node"  # Same name as first node
        add_node2.outputs[0].name = "output2"
        add_node2.outputs[0].shape = input_value.shape
        add_node2.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node2.outputs[0]],
            nodes=[add_node1, add_node2],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Verify both nodes have the same name initially
        self.assertEqual(add_node1.name, "duplicate_node")
        self.assertEqual(add_node2.name, "duplicate_node")

        # Run the pass
        result = naming.NameFixPass()(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify names are now unique
        self.assertNotEqual(add_node1.name, add_node2.name)
        # One should keep the original name, the other should have a suffix
        names = {add_node1.name, add_node2.name}
        self.assertIn("duplicate_node", names)
        self.assertTrue("duplicate_node_1" in names, f"Expected 'duplicate_node_1' in {names}")

    def test_no_modification_when_all_names_unique(self):
        """Test that the pass doesn't modify anything when all names are already unique."""
        input_value = ir.Input(
            "unique_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        add_node = ir.Node("", "Add", inputs=[input_value, input_value])
        add_node.name = "unique_node"
        add_node.outputs[0].name = "unique_output"
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Store original names
        original_input_name = input_value.name
        original_node_name = add_node.name
        original_output_name = add_node.outputs[0].name

        # Run the pass
        result = naming.NameFixPass()(model)

        # Verify the pass didn't modify anything
        self.assertFalse(result.modified)

        # Verify names remain unchanged
        self.assertEqual(input_value.name, original_input_name)
        self.assertEqual(add_node.name, original_node_name)
        self.assertEqual(add_node.outputs[0].name, original_output_name)

    def test_graph_inputs_outputs_have_precedence(self):
        """Test that graph inputs and outputs keep their names when there are conflicts."""
        # Create an input with a specific name
        input_value = ir.Input(
            "important_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create a node that produces an intermediate value with the same name
        add_node = ir.Node("", "Add", inputs=[input_value, input_value])
        add_node.outputs[0].name = "important_input"  # Conflicts with input name
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type

        # Create another node that uses the intermediate value and produces the final output
        mul_node = ir.Node("", "Mul", inputs=[add_node.outputs[0], input_value])
        mul_node.outputs[0].name = "important_output"
        mul_node.outputs[0].shape = input_value.shape
        mul_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[mul_node.outputs[0]],
            nodes=[add_node, mul_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        result = naming.NameFixPass()(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify input keeps its original name (has precedence)
        self.assertEqual(input_value.name, "important_input")

        # Verify output keeps its original name (has precedence)
        self.assertEqual(mul_node.outputs[0].name, "important_output")

        # Verify intermediate value got renamed to avoid conflict
        self.assertNotEqual(add_node.outputs[0].name, "important_input")
        self.assertTrue(add_node.outputs[0].name.startswith("important_input_"))


if __name__ == "__main__":
    unittest.main()
