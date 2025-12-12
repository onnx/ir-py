# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the output fix pass."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common import output_fix


class TestOutputFixPass(unittest.TestCase):
    """Test cases for OutputFixPass."""

    def test_add_identity_when_input_is_direct_output(self):
        """Test: Add Identity node when graph input is directly used as output."""
        # Create a simple model: input -> (direct) -> output
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[input_value],  # Input is directly used as output
            nodes=[],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify an Identity node was added
        nodes = list(result.model.graph)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].op_type, "Identity")

        # Verify the Identity node uses the input
        identity_node = nodes[0]
        self.assertIs(identity_node.inputs[0], input_value)

        # Verify the output is now the Identity node's output
        self.assertEqual(len(result.model.graph.outputs), 1)
        self.assertIs(result.model.graph.outputs[0], identity_node.outputs[0])

        # Verify the output name is preserved
        self.assertEqual(result.model.graph.outputs[0].name, "input")

    def test_no_modification_when_identity_exists(self):
        """Test: No modification when Identity node already exists between input and output."""
        # Create a model: input -> Identity -> output
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        identity_node = ir.Node("", "Identity", inputs=[input_value])
        identity_node.outputs[0].name = "output"
        identity_node.outputs[0].shape = input_value.shape
        identity_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[identity_node.outputs[0]],  # Output is Identity's output
            nodes=[identity_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass did NOT modify the model
        self.assertFalse(result.modified)

        # Verify structure is unchanged
        nodes = list(result.model.graph)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].op_type, "Identity")

    def test_no_modification_when_node_exists_between_input_and_output(self):
        """Test: No modification when a processing node exists between input and output."""
        # Create a model: input -> Add -> output
        input_value = ir.val(
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

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass did NOT modify the model
        self.assertFalse(result.modified)

        # Verify structure is unchanged
        nodes = list(result.model.graph)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].op_type, "Add")

    def test_multiple_inputs_one_direct_output(self):
        """Test: Add Identity for one input that's directly used as output, leave others alone."""
        # Create inputs
        input1 = ir.val(
            "input1", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        input2 = ir.val(
            "input2", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create a node that uses input2
        add_node = ir.Node("", "Add", inputs=[input2, input2])
        add_node.outputs[0].name = "output2"
        add_node.outputs[0].shape = input2.shape
        add_node.outputs[0].type = input2.type

        graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[input1, add_node.outputs[0]],  # input1 is directly used as output
            nodes=[add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify one Identity node was added
        nodes = list(result.model.graph)
        self.assertEqual(len(nodes), 2)  # Add + Identity

        identity_nodes = [n for n in nodes if n.op_type == "Identity"]
        self.assertEqual(len(identity_nodes), 1)

        # Verify the Identity node uses input1
        identity_node = identity_nodes[0]
        self.assertIs(identity_node.inputs[0], input1)

        # Verify outputs
        self.assertEqual(len(result.model.graph.outputs), 2)
        # First output should be the Identity node's output
        self.assertIs(result.model.graph.outputs[0], identity_node.outputs[0])
        # Second output should still be the Add node's output
        self.assertIs(result.model.graph.outputs[1], add_node.outputs[0])

    def test_multiple_direct_outputs(self):
        """Test: Add Identity nodes for multiple inputs used directly as outputs."""
        # Create inputs
        input1 = ir.val(
            "input1", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        input2 = ir.val(
            "input2", shape=ir.Shape([3, 3]), type=ir.TensorType(ir.DataType.INT32)
        )

        graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[input1, input2],  # Both inputs directly used as outputs
            nodes=[],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify two Identity nodes were added
        nodes = list(result.model.graph)
        self.assertEqual(len(nodes), 2)
        self.assertTrue(all(n.op_type == "Identity" for n in nodes))

        # Verify both inputs are used by Identity nodes
        identity_inputs = [n.inputs[0] for n in nodes]
        self.assertIn(input1, identity_inputs)
        self.assertIn(input2, identity_inputs)

        # Verify outputs are now Identity nodes' outputs
        self.assertEqual(len(result.model.graph.outputs), 2)
        for output in result.model.graph.outputs:
            self.assertIsNotNone(output.producer())
            self.assertEqual(output.producer().op_type, "Identity")

    def test_empty_graph(self):
        """Test: Pass on an empty graph."""
        graph = ir.Graph(inputs=[], outputs=[], nodes=[], name="empty_graph")
        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass did not modify the model
        self.assertFalse(result.modified)

        # Verify structure is unchanged
        self.assertEqual(len(list(result.model.graph)), 0)

    def test_graph_with_no_direct_input_output(self):
        """Test: Graph with inputs and outputs but no direct connections."""
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create a Constant node (output doesn't come from input)
        const_node = ir.Node("", "Constant", inputs=[])
        const_node.outputs[0].name = "output"
        const_node.outputs[0].shape = ir.Shape([2, 2])
        const_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[const_node.outputs[0]],
            nodes=[const_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass did not modify the model
        self.assertFalse(result.modified)

        # Verify structure is unchanged
        nodes = list(result.model.graph)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].op_type, "Constant")

    def test_preserve_output_metadata(self):
        """Test: Output metadata (shape, type, name) is preserved."""
        input_value = ir.val(
            "my_input", shape=ir.Shape([5, 10]), type=ir.TensorType(ir.DataType.INT64)
        )
        input_value.doc_string = "Test doc string"
        input_value.metadata_props["custom_key"] = "custom_value"

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[input_value],
            nodes=[],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify metadata is preserved
        output = result.model.graph.outputs[0]
        self.assertEqual(output.name, "my_input")
        self.assertEqual(output.shape, ir.Shape([5, 10]))
        self.assertEqual(output.type, ir.TensorType(ir.DataType.INT64))
        self.assertEqual(output.doc_string, "Test doc string")
        self.assertEqual(output.metadata_props.get("custom_key"), "custom_value")

    def test_subgraph_with_direct_input_output(self):
        """Test: Add Identity in subgraphs (e.g., in If node)."""
        # Create main graph input
        main_input = ir.val(
            "main_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create a condition input for If
        condition = ir.val(
            "condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
        )

        # Create then_branch subgraph with direct input->output
        then_input = ir.val(
            "then_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        then_branch = ir.Graph(
            inputs=[then_input],
            outputs=[then_input],  # Direct input->output
            nodes=[],
            name="then_branch",
        )

        # Create else_branch subgraph with a node
        else_input = ir.val(
            "else_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        else_add = ir.Node("", "Add", inputs=[else_input, else_input])
        else_add.outputs[0].name = "else_output"
        else_add.outputs[0].shape = else_input.shape
        else_add.outputs[0].type = else_input.type
        else_branch = ir.Graph(
            inputs=[else_input],
            outputs=[else_add.outputs[0]],
            nodes=[else_add],
            name="else_branch",
        )

        # Create If node with subgraphs
        if_node = ir.Node("", "If", inputs=[condition])
        if_node.attributes["then_branch"] = ir.AttrGraph("then_branch", then_branch)
        if_node.attributes["else_branch"] = ir.AttrGraph("else_branch", else_branch)
        if_node.outputs[0].name = "if_output"
        if_node.outputs[0].shape = main_input.shape
        if_node.outputs[0].type = main_input.type

        # Create main graph
        main_graph = ir.Graph(
            inputs=[main_input, condition],
            outputs=[if_node.outputs[0]],
            nodes=[if_node],
            name="main_graph",
        )

        model = ir.Model(main_graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify Identity was added in then_branch
        if_node = next(iter(result.model.graph))
        then_branch_after = if_node.attributes["then_branch"].value
        then_nodes = list(then_branch_after)
        self.assertEqual(len(then_nodes), 1)
        self.assertEqual(then_nodes[0].op_type, "Identity")

        # Verify else_branch was not modified
        else_branch_after = if_node.attributes["else_branch"].value
        else_nodes = list(else_branch_after)
        self.assertEqual(len(else_nodes), 1)
        self.assertEqual(else_nodes[0].op_type, "Add")

    def test_function_with_direct_input_output(self):
        """Test: Add Identity in functions."""
        # Create function with direct input->output
        func_input = ir.val(
            "func_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        func_graph = ir.Graph(
            inputs=[func_input],
            outputs=[func_input],  # Direct input->output
            nodes=[],
            name="test_function_graph",
        )

        function = ir.Function(
            domain="test_domain",
            name="test_function",
            graph=func_graph,
            attributes=[],
        )

        # Create main graph that calls the function
        main_input = ir.val(
            "main_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

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
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify Identity was added in the function
        func_after = result.model.functions[function.identifier()]
        func_nodes = list(func_after)
        self.assertEqual(len(func_nodes), 1)
        self.assertEqual(func_nodes[0].op_type, "Identity")

        # Verify the function output is now the Identity node's output
        identity_node = func_nodes[0]
        self.assertIs(func_after.outputs[0], identity_node.outputs[0])
        self.assertIs(identity_node.inputs[0], func_input)

    def test_same_input_used_multiple_times_as_output(self):
        """Test: Same input used multiple times as output."""
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[input_value, input_value],  # Same input used twice
            nodes=[],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify two Identity nodes were added
        nodes = list(result.model.graph)
        self.assertEqual(len(nodes), 2)
        self.assertTrue(all(n.op_type == "Identity" for n in nodes))

        # Verify both use the same input
        for node in nodes:
            self.assertIs(node.inputs[0], input_value)

        # Verify outputs are now different Identity nodes' outputs
        self.assertEqual(len(result.model.graph.outputs), 2)
        self.assertIsNot(result.model.graph.outputs[0], result.model.graph.outputs[1])

    def test_nested_subgraphs(self):
        """Test: Handle nested subgraphs (subgraph within subgraph)."""
        # Create innermost graph with direct input->output
        inner_input = ir.val(
            "inner_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        inner_graph = ir.Graph(
            inputs=[inner_input],
            outputs=[inner_input],  # Direct input->output
            nodes=[],
            name="inner_graph",
        )

        # Create middle graph with an If node containing the inner graph
        middle_input = ir.val(
            "middle_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        middle_condition = ir.val(
            "middle_condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
        )

        middle_if = ir.Node("", "If", inputs=[middle_condition])
        middle_if.attributes["then_branch"] = ir.AttrGraph("then_branch", inner_graph)
        middle_if.attributes["else_branch"] = ir.AttrGraph("else_branch", inner_graph)
        middle_if.outputs[0].name = "middle_output"
        middle_if.outputs[0].shape = middle_input.shape
        middle_if.outputs[0].type = middle_input.type

        middle_graph = ir.Graph(
            inputs=[middle_input, middle_condition],
            outputs=[middle_if.outputs[0]],
            nodes=[middle_if],
            name="middle_graph",
        )

        # Create outer graph with an If node containing the middle graph
        outer_input = ir.val(
            "outer_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        outer_condition = ir.val(
            "outer_condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
        )

        outer_if = ir.Node("", "If", inputs=[outer_condition])
        outer_if.attributes["then_branch"] = ir.AttrGraph("then_branch", middle_graph)
        outer_if.attributes["else_branch"] = ir.AttrGraph("else_branch", middle_graph)
        outer_if.outputs[0].name = "outer_output"
        outer_if.outputs[0].shape = outer_input.shape
        outer_if.outputs[0].type = outer_input.type

        main_graph = ir.Graph(
            inputs=[outer_input, outer_condition],
            outputs=[outer_if.outputs[0]],
            nodes=[outer_if],
            name="main_graph",
        )

        model = ir.Model(main_graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Navigate to innermost graphs and verify Identity nodes were added
        outer_if = next(iter(result.model.graph))
        outer_then = outer_if.attributes["then_branch"].value
        middle_if_node = next(iter(outer_then))
        inner_then = middle_if_node.attributes["then_branch"].value

        inner_nodes = list(inner_then)
        self.assertEqual(len(inner_nodes), 1)
        self.assertEqual(inner_nodes[0].op_type, "Identity")

    def test_pass_is_idempotent(self):
        """Test: Running the pass twice should not modify the model again."""
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[input_value],
            nodes=[],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass first time
        pass_instance = output_fix.OutputFixPass()
        result1 = pass_instance(model)
        self.assertTrue(result1.modified)

        # Run the pass second time on the result
        result2 = pass_instance(result1.model)
        self.assertFalse(result2.modified)

        # Verify structure remains the same
        nodes = list(result2.model.graph)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].op_type, "Identity")

    def test_output_order_preserved(self):
        """Test: The order of outputs is preserved after transformation."""
        input1 = ir.val(
            "input1", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        input2 = ir.val(
            "input2", shape=ir.Shape([3, 3]), type=ir.TensorType(ir.DataType.INT32)
        )
        input3 = ir.val(
            "input3", shape=ir.Shape([4, 4]), type=ir.TensorType(ir.DataType.INT64)
        )

        # Create a processing node for input2
        add_node = ir.Node("", "Add", inputs=[input2, input2])
        add_node.outputs[0].name = "processed_input2"
        add_node.outputs[0].shape = input2.shape
        add_node.outputs[0].type = input2.type

        graph = ir.Graph(
            inputs=[input1, input2, input3],
            outputs=[input1, add_node.outputs[0], input3],  # input1 and input3 are direct
            nodes=[add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify output order is preserved (by checking names)
        output_names = [output.name for output in result.model.graph.outputs]
        self.assertEqual(output_names, ["input1", "processed_input2", "input3"])

        # Verify first and third outputs are now Identity outputs
        self.assertEqual(result.model.graph.outputs[0].producer().op_type, "Identity")
        self.assertEqual(result.model.graph.outputs[1].producer().op_type, "Add")
        self.assertEqual(result.model.graph.outputs[2].producer().op_type, "Identity")

    def test_name_collision_avoided(self):
        """Test: Verify that renaming original inputs doesn't cause name collisions."""
        input_value = ir.val(
            "my_value", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create a node that produces a value with name that could collide
        add_node = ir.Node("", "Add", inputs=[input_value, input_value])
        add_node.outputs[0].name = "my_value_orig"  # This name could collide
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[input_value],  # Direct input as output
            nodes=[add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify no assertion errors or issues occurred
        # (The implementation should handle this gracefully)
        self.assertEqual(len(list(result.model.graph)), 2)  # Add + Identity

    def test_mixed_outputs_with_initializer_and_input(self):
        """Test: Handle case where outputs include both inputs and computed values."""
        input_value = ir.val(
            "input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Create a Constant node (not from input)
        const_node = ir.Node("", "Constant", inputs=[])
        const_node.outputs[0].name = "constant"
        const_node.outputs[0].shape = ir.Shape([2, 2])
        const_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        # Create an Add node
        add_node = ir.Node("", "Add", inputs=[input_value, const_node.outputs[0]])
        add_node.outputs[0].name = "sum"
        add_node.outputs[0].shape = input_value.shape
        add_node.outputs[0].type = input_value.type

        graph = ir.Graph(
            inputs=[input_value],
            outputs=[
                input_value,
                const_node.outputs[0],
                add_node.outputs[0],
            ],  # Mixed outputs
            nodes=[const_node, add_node],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied (only for the direct input)
        self.assertTrue(result.modified)

        # Verify only one Identity was added (for the direct input)
        identity_nodes = [n for n in result.model.graph if n.op_type == "Identity"]
        self.assertEqual(len(identity_nodes), 1)

        # Verify the first output is now an Identity node output
        self.assertEqual(result.model.graph.outputs[0].producer().op_type, "Identity")
        # The second output should still be from Constant
        self.assertEqual(result.model.graph.outputs[1].producer().op_type, "Constant")
        # The third output should still be from Add
        self.assertEqual(result.model.graph.outputs[2].producer().op_type, "Add")

    def test_shape_metadata_types_preserved(self):
        """Test: Various shape types (dynamic, symbolic) are preserved correctly."""
        # Input with concrete shape
        input_concrete = ir.val(
            "concrete",
            shape=ir.Shape([2, 3, 4]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        # Input with dynamic shape
        input_dynamic = ir.val(
            "dynamic",
            shape=ir.Shape([None, 3, None]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        # Input with symbolic shape
        input_symbolic = ir.val(
            "symbolic",
            shape=ir.Shape(["batch", "seq", "hidden"]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        graph = ir.Graph(
            inputs=[input_concrete, input_dynamic, input_symbolic],
            outputs=[input_concrete, input_dynamic, input_symbolic],
            nodes=[],
            name="test_graph",
        )

        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify shapes are preserved
        outputs = result.model.graph.outputs
        self.assertEqual(outputs[0].shape, ir.Shape([2, 3, 4]))
        self.assertEqual(outputs[1].shape, ir.Shape([None, 3, None]))
        self.assertEqual(outputs[2].shape, ir.Shape(["batch", "seq", "hidden"]))

    def test_loop_subgraph_with_direct_input_output(self):
        """Test: Add Identity in Loop node subgraphs."""
        # Create loop body with direct input->output
        iter_num = ir.val(
            "iter_num", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.INT64)
        )
        cond_in = ir.val("cond_in", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL))
        loop_var = ir.val(
            "loop_var", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )

        # Loop body with direct input->output for both cond_in and loop_var
        loop_body = ir.Graph(
            inputs=[iter_num, cond_in, loop_var],
            outputs=[cond_in, loop_var],  # Both directly passed through
            nodes=[],
            name="loop_body",
        )

        # Create main graph with Loop node
        main_input = ir.val(
            "main_input", shape=ir.Shape([2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        max_trip_count = ir.val(
            "max_trip_count",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        condition = ir.val(
            "condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
        )

        loop_node = ir.Node("", "Loop", inputs=[max_trip_count, condition, main_input])
        loop_node.attributes["body"] = ir.AttrGraph("body", loop_body)
        loop_node.outputs[0].name = "loop_output"
        loop_node.outputs[0].shape = main_input.shape
        loop_node.outputs[0].type = main_input.type

        main_graph = ir.Graph(
            inputs=[main_input, max_trip_count, condition],
            outputs=[loop_node.outputs[0]],
            nodes=[loop_node],
            name="main_graph",
        )

        model = ir.Model(main_graph, ir_version=10)

        # Run the pass
        pass_instance = output_fix.OutputFixPass()
        result = pass_instance(model)

        # Verify the pass was applied
        self.assertTrue(result.modified)

        # Verify Identity was added in loop body for direct pass-throughs
        loop_node_after = next(iter(result.model.graph))
        loop_body_after = loop_node_after.attributes["body"].value
        loop_body_nodes = list(loop_body_after)

        # Should have two Identity nodes (one for cond_in, one for loop_var)
        identity_nodes = [n for n in loop_body_nodes if n.op_type == "Identity"]
        self.assertEqual(len(identity_nodes), 2)

        # Verify both outputs are now from Identity nodes
        self.assertEqual(loop_body_after.outputs[0].producer().op_type, "Identity")
        self.assertEqual(loop_body_after.outputs[1].producer().op_type, "Identity")


if __name__ == "__main__":
    unittest.main()
