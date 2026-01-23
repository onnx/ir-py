# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the _extractor module."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir._convenience import _extractor


class ExtractTest(unittest.TestCase):
    def setUp(self):
        """Set up a simple graph for testing."""
        # Create a simple graph: input -> Add -> Mul -> output
        # input (initializer: [1, 2, 3])
        # |
        # Add (with constant [10, 20, 30])
        # |
        # intermediate
        # |
        # Mul (with constant [2, 2, 2])
        # |
        # output
        self.input_tensor = ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="input")
        self.const1_tensor = ir.tensor(np.array([10, 20, 30], dtype=np.float32), name="const1")
        self.const2_tensor = ir.tensor(np.array([2, 2, 2], dtype=np.float32), name="const2")

        self.input_val = ir.val(
            "input",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=self.input_tensor,
        )
        self.const1_val = ir.val(
            "const1",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=self.const1_tensor,
        )
        self.const2_val = ir.val(
            "const2",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=self.const2_tensor,
        )

        self.add_node = ir.node(
            "Add",
            inputs=[self.input_val, self.const1_val],
            outputs=[ir.val("intermediate", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        self.mul_node = ir.node(
            "Mul",
            inputs=[self.add_node.outputs[0], self.const2_val],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        self.graph = ir.Graph(
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
            nodes=[self.add_node, self.mul_node],
            initializers=[self.input_val, self.const1_val, self.const2_val],
            name="test_graph",
        )

    def test_extract_full_graph(self):
        """Test extracting the entire graph."""
        extracted = _extractor.extract(
            self.graph,
            inputs=["input"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 2)
        self.assertEqual(len(extracted.inputs), 1)
        self.assertEqual(len(extracted.outputs), 1)
        self.assertEqual(extracted.inputs[0].name, "input")
        self.assertEqual(extracted.outputs[0].name, "output")

    def test_extract_subgraph(self):
        """Test extracting a subgraph from intermediate to output."""
        extracted = _extractor.extract(
            self.graph,
            inputs=[self.add_node.outputs[0]],
            outputs=[self.mul_node.outputs[0]],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted.inputs[0].name, "intermediate")
        self.assertEqual(extracted.outputs[0].name, "output")
        # Should include const2 as an initializer
        self.assertEqual(len(extracted.initializers), 1)

    def test_extract_with_string_names(self):
        """Test extracting using string names instead of Value objects."""
        extracted = _extractor.extract(
            self.graph,
            inputs=["intermediate"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted.inputs[0].name, "intermediate")
        self.assertEqual(extracted.outputs[0].name, "output")

    def test_extract_with_mixed_inputs(self):
        """Test extracting using a mix of Value objects and strings."""
        extracted = _extractor.extract(
            self.graph,
            inputs=[self.input_val],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted.inputs[0].name, "input")
        self.assertEqual(extracted.outputs[0].name, "output")

    def test_extract_preserves_node_order(self):
        """Test that extraction preserves the original node order."""
        extracted = _extractor.extract(
            self.graph,
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
        )

        self.assertEqual(extracted[0].op_type, "Add")
        self.assertEqual(extracted[1].op_type, "Mul")

    def test_extract_raises_on_value_not_found(self):
        """Test that ValueError is raised when a value name is not found."""
        with self.assertRaisesRegex(
            ValueError, "Value with name 'nonexistent' not found in the graph"
        ):
            _extractor.extract(
                self.graph,
                inputs=["nonexistent"],
                outputs=["output"],
            )

    def test_extract_raises_on_value_from_different_graph(self):
        """Test that ValueError is raised when Value is from a different graph."""
        other_graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            name="other_graph",
        )
        other_value = ir.val("other", dtype=ir.DataType.FLOAT)
        other_value._graph = other_graph

        with self.assertRaisesRegex(
            ValueError, "Value '.*' does not belong to the given Graph"
        ):
            _extractor.extract(
                self.graph,
                inputs=[other_value],
                outputs=[self.mul_node.outputs[0]],
            )

    def test_extract_from_function(self):
        """Test extracting from a Function object."""
        function = ir.Function(
            domain="test",
            name="test_func",
            graph=self.graph,
            attributes=[],
        )

        extracted = _extractor.extract(
            function,
            inputs=["input"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted.inputs[0].name, "input")
        self.assertEqual(extracted.outputs[0].name, "output")

    def test_extract_from_graph_view(self):
        """Test extracting from a GraphView object."""
        graph_view = ir.GraphView(
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
            nodes=[self.add_node, self.mul_node],
        )

        extracted = _extractor.extract(
            graph_view,
            inputs=["intermediate"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted.inputs[0].name, "intermediate")
        self.assertEqual(extracted.outputs[0].name, "output")

    def test_extract_preserves_metadata(self):
        """Test that extraction preserves graph metadata."""
        # Create a fresh graph with metadata for this test
        input_val2 = ir.val(
            "input2",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="input2"),
        )
        const1_val2 = ir.val(
            "const1_2",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([10, 20, 30], dtype=np.float32), name="const1_2"),
        )
        const2_val2 = ir.val(
            "const2_2",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([2, 2, 2], dtype=np.float32), name="const2_2"),
        )

        add_node2 = ir.node(
            "Add",
            inputs=[input_val2, const1_val2],
            outputs=[ir.val("intermediate2", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        mul_node2 = ir.node(
            "Mul",
            inputs=[add_node2.outputs[0], const2_val2],
            outputs=[ir.val("output2", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph_with_metadata = ir.Graph(
            inputs=[input_val2],
            outputs=[mul_node2.outputs[0]],
            nodes=[add_node2, mul_node2],
            initializers=[const1_val2, const2_val2],
            name="metadata_graph",
            doc_string="Test documentation",
            opset_imports={"": 18},
            metadata_props={"key": "value"},
        )

        extracted = _extractor.extract(
            graph_with_metadata,
            inputs=["input2"],
            outputs=["output2"],
        )

        self.assertEqual(extracted.name, "metadata_graph")
        self.assertEqual(extracted.doc_string, "Test documentation")
        self.assertEqual(extracted.opset_imports, {"": 18})
        self.assertEqual(extracted.metadata_props, {"key": "value"})


class ExtractComplexGraphTest(unittest.TestCase):
    def setUp(self):
        """Set up a more complex graph with multiple paths."""
        # Create a diamond-shaped graph:
        #       input
        #      /     \
        #    Add1   Add2
        #      \     /
        #       Mul
        #        |
        #      output
        self.input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[3])
        self.const1_val = ir.val(
            "const1",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="const1"),
        )
        self.const2_val = ir.val(
            "const2",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([4, 5, 6], dtype=np.float32), name="const2"),
        )

        self.add1_node = ir.node(
            "Add",
            inputs=[self.input_val, self.const1_val],
            outputs=[ir.val("intermediate1", dtype=ir.DataType.FLOAT, shape=[3])],
            name="add1",
        )
        self.add2_node = ir.node(
            "Add",
            inputs=[self.input_val, self.const2_val],
            outputs=[ir.val("intermediate2", dtype=ir.DataType.FLOAT, shape=[3])],
            name="add2",
        )
        self.mul_node = ir.node(
            "Mul",
            inputs=[self.add1_node.outputs[0], self.add2_node.outputs[0]],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
            name="mul",
        )

        self.graph = ir.Graph(
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
            nodes=[self.add1_node, self.add2_node, self.mul_node],
            initializers=[self.const1_val, self.const2_val],
            name="diamond_graph",
        )

    def test_extract_includes_all_predecessors(self):
        """Test that extraction includes all predecessor nodes."""
        extracted = _extractor.extract(
            self.graph,
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
        )

        # Should include all three nodes: add1, add2, mul
        self.assertEqual(len(extracted), 3)
        node_names = {node.name for node in extracted}
        self.assertEqual(node_names, {"add1", "add2", "mul"})

    def test_extract_single_branch(self):
        """Test extracting a single branch of the diamond."""
        extracted = _extractor.extract(
            self.graph,
            inputs=["input"],
            outputs=["intermediate1"],
        )

        # Should only include add1
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].name, "add1")
        # const1 is not an initializer in the input, so it should be in the extracted initializers
        self.assertEqual(len(extracted.initializers), 1)

    def test_extract_from_intermediate_to_output(self):
        """Test extracting from intermediate values to final output."""
        extracted = _extractor.extract(
            self.graph,
            inputs=["intermediate1", "intermediate2"],
            outputs=["output"],
        )

        # Should only include mul node
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].name, "mul")
        self.assertEqual(len(extracted.inputs), 2)


class ExtractEdgeCasesTest(unittest.TestCase):
    def test_extract_with_no_nodes(self):
        """Test extracting when input is directly connected to output (no ops)."""
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[3])

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[input_val],
            nodes=[],
            name="passthrough_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=[input_val],
            outputs=[input_val],
        )

        self.assertEqual(len(extracted), 0)
        self.assertEqual(len(extracted.inputs), 1)
        self.assertEqual(len(extracted.outputs), 1)
        self.assertEqual(extracted.inputs[0].name, "input")
        self.assertEqual(extracted.outputs[0].name, "input")

    def test_extract_with_multiple_outputs(self):
        """Test extracting a graph with multiple outputs."""
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[3])
        const_val = ir.val(
            "const",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="const"),
        )

        add_node = ir.node(
            "Add",
            inputs=[input_val, const_val],
            outputs=[ir.val("sum", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        mul_node = ir.node(
            "Mul",
            inputs=[input_val, const_val],
            outputs=[ir.val("product", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[add_node.outputs[0], mul_node.outputs[0]],
            nodes=[add_node, mul_node],
            initializers=[const_val],
            name="multi_output_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=[input_val],
            outputs=[add_node.outputs[0], mul_node.outputs[0]],
        )

        self.assertEqual(len(extracted), 2)
        self.assertEqual(len(extracted.outputs), 2)
        output_names = {out.name for out in extracted.outputs}
        self.assertEqual(output_names, {"sum", "product"})

    def test_extract_single_node(self):
        """Test extracting a single node."""
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[3])
        const_val = ir.val(
            "const",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="const"),
        )

        add_node = ir.node(
            "Add",
            inputs=[input_val, const_val],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            initializers=[const_val],
            name="single_node_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=[input_val],
            outputs=[add_node.outputs[0]],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].op_type, "Add")

    def test_extract_with_multiple_inputs(self):
        """Test extracting with multiple input values."""
        input1 = ir.val("input1", dtype=ir.DataType.FLOAT, shape=[3])
        input2 = ir.val("input2", dtype=ir.DataType.FLOAT, shape=[3])

        add_node = ir.node(
            "Add",
            inputs=[input1, input2],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="multi_input_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=["input1", "input2"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(len(extracted.inputs), 2)
        input_names = {inp.name for inp in extracted.inputs}
        self.assertEqual(input_names, {"input1", "input2"})

    def test_extract_with_shared_input(self):
        """Test extracting when a value is used by multiple nodes."""
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[3])
        const_val = ir.val(
            "const",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="const"),
        )

        # Both nodes use the same input_val
        add_node = ir.node(
            "Add",
            inputs=[input_val, const_val],
            outputs=[ir.val("sum", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        mul_node = ir.node(
            "Mul",
            inputs=[input_val, const_val],
            outputs=[ir.val("product", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        concat_node = ir.node(
            "Concat",
            inputs=[add_node.outputs[0], mul_node.outputs[0]],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[6])],
            attributes={"axis": 0},
        )

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[concat_node.outputs[0]],
            nodes=[add_node, mul_node, concat_node],
            initializers=[const_val],
            name="shared_input_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=["input"],
            outputs=["output"],
        )

        # Should include all three nodes since they're all needed
        self.assertEqual(len(extracted), 3)

    def test_extract_raises_on_output_not_found(self):
        """Test that ValueError is raised when an output value is not found."""
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[3])
        graph = ir.Graph(
            inputs=[input_val],
            outputs=[input_val],
            nodes=[],
            name="simple_graph",
        )

        with self.assertRaisesRegex(
            ValueError, "Value with name 'nonexistent_output' not found in the graph"
        ):
            _extractor.extract(
                graph,
                inputs=["input"],
                outputs=["nonexistent_output"],
            )

    def test_extract_with_node_with_optional_inputs(self):
        """Test extracting nodes that have None/optional inputs."""
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[3, 4])

        # Reshape node can have optional inputs
        shape_val = ir.val(
            "shape",
            dtype=ir.DataType.INT64,
            shape=[1],
            const_value=ir.tensor(np.array([12], dtype=np.int64), name="shape"),
        )
        reshape_node = ir.node(
            "Reshape",
            inputs=[input_val, shape_val],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[12])],
        )

        graph = ir.Graph(
            inputs=[input_val],
            outputs=[reshape_node.outputs[0]],
            nodes=[reshape_node],
            initializers=[shape_val],
            name="reshape_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=["input"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].op_type, "Reshape")

    def test_extract_partial_from_multi_output_node(self):
        """Test extracting only one output from a node with multiple outputs."""
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[10])

        # Split creates multiple outputs
        split_node = ir.node(
            "Split",
            inputs=[input_val],
            outputs=[
                ir.val("output1", dtype=ir.DataType.FLOAT, shape=[5]),
                ir.val("output2", dtype=ir.DataType.FLOAT, shape=[5]),
            ],
            attributes={"axis": 0},
        )

        graph = ir.Graph(
            inputs=[input_val],
            outputs=split_node.outputs,
            nodes=[split_node],
            name="split_graph",
        )

        # Extract only one output
        extracted = _extractor.extract(
            graph,
            inputs=["input"],
            outputs=["output1"],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(len(extracted.outputs), 1)
        self.assertEqual(extracted.outputs[0].name, "output1")


class ExtractSubgraphAttributesTest(unittest.TestCase):
    def test_extract_with_if_node(self):
        """Test extracting a graph containing an If node with subgraph attributes.

        Graph structure:
            condition, x, y (inputs)
                  |
                  v
                 If  <--- then_branch: Add(x, y) -> then_out
                  |   <--- else_branch: Identity(x) -> else_out
                  v
               result (output)
        """
        # Create main graph values
        condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
        y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[3])

        # Create then branch subgraph that uses x and y from parent
        then_add = ir.node(
            "Add",
            inputs=[x, y],
            outputs=[ir.val("then_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        then_graph = ir.Graph(
            inputs=[],
            outputs=[then_add.outputs[0]],
            nodes=[then_add],
            name="then_branch",
        )

        # Create else branch subgraph that uses x from parent
        else_identity = ir.node(
            "Identity",
            inputs=[x],
            outputs=[ir.val("else_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        else_graph = ir.Graph(
            inputs=[],
            outputs=[else_identity.outputs[0]],
            nodes=[else_identity],
            name="else_branch",
        )

        # Create If node
        if_node = ir.node(
            "If",
            inputs=[condition],
            outputs=[ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": then_graph,
                "else_branch": else_graph,
            },
        )

        # Create main graph
        graph = ir.Graph(
            inputs=[condition, x, y],
            outputs=[if_node.outputs[0]],
            nodes=[if_node],
            name="if_graph",
        )

        # Extract the graph - should include the If node and trace back to x, y inputs
        extracted = _extractor.extract(
            graph,
            inputs=["condition", "x", "y"],
            outputs=["result"],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].op_type, "If")
        # Verify the subgraphs are preserved
        then_attr = extracted[0].attributes["then_branch"]
        else_attr = extracted[0].attributes["else_branch"]
        self.assertIsNotNone(then_attr)
        self.assertIsNotNone(else_attr)

    def test_extract_partial_with_if_node_dependencies(self):
        """Test extraction where subgraph references create implicit dependencies.

        Graph structure:
            const_a, const_b (initializers, NOT in input list)
                   |
                   v
                Add (implicit_node)
                   |
                   v
              computed_val
                   |
            condition (input)
                   |
                   v
                  If  <--- then_branch: Identity(computed_val) -> then_out
                   |  <--- else_branch: Constant(0) -> else_out
                   v
                result (output)

        Note: When extracting with only 'condition' as input, the Add node should
        still be included because the If node's then_branch references computed_val.
        """
        # Create constants that will NOT be in the input list
        const_a = ir.val(
            "const_a",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="const_a"),
        )
        const_b = ir.val(
            "const_b",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([4, 5, 6], dtype=np.float32), name="const_b"),
        )

        # This node produces a value from const_a and const_b
        implicit_node = ir.node(
            "Add",
            inputs=[const_a, const_b],
            outputs=[ir.val("computed_val", dtype=ir.DataType.FLOAT, shape=[3])],
            name="implicit_node",
        )

        condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])

        # Then branch uses the computed value from implicit_node
        then_identity = ir.node(
            "Identity",
            inputs=[implicit_node.outputs[0]],
            outputs=[ir.val("then_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        then_graph = ir.Graph(
            inputs=[],
            outputs=[then_identity.outputs[0]],
            nodes=[then_identity],
            name="then_branch",
        )

        # Else branch doesn't use computed_val
        else_const = ir.node(
            "Constant",
            inputs=[],
            outputs=[ir.val("else_out", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={"value": ir.tensor(np.zeros(3, dtype=np.float32), name="zero")},
        )
        else_graph = ir.Graph(
            inputs=[],
            outputs=[else_const.outputs[0]],
            nodes=[else_const],
            name="else_branch",
        )

        if_node = ir.node(
            "If",
            inputs=[condition],
            outputs=[ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": then_graph,
                "else_branch": else_graph,
            },
            name="if_node",
        )

        graph = ir.Graph(
            inputs=[condition],
            outputs=[if_node.outputs[0]],
            nodes=[implicit_node, if_node],
            initializers=[const_a, const_b],
            name="implicit_dependency_graph",
        )

        # Extract with ONLY condition as input (not const_a or const_b)
        # The implicit_node should still be included because then_branch references computed_val
        extracted = _extractor.extract(
            graph,
            inputs=["condition"],
            outputs=["result"],
        )

        # Verify implicit_node is included even though its inputs aren't in the input list
        self.assertEqual(len(extracted), 2)
        node_names = {node.name for node in extracted}
        self.assertIn("if_node", node_names)
        self.assertIn("implicit_node", node_names)
        # Verify the initializers are preserved
        self.assertIn("const_a", extracted.initializers)
        self.assertIn("const_b", extracted.initializers)

    def test_extract_nested_subgraphs(self):
        r"""Test extraction with nested subgraphs (If inside If).

        Graph structure:
            input, outer_cond, inner_cond (inputs)
                          |
                          v
                     Outer If
                      /      \
          then_branch        else_branch
               |                  |
            Inner If         Identity(input)
             /    \                |
        then: Add  else: Identity   |
         (input)    (input)         |
            \         /             /
             \       /             /
              inner_result        /
                    \            /
                     \____ _____/
                           |
                           v
                        final (output)

        Note: Both inner branches reference 'input' from the outermost parent graph.
        """
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[3])
        outer_cond = ir.val("outer_cond", dtype=ir.DataType.BOOL, shape=[])
        inner_cond = ir.val("inner_cond", dtype=ir.DataType.BOOL, shape=[])

        # Inner then branch
        inner_then_add = ir.node(
            "Add",
            inputs=[input_val, input_val],
            outputs=[ir.val("inner_then_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        inner_then_graph = ir.Graph(
            inputs=[],
            outputs=[inner_then_add.outputs[0]],
            nodes=[inner_then_add],
            name="inner_then",
        )

        # Inner else branch
        inner_else_identity = ir.node(
            "Identity",
            inputs=[input_val],
            outputs=[ir.val("inner_else_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        inner_else_graph = ir.Graph(
            inputs=[],
            outputs=[inner_else_identity.outputs[0]],
            nodes=[inner_else_identity],
            name="inner_else",
        )

        # Inner If node (in outer then branch)
        inner_if = ir.node(
            "If",
            inputs=[inner_cond],
            outputs=[ir.val("inner_result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": inner_then_graph,
                "else_branch": inner_else_graph,
            },
        )

        # Outer then branch containing inner If
        outer_then_graph = ir.Graph(
            inputs=[],
            outputs=[inner_if.outputs[0]],
            nodes=[inner_if],
            name="outer_then",
        )

        # Outer else branch
        outer_else_identity = ir.node(
            "Identity",
            inputs=[input_val],
            outputs=[ir.val("outer_else_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        outer_else_graph = ir.Graph(
            inputs=[],
            outputs=[outer_else_identity.outputs[0]],
            nodes=[outer_else_identity],
            name="outer_else",
        )

        # Outer If node
        outer_if = ir.node(
            "If",
            inputs=[outer_cond],
            outputs=[ir.val("final", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": outer_then_graph,
                "else_branch": outer_else_graph,
            },
        )

        graph = ir.Graph(
            inputs=[input_val, outer_cond, inner_cond],
            outputs=[outer_if.outputs[0]],
            nodes=[outer_if],
            name="nested_if_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=["input", "outer_cond", "inner_cond"],
            outputs=["final"],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].op_type, "If")


class ExtractBoundaryValidationTest(unittest.TestCase):
    """Test that subgraph extraction validates proper boundaries."""

    def test_extract_unbounded_subgraph_missing_input(self):
        """Test that extraction raises error when required graph input is not specified.

        Graph structure:
            input1, input2 (graph inputs)
              |      |
              |      |
              v      v
              Add -> output

        Extract with only input1 specified should fail because input2 is needed.
        """
        input1 = ir.val("input1", dtype=ir.DataType.FLOAT, shape=[3])
        input2 = ir.val("input2", dtype=ir.DataType.FLOAT, shape=[3])

        add_node = ir.node(
            "Add",
            inputs=[input1, input2],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="two_input_graph",
        )

        # Try to extract with only input1 - should fail
        with self.assertRaisesRegex(
            ValueError,
            r"The subgraph is not properly bounded.*graph inputs are required but not provided.*input2",
        ):
            _extractor.extract(
                graph,
                inputs=["input1"],
                outputs=["output"],
            )

    def test_extract_bounded_subgraph_with_initializer(self):
        """Test that extraction succeeds when missing inputs are initializers.

        Graph structure:
            input1 (graph input), const1 (initializer)
              |      |
              |      |
              v      v
              Add -> output

        Extract with only input1 specified should succeed because const1 is an initializer.
        """
        input1 = ir.val("input1", dtype=ir.DataType.FLOAT, shape=[3])
        const1 = ir.val(
            "const1",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="const1"),
        )

        add_node = ir.node(
            "Add",
            inputs=[input1, const1],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input1],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            initializers=[const1],
            name="input_and_const_graph",
        )

        # This should succeed - const1 is an initializer, not a required input
        extracted = _extractor.extract(
            graph,
            inputs=["input1"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].op_type, "Add")
        # const1 should be in the initializers
        self.assertIn("const1", extracted.initializers)

    def test_extract_unbounded_chain_missing_intermediate(self):
        """Test that extraction fails when intermediate input is missing.

        Graph structure:
            input1 (graph input)
              |
              v
            Add1 -> intermediate
              |
              v
            Add2 (uses intermediate + input2)
              |
              v
            output

        Extract from input1 to output should fail if input2 is not specified and is not an initializer.
        """
        input1 = ir.val("input1", dtype=ir.DataType.FLOAT, shape=[3])
        input2 = ir.val("input2", dtype=ir.DataType.FLOAT, shape=[3])
        const1 = ir.val(
            "const1",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="const1"),
        )

        add1_node = ir.node(
            "Add",
            inputs=[input1, const1],
            outputs=[ir.val("intermediate", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        add2_node = ir.node(
            "Add",
            inputs=[add1_node.outputs[0], input2],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[add2_node.outputs[0]],
            nodes=[add1_node, add2_node],
            initializers=[const1],
            name="chain_graph",
        )

        # Try to extract without specifying input2 - should fail
        with self.assertRaisesRegex(
            ValueError,
            r"The subgraph is not properly bounded.*graph inputs are required but not provided.*input2",
        ):
            _extractor.extract(
                graph,
                inputs=["input1"],
                outputs=["output"],
            )

    def test_extract_bounded_all_inputs_specified(self):
        r"""Test that extraction succeeds when all required inputs are specified.

        Graph structure:
            input1, input2, input3 (graph inputs)
              |      |      |
              v      v      v
            Add1   Add2
              \      /
               \    /
                Mul
                 |
                 v
               output

        Extract with all inputs specified should succeed.
        """
        input1 = ir.val("input1", dtype=ir.DataType.FLOAT, shape=[3])
        input2 = ir.val("input2", dtype=ir.DataType.FLOAT, shape=[3])
        input3 = ir.val("input3", dtype=ir.DataType.FLOAT, shape=[3])

        add1_node = ir.node(
            "Add",
            inputs=[input1, input2],
            outputs=[ir.val("intermediate1", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        add2_node = ir.node(
            "Add",
            inputs=[add1_node.outputs[0], input3],
            outputs=[ir.val("intermediate2", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        mul_node = ir.node(
            "Mul",
            inputs=[add2_node.outputs[0], input1],  # Reuses input1
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input1, input2, input3],
            outputs=[mul_node.outputs[0]],
            nodes=[add1_node, add2_node, mul_node],
            name="complex_input_graph",
        )

        # All inputs specified - should succeed
        extracted = _extractor.extract(
            graph,
            inputs=["input1", "input2", "input3"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 3)

    def test_extract_bounded_with_unused_input_specified(self):
        """Test that extraction succeeds even if extra inputs are specified.

        Graph structure:
            input1, input2 (graph inputs)
              |
              v
            Identity -> output
            (only uses input1)

        Extract with both inputs specified should succeed (input2 is just unused).
        """
        input1 = ir.val("input1", dtype=ir.DataType.FLOAT, shape=[3])
        input2 = ir.val("input2", dtype=ir.DataType.FLOAT, shape=[3])

        identity_node = ir.node(
            "Identity",
            inputs=[input1],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input1, input2],
            outputs=[identity_node.outputs[0]],
            nodes=[identity_node],
            name="unused_input_graph",
        )

        # Specify both inputs even though input2 is not used
        extracted = _extractor.extract(
            graph,
            inputs=["input1", "input2"],
            outputs=["output"],
        )

        self.assertEqual(len(extracted), 1)
        # Only input1 should be in the extracted graph inputs
        self.assertEqual(len(extracted.inputs), 2)

    def test_extract_unbounded_multiple_missing_inputs(self):
        r"""Test error message includes all missing graph inputs.

        Graph structure:
            input1, input2, input3 (all graph inputs)
              |      |      |
              v      v      v
             Add1   Add2
               \    /
                Mul
                 |
                 v
               output

        Extract with no inputs specified should list all missing inputs.
        """
        input1 = ir.val("input1", dtype=ir.DataType.FLOAT, shape=[3])
        input2 = ir.val("input2", dtype=ir.DataType.FLOAT, shape=[3])
        input3 = ir.val("input3", dtype=ir.DataType.FLOAT, shape=[3])

        add1_node = ir.node(
            "Add",
            inputs=[input1, input2],
            outputs=[ir.val("intermediate", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        add2_node = ir.node(
            "Add",
            inputs=[add1_node.outputs[0], input3],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[input1, input2, input3],
            outputs=[add2_node.outputs[0]],
            nodes=[add1_node, add2_node],
            name="multi_missing_graph",
        )

        # Try to extract with no inputs - should fail and list all three
        with self.assertRaisesRegex(
            ValueError,
            r"The subgraph is not properly bounded.*graph inputs are required but not provided",
        ) as cm:
            _extractor.extract(
                graph,
                inputs=[],
                outputs=["output"],
            )

        # Verify all three inputs are mentioned in the error
        error_msg = str(cm.exception)
        self.assertIn("input1", error_msg)
        self.assertIn("input2", error_msg)
        self.assertIn("input3", error_msg)

    def test_extract_partial_bounded_intermediate_as_input(self):
        """Test that extraction works when using intermediate value as input boundary.

        Graph structure:
            actual_input (graph input)
              |
              v
            Add1 -> intermediate1
              |
              v
            Add2 -> intermediate2
              |
              v
            output

        Extract from intermediate1 to output should succeed (properly bounded).
        """
        actual_input = ir.val("actual_input", dtype=ir.DataType.FLOAT, shape=[3])
        const1 = ir.val(
            "const1",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([1, 2, 3], dtype=np.float32), name="const1"),
        )
        const2 = ir.val(
            "const2",
            dtype=ir.DataType.FLOAT,
            shape=[3],
            const_value=ir.tensor(np.array([4, 5, 6], dtype=np.float32), name="const2"),
        )

        add1_node = ir.node(
            "Add",
            inputs=[actual_input, const1],
            outputs=[ir.val("intermediate1", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        add2_node = ir.node(
            "Add",
            inputs=[add1_node.outputs[0], const2],
            outputs=[ir.val("intermediate2", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        add3_node = ir.node(
            "Add",
            inputs=[add2_node.outputs[0], const1],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        graph = ir.Graph(
            inputs=[actual_input],
            outputs=[add3_node.outputs[0]],
            nodes=[add1_node, add2_node, add3_node],
            initializers=[const1, const2],
            name="chain_partial_graph",
        )

        # Extract from intermediate1 (produced value) to output - should work
        extracted = _extractor.extract(
            graph,
            inputs=["intermediate1"],
            outputs=["output"],
        )

        # Should only include add2_node and add3_node
        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted.inputs[0].name, "intermediate1")


if __name__ == "__main__":
    unittest.main()
