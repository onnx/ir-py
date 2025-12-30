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
            name="test_graph",
        )

    def test_extract_full_graph(self):
        """Test extracting the entire graph."""
        extracted = _extractor.extract(
            self.graph,
            inputs=["input"],
            outputs=["output"],
        )

        self.assertEqual(len(list(extracted)), 2)
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

        self.assertEqual(len(list(extracted)), 1)
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

        self.assertEqual(len(list(extracted)), 1)
        self.assertEqual(extracted.inputs[0].name, "intermediate")
        self.assertEqual(extracted.outputs[0].name, "output")

    def test_extract_with_mixed_inputs(self):
        """Test extracting using a mix of Value objects and strings."""
        extracted = _extractor.extract(
            self.graph,
            inputs=[self.input_val],
            outputs=["output"],
        )

        self.assertEqual(len(list(extracted)), 2)
        self.assertEqual(extracted.inputs[0].name, "input")
        self.assertEqual(extracted.outputs[0].name, "output")

    def test_extract_preserves_node_order(self):
        """Test that extraction preserves the original node order."""
        extracted = _extractor.extract(
            self.graph,
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
        )

        nodes = list(extracted)
        self.assertEqual(nodes[0].op_type, "Add")
        self.assertEqual(nodes[1].op_type, "Mul")

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
        )

        extracted = _extractor.extract(
            function,
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
        )

        self.assertEqual(len(list(extracted)), 2)
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
            inputs=[self.add_node.outputs[0]],
            outputs=[self.mul_node.outputs[0]],
        )

        self.assertEqual(len(list(extracted)), 1)
        self.assertEqual(extracted.inputs[0].name, "intermediate")
        self.assertEqual(extracted.outputs[0].name, "output")

    def test_extract_preserves_metadata(self):
        """Test that extraction preserves graph metadata."""
        graph_with_metadata = ir.Graph(
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
            nodes=[self.add_node, self.mul_node],
            name="metadata_graph",
            doc_string="Test documentation",
            opset_imports={"": 18},
            metadata_props={"key": "value"},
        )

        extracted = _extractor.extract(
            graph_with_metadata,
            inputs=[self.input_val],
            outputs=[self.mul_node.outputs[0]],
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
        nodes = list(extracted)
        self.assertEqual(len(nodes), 3)
        node_names = {node.name for node in nodes}
        self.assertEqual(node_names, {"add1", "add2", "mul"})

    def test_extract_single_branch(self):
        """Test extracting a single branch of the diamond."""
        extracted = _extractor.extract(
            self.graph,
            inputs=[self.input_val],
            outputs=[self.add1_node.outputs[0]],
        )

        # Should only include add1
        nodes = list(extracted)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, "add1")
        # Should include const1 initializer
        self.assertEqual(len(extracted.initializers), 1)

    def test_extract_from_intermediate_to_output(self):
        """Test extracting from intermediate values to final output."""
        extracted = _extractor.extract(
            self.graph,
            inputs=[self.add1_node.outputs[0], self.add2_node.outputs[0]],
            outputs=[self.mul_node.outputs[0]],
        )

        # Should only include mul node
        nodes = list(extracted)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, "mul")
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

        self.assertEqual(len(list(extracted)), 0)
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
            name="multi_output_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=[input_val],
            outputs=[add_node.outputs[0], mul_node.outputs[0]],
        )

        self.assertEqual(len(list(extracted)), 2)
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
            name="single_node_graph",
        )

        extracted = _extractor.extract(
            graph,
            inputs=[input_val],
            outputs=[add_node.outputs[0]],
        )

        self.assertEqual(len(list(extracted)), 1)
        self.assertEqual(list(extracted)[0].op_type, "Add")


if __name__ == "__main__":
    unittest.main()
