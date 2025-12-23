# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir import _core, _graph_comparison


class TopologicallyEqualTest(unittest.TestCase):
    def test_empty_graphs_are_equal(self):
        """Test that two empty graphs are topologically equal."""
        graph1 = _core.Graph(inputs=(), outputs=(), nodes=())
        graph2 = _core.Graph(inputs=(), outputs=(), nodes=())
        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_different_number_of_nodes(self):
        """Test that graphs with different numbers of nodes are not equal."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        graph2 = _core.Graph((v2,), (v2,), nodes=())

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_different_number_of_inputs(self):
        """Test that graphs with different numbers of inputs are not equal."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        v3 = _core.Value(name="v3")
        node2 = _core.Node("", "Identity", inputs=(v2,), num_outputs=1)
        graph2 = _core.Graph((v2, v3), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_different_number_of_outputs(self):
        """Test that graphs with different numbers of outputs are not equal."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Identity", inputs=(v2,), num_outputs=2)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_simple_identical_graphs(self):
        """Test that two simple identical graphs are equal."""
        # Graph 1
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 - identical structure
        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_different_op_types(self):
        """Test that graphs with different op types are not equal."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Mul", inputs=(v2, v2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_different_domains(self):
        """Test that graphs with different domains are not equal."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        node2 = _core.Node("custom.domain", "Add", inputs=(v2, v2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_linear_chain_equal(self):
        """Test that two linear chains of nodes are equal."""
        # Graph 1: v1 -> node1 -> node2 -> out
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        node2 = _core.Node(
            "", "Mul", inputs=(node1.outputs[0], node1.outputs[0]), num_outputs=1
        )
        graph1 = _core.Graph((v1,), node2.outputs, nodes=(node1, node2))

        # Graph 2: v2 -> node3 -> node4 -> out
        v2 = _core.Value(name="v2")
        node3 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1)
        node4 = _core.Node(
            "", "Mul", inputs=(node3.outputs[0], node3.outputs[0]), num_outputs=1
        )
        graph2 = _core.Graph((v2,), node4.outputs, nodes=(node3, node4))

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_different_connectivity(self):
        """Test that graphs with different connectivity are not equal."""
        # Graph 1: v1 -> node1, v1 -> node2, node1 + node2 -> node3
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        node2 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        node3 = _core.Node(
            "", "Add", inputs=(node1.outputs[0], node2.outputs[0]), num_outputs=1
        )
        graph1 = _core.Graph((v1,), node3.outputs, nodes=(node1, node2, node3))

        # Graph 2: v2 -> node4, node4 -> node5, node4 + node5 -> node6 (different structure)
        v2 = _core.Value(name="v2")
        node4 = _core.Node("", "Identity", inputs=(v2,), num_outputs=1)
        node5 = _core.Node("", "Identity", inputs=(node4.outputs[0],), num_outputs=1)
        node6 = _core.Node(
            "", "Add", inputs=(node4.outputs[0], node5.outputs[0]), num_outputs=1
        )
        graph2 = _core.Graph((v2,), node6.outputs, nodes=(node4, node5, node6))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_attributes_equal(self):
        """Test that graphs with matching attributes are equal."""
        # Graph 1
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("axis", ir.AttributeType.INT, 1)
        node1 = _core.Node("", "Squeeze", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 - same attributes
        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("axis", ir.AttributeType.INT, 1)
        node2 = _core.Node("", "Squeeze", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_different_attribute_names(self):
        """Test that graphs with different attribute names are not equal."""
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("axis", ir.AttributeType.INT, 1)
        node1 = _core.Node("", "Op", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("dim", ir.AttributeType.INT, 1)
        node2 = _core.Node("", "Op", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_different_attribute_types(self):
        """Test that graphs with different attribute types are not equal."""
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("value", ir.AttributeType.INT, 1)
        node1 = _core.Node("", "Op", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("value", ir.AttributeType.FLOAT, 1.0)
        node2 = _core.Node("", "Op", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_without_initializers_by_default(self):
        """Test that initializer data comparison can be skipped with tensor_size_limit=0."""
        # Graph 1 with initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with different initializer data but same shape/dtype
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=ir.tensor(np.array([3.0, 4.0], dtype=np.float32))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        # Should be equal when using tensor_size_limit=0 (skip data comparison for tensors > size 0)
        self.assertTrue(
            _graph_comparison.topologically_equal(graph1, graph2, tensor_size_limit=0)
        )

        # Should NOT be equal with default (None) - data is compared
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_initializers_when_enabled(self):
        """Test that initializers data is compared when tensor_size_limit=None."""
        # Graph 1 with initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with same initializer shape and dtype
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        # Should be equal when comparing initializer data (tensor_size_limit=None means compare all)
        self.assertTrue(
            _graph_comparison.topologically_equal(graph1, graph2, tensor_size_limit=None)
        )

    def test_with_different_initializer_shapes(self):
        """Test that graphs with different initializer shapes are not equal."""
        # Graph 1 with initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with different initializer shape
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=ir.tensor(np.array([1.0], dtype=np.float32))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        # Should not be equal - shapes are always compared
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_different_initializer_dtypes(self):
        """Test that graphs with different initializer dtypes are not equal."""
        # Graph 1 with float32 initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with float64 initializer
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float64))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        # Should not be equal - dtypes are always compared
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_different_number_of_initializers(self):
        """Test that graphs with different numbers of initializers are not equal."""
        # Graph 1 with one initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=ir.tensor(np.array([1.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with no initializers
        v2 = _core.Value(name="v2")
        v3 = _core.Value(name="v3")
        node2 = _core.Node("", "Add", inputs=(v2, v3), num_outputs=1)
        graph2 = _core.Graph((v2, v3), node2.outputs, nodes=(node2,))

        # Should not be equal - different number of inputs (one has initializer, one doesn't)
        self.assertFalse(
            _graph_comparison.topologically_equal(
                graph1,
                graph2,
            )
        )

    def test_with_subgraph_attributes(self):
        """Test that graphs with subgraph attributes are compared recursively."""
        # Create a simple subgraph
        sub_input1 = _core.Value(name="sub_input1")
        sub_node1 = _core.Node("", "Identity", inputs=(sub_input1,), num_outputs=1)
        subgraph1 = _core.Graph((sub_input1,), sub_node1.outputs, nodes=(sub_node1,))

        # Graph 1 with subgraph attribute
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("body", ir.AttributeType.GRAPH, subgraph1)
        node1 = _core.Node("", "Loop", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Create an identical subgraph
        sub_input2 = _core.Value(name="sub_input2")
        sub_node2 = _core.Node("", "Identity", inputs=(sub_input2,), num_outputs=1)
        subgraph2 = _core.Graph((sub_input2,), sub_node2.outputs, nodes=(sub_node2,))

        # Graph 2 with identical subgraph attribute
        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("body", ir.AttributeType.GRAPH, subgraph2)
        node2 = _core.Node("", "Loop", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_different_subgraph_structures(self):
        """Test that graphs with different subgraph structures are not equal."""
        # Create a simple subgraph
        sub_input1 = _core.Value(name="sub_input1")
        sub_node1 = _core.Node("", "Identity", inputs=(sub_input1,), num_outputs=1)
        subgraph1 = _core.Graph((sub_input1,), sub_node1.outputs, nodes=(sub_node1,))

        # Graph 1 with subgraph attribute
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("body", ir.AttributeType.GRAPH, subgraph1)
        node1 = _core.Node("", "Loop", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Create a different subgraph (different op type)
        sub_input2 = _core.Value(name="sub_input2")
        sub_node2 = _core.Node("", "Relu", inputs=(sub_input2,), num_outputs=1)
        subgraph2 = _core.Graph((sub_input2,), sub_node2.outputs, nodes=(sub_node2,))

        # Graph 2 with different subgraph
        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("body", ir.AttributeType.GRAPH, subgraph2)
        node2 = _core.Node("", "Loop", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_multiple_subgraphs(self):
        """Test that graphs with multiple subgraph attributes are compared correctly."""
        # Create two subgraphs for graph1
        sub_input1_a = _core.Value(name="sub_input1_a")
        sub_node1_a = _core.Node("", "Identity", inputs=(sub_input1_a,), num_outputs=1)
        subgraph1_a = _core.Graph((sub_input1_a,), sub_node1_a.outputs, nodes=(sub_node1_a,))

        sub_input1_b = _core.Value(name="sub_input1_b")
        sub_node1_b = _core.Node("", "Relu", inputs=(sub_input1_b,), num_outputs=1)
        subgraph1_b = _core.Graph((sub_input1_b,), sub_node1_b.outputs, nodes=(sub_node1_b,))

        # Graph 1 with GRAPHS attribute
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("branches", ir.AttributeType.GRAPHS, [subgraph1_a, subgraph1_b])
        node1 = _core.Node("", "If", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Create identical subgraphs for graph2
        sub_input2_a = _core.Value(name="sub_input2_a")
        sub_node2_a = _core.Node("", "Identity", inputs=(sub_input2_a,), num_outputs=1)
        subgraph2_a = _core.Graph((sub_input2_a,), sub_node2_a.outputs, nodes=(sub_node2_a,))

        sub_input2_b = _core.Value(name="sub_input2_b")
        sub_node2_b = _core.Node("", "Relu", inputs=(sub_input2_b,), num_outputs=1)
        subgraph2_b = _core.Graph((sub_input2_b,), sub_node2_b.outputs, nodes=(sub_node2_b,))

        # Graph 2 with identical GRAPHS attribute
        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("branches", ir.AttributeType.GRAPHS, [subgraph2_a, subgraph2_b])
        node2 = _core.Node("", "If", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))


class AssertTopologicallyEqualTest(unittest.TestCase):
    def test_empty_graphs_do_not_raise(self):
        """Test that two empty graphs do not raise an assertion error."""
        graph1 = _core.Graph(inputs=(), outputs=(), nodes=())
        graph2 = _core.Graph(inputs=(), outputs=(), nodes=())
        # Should not raise
        _graph_comparison.assert_topologically_equal(graph1, graph2)

    def test_identical_graphs_do_not_raise(self):
        """Test that identical graphs do not raise an assertion error."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1, name="add_node")
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1, name="add_node")
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        # Should not raise
        _graph_comparison.assert_topologically_equal(graph1, graph2)

    def test_different_number_of_nodes_raises(self):
        """Test that graphs with different numbers of nodes raise an error."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        graph2 = _core.Graph((v2,), (v2,), nodes=())

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        # With backward traversal, the error manifests as one value being an input
        # when the other is not (since graph1 has a node producing the output,
        # but graph2's output is directly an input)
        self.assertIn("One value is a graph input, the other is not", error_msg)

    def test_different_op_types_raises_with_node_name(self):
        """Test that graphs with different op types raise an error with node name."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1, name="my_add")
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Mul", inputs=(v2, v2), num_outputs=1, name="my_mul")
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("name='my_add'", error_msg)
        self.assertIn("name='my_mul'", error_msg)
        self.assertIn("Different op_type", error_msg)

    def test_different_domains_raises_with_node_name(self):
        """Test that graphs with different domains raise an error with node name."""
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1, name="standard_add")
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        node2 = _core.Node(
            "custom.domain", "Add", inputs=(v2, v2), num_outputs=1, name="custom_add"
        )
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("name='standard_add'", error_msg)
        self.assertIn("Different domain", error_msg)

    def test_different_number_of_graph_inputs_raises(self):
        """Test that different number of graph inputs raises an error."""
        v1 = _core.Value(name="v1")
        v2 = _core.Value(name="v2")
        node1 = _core.Node("", "Add", inputs=(v1, v2), num_outputs=1, name="add_two_inputs")
        graph1 = _core.Graph((v1, v2), node1.outputs, nodes=(node1,))

        v3 = _core.Value(name="v3")
        node2 = _core.Node(
            "", "Identity", inputs=(v3,), num_outputs=1, name="identity_one_input"
        )
        graph2 = _core.Graph((v3,), node2.outputs, nodes=(node2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("Different number of inputs", error_msg)
        self.assertIn("2 vs 1", error_msg)

    def test_different_number_of_node_inputs_raises_with_node_name(self):
        """Test that different number of node inputs raises an error with node name."""
        # Both graphs have same number of graph inputs, but nodes take different number
        v1 = _core.Value(name="v1")
        v2 = _core.Value(name="v2")
        node1 = _core.Node("", "Add", inputs=(v1, v2), num_outputs=1, name="add_two_inputs")
        graph1 = _core.Graph((v1, v2), node1.outputs, nodes=(node1,))

        v3 = _core.Value(name="v3")
        v4 = _core.Value(name="v4")
        node2 = _core.Node("", "Add", inputs=(v3,), num_outputs=1, name="add_one_input")
        graph2 = _core.Graph((v3, v4), node2.outputs, nodes=(node2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("name='add_two_inputs'", error_msg)
        self.assertIn("Different number of inputs", error_msg)

    def test_different_attribute_values_raises_with_node_name(self):
        """Test that different attribute values raise an error with node name."""
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("axis", ir.AttributeType.INT, 1)
        node1 = _core.Node(
            "",
            "Squeeze",
            inputs=(v1,),
            attributes=(attr1,),
            num_outputs=1,
            name="squeeze_axis_1",
        )
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("axis", ir.AttributeType.INT, 2)
        node2 = _core.Node(
            "",
            "Squeeze",
            inputs=(v2,),
            attributes=(attr2,),
            num_outputs=1,
            name="squeeze_axis_2",
        )
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("name='squeeze_axis_1'", error_msg)
        self.assertIn("attribute 'axis'", error_msg)
        self.assertIn("Value mismatch", error_msg)

    def test_different_initializer_shapes_raises_with_node_name(self):
        """Test that different initializer shapes raise an error with node name."""
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1, name="add_with_init")
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=ir.tensor(np.array([1.0], dtype=np.float32))
        )
        node2 = _core.Node(
            "", "Add", inputs=(v2, init2), num_outputs=1, name="add_with_different_init"
        )
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("name='add_with_init'", error_msg)
        self.assertIn("Initializer shape mismatch", error_msg)

    def test_tensor_data_difference_continues_checking(self):
        """Test that tensor data differences are collected but checking continues."""
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1, name="add_node")
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=ir.tensor(np.array([3.0, 4.0], dtype=np.float32))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1, name="add_node")
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("Tensor data differences found", error_msg)
        self.assertIn("name='add_node'", error_msg)
        self.assertIn("initializer data differs", error_msg)

    def test_multiple_errors_reported_together(self):
        """Test that multiple errors are collected and reported together."""
        # Create graph with multiple differences
        v1 = _core.Value(name="v1")
        init1a = _core.Value(
            name="init1a", const_value=ir.tensor(np.array([1.0], dtype=np.float32))
        )
        init1b = _core.Value(
            name="init1b", const_value=ir.tensor(np.array([2.0], dtype=np.float32))
        )
        node1a = _core.Node("", "Add", inputs=(v1, init1a), num_outputs=1, name="node_a")
        node1b = _core.Node(
            "", "Mul", inputs=(node1a.outputs[0], init1b), num_outputs=1, name="node_b"
        )
        graph1 = _core.Graph(
            (v1,), node1b.outputs, nodes=(node1a, node1b), initializers=(init1a, init1b)
        )

        # Graph 2 with different tensor data in both nodes
        v2 = _core.Value(name="v2")
        init2a = _core.Value(
            name="init2a", const_value=ir.tensor(np.array([10.0], dtype=np.float32))
        )
        init2b = _core.Value(
            name="init2b", const_value=ir.tensor(np.array([20.0], dtype=np.float32))
        )
        node2a = _core.Node("", "Add", inputs=(v2, init2a), num_outputs=1, name="node_a")
        node2b = _core.Node(
            "", "Mul", inputs=(node2a.outputs[0], init2b), num_outputs=1, name="node_b"
        )
        graph2 = _core.Graph(
            (v2,), node2b.outputs, nodes=(node2a, node2b), initializers=(init2a, init2b)
        )

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        # Should have both tensor data differences
        self.assertIn("Tensor data differences found", error_msg)
        # Count occurrences - should have 2 tensor data differences
        self.assertEqual(error_msg.count("initializer data differs"), 2)

    def test_subgraph_difference_includes_context(self):
        """Test that subgraph differences include context information."""
        # Create a subgraph with Identity
        sub_input1 = _core.Value(name="sub_input1")
        sub_node1 = _core.Node(
            "", "Identity", inputs=(sub_input1,), num_outputs=1, name="sub_identity"
        )
        subgraph1 = _core.Graph((sub_input1,), sub_node1.outputs, nodes=(sub_node1,))

        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("body", ir.AttributeType.GRAPH, subgraph1)
        node1 = _core.Node(
            "", "Loop", inputs=(v1,), attributes=(attr1,), num_outputs=1, name="loop_node"
        )
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Create a different subgraph with Relu
        sub_input2 = _core.Value(name="sub_input2")
        sub_node2 = _core.Node(
            "", "Relu", inputs=(sub_input2,), num_outputs=1, name="sub_relu"
        )
        subgraph2 = _core.Graph((sub_input2,), sub_node2.outputs, nodes=(sub_node2,))

        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("body", ir.AttributeType.GRAPH, subgraph2)
        node2 = _core.Node(
            "", "Loop", inputs=(v2,), attributes=(attr2,), num_outputs=1, name="loop_node"
        )
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("name='loop_node'", error_msg)
        self.assertIn("attribute 'body' subgraph", error_msg)

    def test_tensor_size_limit_skips_data_comparison(self):
        """Test that tensor_size_limit=0 skips data comparison and does not raise."""
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1, name="add_node")
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=ir.tensor(np.array([3.0, 4.0], dtype=np.float32))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1, name="add_node")
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        # Should not raise with tensor_size_limit=0
        _graph_comparison.assert_topologically_equal(graph1, graph2, tensor_size_limit=0)

    def test_tensor_attribute_difference_includes_node_name(self):
        """Test that tensor attribute differences include node name."""
        tensor_attr1 = _core.Attr(
            "value",
            ir.AttributeType.TENSOR,
            ir.tensor(np.array([1.0], dtype=np.float32)),
        )
        node1 = _core.Node(
            "",
            "Constant",
            inputs=(),
            attributes=(tensor_attr1,),
            num_outputs=1,
            name="const_1",
        )
        graph1 = _core.Graph((), node1.outputs, nodes=(node1,))

        # Different shape
        tensor_attr2 = _core.Attr(
            "value",
            ir.AttributeType.TENSOR,
            ir.tensor(np.array([1.0, 2.0], dtype=np.float32)),
        )
        node2 = _core.Node(
            "",
            "Constant",
            inputs=(),
            attributes=(tensor_attr2,),
            num_outputs=1,
            name="const_2",
        )
        graph2 = _core.Graph((), node2.outputs, nodes=(node2,))

        with self.assertRaises(AssertionError) as cm:
            _graph_comparison.assert_topologically_equal(graph1, graph2)

        error_msg = str(cm.exception)
        self.assertIn("name='const_1'", error_msg)
        self.assertIn("Tensor shape mismatch", error_msg)


class NodeOrderingTest(unittest.TestCase):
    """Tests to ensure that node ordering does not affect topological equivalence."""

    def test_simple_chain_different_order(self):
        """Test that a simple chain with nodes in different order is still equal."""
        # Graph 1: Add then Mul (in order)
        v1 = _core.Value(name="v1")
        v2 = _core.Value(name="v2")
        add_node1 = _core.Node("", "Add", inputs=(v1, v2), num_outputs=1)
        mul_node1 = _core.Node("", "Mul", inputs=(add_node1.outputs[0], v2), num_outputs=1)
        graph1 = _core.Graph((v1, v2), mul_node1.outputs, nodes=(add_node1, mul_node1))

        # Graph 2: Mul then Add (reversed order)
        v3 = _core.Value(name="v3")
        v4 = _core.Value(name="v4")
        mul_node2 = _core.Node("", "Mul", inputs=(None, v4), num_outputs=1)
        add_node2 = _core.Node("", "Add", inputs=(v3, v4), num_outputs=1)
        mul_node2._inputs = (add_node2.outputs[0], v4)
        graph2 = _core.Graph((v3, v4), mul_node2.outputs, nodes=(mul_node2, add_node2))

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_diamond_pattern_different_order(self):
        """Test diamond pattern (one input, two branches, merge) with different node order."""
        # Graph 1: Linear order (input -> add -> mul -> concat)
        v1 = _core.Value(name="v1")
        add_node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        mul_node1 = _core.Node("", "Mul", inputs=(v1, v1), num_outputs=1)
        concat_node1 = _core.Node(
            "", "Concat", inputs=(add_node1.outputs[0], mul_node1.outputs[0]), num_outputs=1
        )
        graph1 = _core.Graph(
            (v1,), concat_node1.outputs, nodes=(add_node1, mul_node1, concat_node1)
        )

        # Graph 2: Different order (input -> mul -> concat -> add)
        v2 = _core.Value(name="v2")
        mul_node2 = _core.Node("", "Mul", inputs=(v2, v2), num_outputs=1)
        concat_node2 = _core.Node(
            "", "Concat", inputs=(None, mul_node2.outputs[0]), num_outputs=1
        )
        add_node2 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1)
        concat_node2._inputs = (add_node2.outputs[0], mul_node2.outputs[0])
        graph2 = _core.Graph(
            (v2,), concat_node2.outputs, nodes=(mul_node2, concat_node2, add_node2)
        )

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_complex_dag_different_order(self):
        """Test a more complex DAG with completely different node ordering."""
        # Graph 1: Create a DAG: v1 -> add1 -> mul1 \
        #                         v2 -> add2 -> mul2 -> sub -> output
        v1 = _core.Value(name="v1")
        v2 = _core.Value(name="v2")
        add1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        add2 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1)
        mul1 = _core.Node("", "Mul", inputs=(add1.outputs[0], v1), num_outputs=1)
        mul2 = _core.Node("", "Mul", inputs=(add2.outputs[0], v2), num_outputs=1)
        sub = _core.Node("", "Sub", inputs=(mul1.outputs[0], mul2.outputs[0]), num_outputs=1)
        graph1 = _core.Graph((v1, v2), sub.outputs, nodes=(add1, add2, mul1, mul2, sub))

        # Graph 2: Same structure but completely different order
        v3 = _core.Value(name="v3")
        v4 = _core.Value(name="v4")
        # Create in reverse order
        sub2 = _core.Node("", "Sub", inputs=(None, None), num_outputs=1)
        mul2_2 = _core.Node("", "Mul", inputs=(None, v4), num_outputs=1)
        mul1_2 = _core.Node("", "Mul", inputs=(None, v3), num_outputs=1)
        add2_2 = _core.Node("", "Add", inputs=(v4, v4), num_outputs=1)
        add1_2 = _core.Node("", "Add", inputs=(v3, v3), num_outputs=1)
        # Fix connections
        mul1_2._inputs = (add1_2.outputs[0], v3)
        mul2_2._inputs = (add2_2.outputs[0], v4)
        sub2._inputs = (mul1_2.outputs[0], mul2_2.outputs[0])
        graph2 = _core.Graph(
            (v3, v4), sub2.outputs, nodes=(sub2, mul2_2, mul1_2, add2_2, add1_2)
        )

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_different_order_with_initializers(self):
        """Test that graphs with initializers and different node order are equal."""
        # Graph 1: Add with initializer, then Mul
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="weight1", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        add_node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        mul_node1 = _core.Node("", "Mul", inputs=(add_node1.outputs[0], v1), num_outputs=1)
        graph1 = _core.Graph(
            (v1,), mul_node1.outputs, nodes=(add_node1, mul_node1), initializers=(init1,)
        )

        # Graph 2: Mul then Add (reversed order)
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="weight2", const_value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        mul_node2 = _core.Node("", "Mul", inputs=(None, v2), num_outputs=1)
        add_node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1)
        mul_node2._inputs = (add_node2.outputs[0], v2)
        graph2 = _core.Graph(
            (v2,), mul_node2.outputs, nodes=(mul_node2, add_node2), initializers=(init2,)
        )

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_multiple_outputs_different_order(self):
        """Test graphs with multiple outputs and different node ordering."""
        # Graph 1: Two parallel branches with different ops
        v1 = _core.Value(name="v1")
        add_node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        mul_node1 = _core.Node("", "Mul", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph(
            (v1,), (add_node1.outputs[0], mul_node1.outputs[0]), nodes=(add_node1, mul_node1)
        )

        # Graph 2: Same structure but reversed node order
        v2 = _core.Value(name="v2")
        mul_node2 = _core.Node("", "Mul", inputs=(v2, v2), num_outputs=1)
        add_node2 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1)
        graph2 = _core.Graph(
            (v2,), (add_node2.outputs[0], mul_node2.outputs[0]), nodes=(mul_node2, add_node2)
        )

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_assert_equal_with_different_order(self):
        """Test that assert_topologically_equal doesn't raise for different node orders."""
        # Graph 1: Sequential order
        v1 = _core.Value(name="v1")
        add1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        relu1 = _core.Node("", "Relu", inputs=(add1.outputs[0],), num_outputs=1)
        graph1 = _core.Graph((v1,), relu1.outputs, nodes=(add1, relu1))

        # Graph 2: Reversed order
        v2 = _core.Value(name="v2")
        relu2 = _core.Node("", "Relu", inputs=(None,), num_outputs=1)
        add2 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1)
        relu2._inputs = (add2.outputs[0],)
        graph2 = _core.Graph((v2,), relu2.outputs, nodes=(relu2, add2))

        # Should not raise
        _graph_comparison.assert_topologically_equal(graph1, graph2)


class ValidationFeaturesTest(unittest.TestCase):
    """Test validation features added in commit 8d60d40."""

    def test_input_type_mismatch_detected(self):
        """Test that mismatched input types are detected."""
        # Graph 1 with input of type INT32
        v1 = _core.Value(name="v1")
        v1._type = ir.TensorType(ir.DataType.INT32)
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 with input of type FLOAT
        v2 = _core.Value(name="v2")
        v2._type = ir.TensorType(ir.DataType.FLOAT)
        node2 = _core.Node("", "Identity", inputs=(v2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        # Should detect type mismatch
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_input_shape_mismatch_detected(self):
        """Test that mismatched input shapes are detected."""
        # Graph 1 with input of shape (2, 3)
        v1 = _core.Value(name="v1")
        v1._type = ir.TensorType(ir.DataType.FLOAT)
        v1._shape = (2, 3)
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 with input of shape (3, 2)
        v2 = _core.Value(name="v2")
        v2._type = ir.TensorType(ir.DataType.FLOAT)
        v2._shape = (3, 2)
        node2 = _core.Node("", "Identity", inputs=(v2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        # Should detect shape mismatch
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_value_type_mismatch_detected(self):
        """Test that type mismatches in intermediate values are detected."""
        # Graph 1 - Add produces INT32
        v1 = _core.Value(name="v1")
        v1._type = ir.TensorType(ir.DataType.INT32)
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        # Manually set output type to INT32
        node1.outputs[0]._type = ir.TensorType(ir.DataType.INT32)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 - Add produces FLOAT
        v2 = _core.Value(name="v2")
        v2._type = ir.TensorType(ir.DataType.INT32)
        node2 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1)
        # Manually set output type to FLOAT
        node2.outputs[0]._type = ir.TensorType(ir.DataType.FLOAT)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        # Should detect type mismatch in intermediate value
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_value_shape_mismatch_detected(self):
        """Test that shape mismatches in intermediate values are detected."""
        # Graph 1 - produces shape (2,)
        v1 = _core.Value(name="v1")
        v1._type = ir.TensorType(ir.DataType.FLOAT)
        v1._shape = (2,)
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        node1.outputs[0]._shape = (2,)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 - produces shape (3,)
        v2 = _core.Value(name="v2")
        v2._type = ir.TensorType(ir.DataType.FLOAT)
        v2._shape = (3,)
        node2 = _core.Node("", "Identity", inputs=(v2,), num_outputs=1)
        node2.outputs[0]._shape = (3,)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        # Should detect shape mismatch
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_output_index_mismatch_detected(self):
        """Test that output index mismatches are detected."""
        # Graph 1 - node with 2 outputs, use first output
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Op", inputs=(v1,), num_outputs=2)
        graph1 = _core.Graph((v1,), (node1.outputs[0],), nodes=(node1,))

        # Graph 2 - node with 2 outputs, use second output
        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Op", inputs=(v2,), num_outputs=2)
        graph2 = _core.Graph((v2,), (node2.outputs[1],), nodes=(node2,))

        # Should detect output index mismatch
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_output_index_match_accepted(self):
        """Test that matching output indices are accepted."""
        # Graph 1 - node with 2 outputs, use second output
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Op", inputs=(v1,), num_outputs=2)
        graph1 = _core.Graph((v1,), (node1.outputs[1],), nodes=(node1,))

        # Graph 2 - node with 2 outputs, use second output
        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Op", inputs=(v2,), num_outputs=2)
        graph2 = _core.Graph((v2,), (node2.outputs[1],), nodes=(node2,))

        # Should be equal - both use second output
        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_unused_nodes_detected_in_graph1(self):
        """Test that unused nodes in graph1 are detected."""
        # Graph 1 with an unused node
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        unused_node = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1, unused_node))

        # Graph 2 without unused node
        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Identity", inputs=(v2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        # Should detect unused node
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_unused_nodes_detected_in_graph2(self):
        """Test that unused nodes in graph2 are detected."""
        # Graph 1 without unused node
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 with an unused node
        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Identity", inputs=(v2,), num_outputs=1)
        unused_node = _core.Node("", "Mul", inputs=(v2, v2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2, unused_node))

        # Should detect unused node
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_unused_nodes_detected_in_both_graphs(self):
        """Test that unused nodes in both graphs are detected."""
        # Graph 1 with unused node
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Identity", inputs=(v1,), num_outputs=1)
        unused1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1, unused1))

        # Graph 2 with different unused node
        v2 = _core.Value(name="v2")
        node2 = _core.Node("", "Identity", inputs=(v2,), num_outputs=1)
        unused2 = _core.Node("", "Mul", inputs=(v2, v2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2, unused2))

        # Should detect unused nodes in both
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_all_nodes_visited_when_equal(self):
        """Test that all nodes are visited when graphs are equal."""
        # Graph 1 with chain of operations
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        node2 = _core.Node("", "Mul", inputs=(node1.outputs[0], node1.outputs[0]), num_outputs=1)
        node3 = _core.Node("", "Relu", inputs=(node2.outputs[0],), num_outputs=1)
        graph1 = _core.Graph((v1,), node3.outputs, nodes=(node1, node2, node3))

        # Graph 2 with same structure
        v2 = _core.Value(name="v2")
        node4 = _core.Node("", "Add", inputs=(v2, v2), num_outputs=1)
        node5 = _core.Node("", "Mul", inputs=(node4.outputs[0], node4.outputs[0]), num_outputs=1)
        node6 = _core.Node("", "Relu", inputs=(node5.outputs[0],), num_outputs=1)
        graph2 = _core.Graph((v2,), node6.outputs, nodes=(node4, node5, node6))

        # Should be equal - all nodes visited
        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_error_messages_include_value_names(self):
        """Test that error messages include value names for clarity."""
        # Graph 1
        v1 = _core.Value(name="input_a")
        node1 = _core.Node("", "Add", inputs=(v1, v1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 with different op
        v2 = _core.Value(name="input_b")
        node2 = _core.Node("", "Mul", inputs=(v2, v2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        # Get error message using assert function
        try:
            _graph_comparison.assert_topologically_equal(graph1, graph2)
            self.fail("Should have raised AssertionError")
        except AssertionError as e:
            error_msg = str(e)
            # Error message should contain information about the op type mismatch
            self.assertTrue("Add" in error_msg or "Mul" in error_msg)


if __name__ == "__main__":
    unittest.main()
