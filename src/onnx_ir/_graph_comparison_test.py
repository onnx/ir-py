# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

from onnx_ir import _core, _enums, _graph_comparison


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
        attr1 = _core.Attr("axis", _enums.AttributeType.INT, 1)
        node1 = _core.Node("", "Squeeze", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Graph 2 - same attributes
        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("axis", _enums.AttributeType.INT, 1)
        node2 = _core.Node("", "Squeeze", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_different_attribute_names(self):
        """Test that graphs with different attribute names are not equal."""
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("axis", _enums.AttributeType.INT, 1)
        node1 = _core.Node("", "Op", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("dim", _enums.AttributeType.INT, 1)
        node2 = _core.Node("", "Op", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_different_attribute_types(self):
        """Test that graphs with different attribute types are not equal."""
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("value", _enums.AttributeType.INT, 1)
        node1 = _core.Node("", "Op", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("value", _enums.AttributeType.FLOAT, 1.0)
        node2 = _core.Node("", "Op", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_without_initializers_by_default(self):
        """Test that initializer data comparison can be skipped with tensor_size_limit=0."""
        # Graph 1 with initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=_core.Tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with different initializer data but same shape/dtype
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=_core.Tensor(np.array([3.0, 4.0], dtype=np.float32))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        # Should be equal when using tensor_size_limit=0 (skip data comparison for tensors > size 0)
        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2, tensor_size_limit=0))
        
        # Should NOT be equal with default (None) - data is compared
        self.assertFalse(_graph_comparison.topologically_equal(graph1, graph2))

    def test_with_initializers_when_enabled(self):
        """Test that initializers data is compared when tensor_size_limit=None."""
        # Graph 1 with initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=_core.Tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with same initializer shape and dtype
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=_core.Tensor(np.array([1.0, 2.0], dtype=np.float32))
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
            name="init1", const_value=_core.Tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with different initializer shape
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=_core.Tensor(np.array([1.0], dtype=np.float32))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        # Should not be equal - shapes are always compared
        self.assertFalse(
            _graph_comparison.topologically_equal(graph1, graph2)
        )

    def test_with_different_initializer_dtypes(self):
        """Test that graphs with different initializer dtypes are not equal."""
        # Graph 1 with float32 initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=_core.Tensor(np.array([1.0, 2.0], dtype=np.float32))
        )
        node1 = _core.Node("", "Add", inputs=(v1, init1), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,), initializers=(init1,))

        # Graph 2 with float64 initializer
        v2 = _core.Value(name="v2")
        init2 = _core.Value(
            name="init2", const_value=_core.Tensor(np.array([1.0, 2.0], dtype=np.float64))
        )
        node2 = _core.Node("", "Add", inputs=(v2, init2), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,), initializers=(init2,))

        # Should not be equal - dtypes are always compared
        self.assertFalse(
            _graph_comparison.topologically_equal(graph1, graph2)
        )

    def test_with_different_number_of_initializers(self):
        """Test that graphs with different numbers of initializers are not equal."""
        # Graph 1 with one initializer
        v1 = _core.Value(name="v1")
        init1 = _core.Value(
            name="init1", const_value=_core.Tensor(np.array([1.0], dtype=np.float32))
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
            _graph_comparison.topologically_equal(graph1, graph2, )
        )

    def test_with_subgraph_attributes(self):
        """Test that graphs with subgraph attributes are compared recursively."""
        # Create a simple subgraph
        sub_input1 = _core.Value(name="sub_input1")
        sub_node1 = _core.Node("", "Identity", inputs=(sub_input1,), num_outputs=1)
        subgraph1 = _core.Graph((sub_input1,), sub_node1.outputs, nodes=(sub_node1,))

        # Graph 1 with subgraph attribute
        v1 = _core.Value(name="v1")
        attr1 = _core.Attr("body", _enums.AttributeType.GRAPH, subgraph1)
        node1 = _core.Node("", "Loop", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Create an identical subgraph
        sub_input2 = _core.Value(name="sub_input2")
        sub_node2 = _core.Node("", "Identity", inputs=(sub_input2,), num_outputs=1)
        subgraph2 = _core.Graph((sub_input2,), sub_node2.outputs, nodes=(sub_node2,))

        # Graph 2 with identical subgraph attribute
        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("body", _enums.AttributeType.GRAPH, subgraph2)
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
        attr1 = _core.Attr("body", _enums.AttributeType.GRAPH, subgraph1)
        node1 = _core.Node("", "Loop", inputs=(v1,), attributes=(attr1,), num_outputs=1)
        graph1 = _core.Graph((v1,), node1.outputs, nodes=(node1,))

        # Create a different subgraph (different op type)
        sub_input2 = _core.Value(name="sub_input2")
        sub_node2 = _core.Node("", "Relu", inputs=(sub_input2,), num_outputs=1)
        subgraph2 = _core.Graph((sub_input2,), sub_node2.outputs, nodes=(sub_node2,))

        # Graph 2 with different subgraph
        v2 = _core.Value(name="v2")
        attr2 = _core.Attr("body", _enums.AttributeType.GRAPH, subgraph2)
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
        attr1 = _core.Attr("branches", _enums.AttributeType.GRAPHS, [subgraph1_a, subgraph1_b])
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
        attr2 = _core.Attr("branches", _enums.AttributeType.GRAPHS, [subgraph2_a, subgraph2_b])
        node2 = _core.Node("", "If", inputs=(v2,), attributes=(attr2,), num_outputs=1)
        graph2 = _core.Graph((v2,), node2.outputs, nodes=(node2,))

        self.assertTrue(_graph_comparison.topologically_equal(graph1, graph2))


if __name__ == "__main__":
    unittest.main()
