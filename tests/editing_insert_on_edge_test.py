# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for editing.insert_on_edge."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir import editing


class InsertOnEdgeBasicTest(unittest.TestCase):
    """Test basic insert_on_edge behavior."""

    def test_insert_between_two_nodes(self):
        """Insert D between A and B in A -> B chain."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )

        # Insert D after A's output (between A and B)
        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        new_val = editing.insert_on_edge(node_a.outputs[0], node_d)

        # B should now consume D's output
        self.assertIs(node_b.inputs[0], node_d.outputs[0])
        # D should consume A's output
        self.assertIs(node_d.inputs[0], node_a.outputs[0])
        # Return value should be D's output
        self.assertIs(new_val, node_d.outputs[0])
        # D should be in the graph between A and B
        graph_nodes = list(graph)
        self.assertEqual(graph_nodes, [node_a, node_d, node_b])

    def test_insert_redirects_all_consumers(self):
        """All consumers of the value are redirected, not just one."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        node_c = ir.Node("", "C", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[*node_b.outputs, *node_c.outputs],
            nodes=[node_a, node_b, node_c],
        )

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        editing.insert_on_edge(node_a.outputs[0], node_d)

        # Both B and C should now consume D's output
        self.assertIs(node_b.inputs[0], node_d.outputs[0])
        self.assertIs(node_c.inputs[0], node_d.outputs[0])

    def test_returns_new_output_value(self):
        """The return value is the new output for composition."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_a.outputs),
            nodes=[node_a],
        )

        node_cast = ir.Node("", "Cast", inputs=list(node_a.outputs), num_outputs=1)
        result = editing.insert_on_edge(node_a.outputs[0], node_cast)

        self.assertIs(result, node_cast.outputs[0])

    def test_insert_with_output_index(self):
        """Use a non-zero output_index for multi-output new_node."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )

        # new_node has 3 outputs; we use index 2
        node_d = ir.Node(
            "", "D", inputs=list(node_a.outputs), num_outputs=3
        )
        result = editing.insert_on_edge(
            node_a.outputs[0], node_d, output_index=2
        )

        self.assertIs(result, node_d.outputs[2])
        self.assertIs(node_b.inputs[0], node_d.outputs[2])

    def test_new_node_input_is_preserved(self):
        """new_node's input (the original value) is NOT redirected."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        editing.insert_on_edge(node_a.outputs[0], node_d)

        # D should still consume A's output, not its own
        self.assertIs(node_d.inputs[0], node_a.outputs[0])


class InsertOnEdgeGraphInputTest(unittest.TestCase):
    """Test insert_on_edge when value is a graph input."""

    def test_insert_after_graph_input(self):
        """Insert a node after a graph input value."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_a.outputs),
            nodes=[node_a],
        )

        node_cast = ir.Node(
            "", "Cast", inputs=[graph_input], num_outputs=1
        )
        editing.insert_on_edge(graph_input, node_cast)

        # A should now consume Cast's output
        self.assertIs(node_a.inputs[0], node_cast.outputs[0])
        # Cast should consume the graph input
        self.assertIs(node_cast.inputs[0], graph_input)
        # Cast should be first in the graph (before A)
        graph_nodes = list(graph)
        self.assertEqual(graph_nodes[0], node_cast)

    def test_insert_after_graph_input_with_multiple_consumers(self):
        """Graph input consumed by multiple nodes — all redirected."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[*node_a.outputs, *node_b.outputs],
            nodes=[node_a, node_b],
        )

        node_d = ir.Node("", "D", inputs=[graph_input], num_outputs=1)
        editing.insert_on_edge(graph_input, node_d)

        self.assertIs(node_a.inputs[0], node_d.outputs[0])
        self.assertIs(node_b.inputs[0], node_d.outputs[0])
        # D should be first
        self.assertEqual(list(graph)[0], node_d)


class InsertOnEdgeGraphOutputTest(unittest.TestCase):
    """Test insert_on_edge when value is a graph output."""

    def test_insert_on_graph_output_value(self):
        """When the value is a graph output, the graph output is updated."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_a.outputs),
            nodes=[node_a],
        )

        node_identity = ir.Node(
            "", "Identity", inputs=list(node_a.outputs), num_outputs=1
        )
        new_val = editing.insert_on_edge(node_a.outputs[0], node_identity)

        # Graph output should now be the Identity's output
        self.assertIs(graph.outputs[0], new_val)
        self.assertIs(graph.outputs[0], node_identity.outputs[0])

    def test_graph_output_value_is_also_graph_input(self):
        """Edge case: value that is both graph input and graph output."""
        graph_input = ir.Value(name="passthrough")
        # Graph output IS the graph input (identity graph)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[graph_input],
            nodes=[],
        )

        node_identity = ir.Node(
            "", "Identity", inputs=[graph_input], num_outputs=1
        )
        new_val = editing.insert_on_edge(graph_input, node_identity)

        self.assertIs(graph.outputs[0], new_val)
        self.assertEqual(list(graph), [node_identity])

    def test_value_appears_multiple_times_in_graph_outputs(self):
        """Value that appears multiple times in graph outputs."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        # Same value as graph output twice
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[node_a.outputs[0], node_a.outputs[0]],
            nodes=[node_a],
        )

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        editing.insert_on_edge(node_a.outputs[0], node_d)

        # Both graph output slots should be updated
        self.assertIs(graph.outputs[0], node_d.outputs[0])
        self.assertIs(graph.outputs[1], node_d.outputs[0])


class InsertOnEdgePositionTest(unittest.TestCase):
    """Test that new_node is inserted at the correct graph position."""

    def test_insert_after_producer_node(self):
        """new_node is placed right after the producer in graph order."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        node_c = ir.Node("", "C", inputs=node_b.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_c.outputs),
            nodes=[node_a, node_b, node_c],
        )

        # Insert D after A's output
        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        editing.insert_on_edge(node_a.outputs[0], node_d)

        graph_nodes = list(graph)
        # D should be right after A
        self.assertEqual(
            graph_nodes.index(node_d), graph_nodes.index(node_a) + 1
        )

    def test_insert_before_first_node_when_graph_input(self):
        """When value is a graph input, new_node is placed before the first node."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )

        node_d = ir.Node("", "D", inputs=[graph_input], num_outputs=1)
        editing.insert_on_edge(graph_input, node_d)

        self.assertEqual(list(graph)[0], node_d)


class InsertOnEdgeValidationTest(unittest.TestCase):
    """Test error handling and validation."""

    def test_error_when_new_node_does_not_consume_value(self):
        """Raise ValueError if new_node doesn't have value as input."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_a.outputs),
            nodes=[node_a],
        )

        # new_node uses a different value, not node_a's output
        other_value = ir.Value(name="other")
        node_d = ir.Node("", "D", inputs=[other_value], num_outputs=1)

        with self.assertRaises(ValueError):
            editing.insert_on_edge(node_a.outputs[0], node_d)

    def test_error_when_new_node_already_in_graph(self):
        """Raise ValueError if new_node already belongs to a graph."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )

        # node_b already belongs to the graph
        with self.assertRaises(ValueError):
            editing.insert_on_edge(node_a.outputs[0], node_b)

    def test_error_when_value_has_no_graph(self):
        """Raise ValueError if value is detached (no graph context)."""
        detached_value = ir.Value(name="detached")
        node_d = ir.Node("", "D", inputs=[detached_value], num_outputs=1)

        with self.assertRaises(ValueError):
            editing.insert_on_edge(detached_value, node_d)


class InsertOnEdgeCompositionTest(unittest.TestCase):
    """Test that insert_on_edge composes well for real-world patterns."""

    def test_chain_two_insertions(self):
        """Insert two nodes in sequence: Cast then Relu after A."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )

        # First insert Cast
        cast = ir.Node("", "Cast", inputs=list(node_a.outputs), num_outputs=1)
        cast_out = editing.insert_on_edge(node_a.outputs[0], cast)

        # Then insert Relu after Cast
        relu = ir.Node("", "Relu", inputs=[cast_out], num_outputs=1)
        relu_out = editing.insert_on_edge(cast_out, relu)

        # B should consume Relu's output
        self.assertIs(node_b.inputs[0], relu_out)
        # Graph order: A, Cast, Relu, B
        self.assertEqual(list(graph), [node_a, cast, relu, node_b])

    def test_insert_relu_after_every_conv_pattern(self):
        """Simulate: insert Relu after every Conv output."""
        graph_input = ir.Value(name="input")
        conv1 = ir.Node("", "Conv", inputs=[graph_input], num_outputs=1)
        conv2 = ir.Node("", "Conv", inputs=conv1.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(conv2.outputs),
            nodes=[conv1, conv2],
        )

        # Insert Relu after each Conv
        for conv_node in [conv1, conv2]:
            relu = ir.Node(
                "", "Relu", inputs=list(conv_node.outputs), num_outputs=1
            )
            editing.insert_on_edge(conv_node.outputs[0], relu)

        graph_nodes = list(graph)
        op_types = [n.op_type for n in graph_nodes]
        self.assertEqual(op_types, ["Conv", "Relu", "Conv", "Relu"])

    def test_does_not_mutate_original_graph_order_beyond_insertion(self):
        """Existing nodes stay in the same relative order."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        node_c = ir.Node("", "C", inputs=node_b.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node_c.outputs),
            nodes=[node_a, node_b, node_c],
        )

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        editing.insert_on_edge(node_a.outputs[0], node_d)

        graph_nodes = list(graph)
        # Original nodes maintain relative order
        self.assertLess(graph_nodes.index(node_a), graph_nodes.index(node_b))
        self.assertLess(graph_nodes.index(node_b), graph_nodes.index(node_c))


if __name__ == "__main__":
    unittest.main()
