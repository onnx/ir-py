# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for onnx_ir.traversal, including topological_order."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir import traversal


def _names(nodes):
    """Extract node names from an iterable of nodes for easy assertion."""
    return [node.name for node in nodes]


class TopologicalOrderTest(unittest.TestCase):
    """Tests for traversal.topological_order."""

    def test_empty_graph(self):
        graph = ir.Graph([], [], nodes=[])
        result = list(traversal.topological_order(graph))
        self.assertEqual(result, [])

    def test_single_node(self):
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        graph = ir.Graph([v_in], node_a.outputs, nodes=[node_a])

        result = list(traversal.topological_order(graph))
        self.assertEqual(_names(result), ["A"])

    def test_linear_chain_already_sorted(self):
        """A -> B -> C: already in correct order."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        node_c = ir.node("Op", inputs=[node_b.outputs[0]], name="C")
        graph = ir.Graph(
            [v_in], node_c.outputs, nodes=[node_a, node_b, node_c]
        )

        result = list(traversal.topological_order(graph))
        self.assertEqual(_names(result), ["A", "B", "C"])

    def test_linear_chain_reversed_input(self):
        """C -> B -> A in original order, but dependency is A -> B -> C."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        node_c = ir.node("Op", inputs=[node_b.outputs[0]], name="C")
        # Insert nodes in reverse order
        graph = ir.Graph(
            [v_in], node_c.outputs, nodes=[node_c, node_b, node_a]
        )

        result = list(traversal.topological_order(graph))
        self.assertEqual(_names(result), ["A", "B", "C"])

    def test_diamond_dependency(self):
        """Diamond: A -> B, A -> C, B -> D, C -> D."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        node_c = ir.node("Op", inputs=[node_a.outputs[0]], name="C")
        node_d = ir.node(
            "Op", inputs=[node_b.outputs[0], node_c.outputs[0]], name="D"
        )
        graph = ir.Graph(
            [v_in], node_d.outputs, nodes=[node_a, node_b, node_c, node_d]
        )

        result = _names(traversal.topological_order(graph))
        # A must come first, D must come last, B and C in between
        self.assertEqual(result[0], "A")
        self.assertEqual(result[-1], "D")
        self.assertIn("B", result)
        self.assertIn("C", result)
        # B and C preserve original order as tie-break
        self.assertEqual(result, ["A", "B", "C", "D"])

    def test_disconnected_nodes_preserve_original_order(self):
        """Independent nodes should preserve original graph order."""
        v1 = ir.Value(name="v1")
        v2 = ir.Value(name="v2")
        v3 = ir.Value(name="v3")
        node_a = ir.node("Op", inputs=[v1], name="A")
        node_b = ir.node("Op", inputs=[v2], name="B")
        node_c = ir.node("Op", inputs=[v3], name="C")
        graph = ir.Graph(
            [v1, v2, v3],
            [*node_a.outputs, *node_b.outputs, *node_c.outputs],
            nodes=[node_a, node_b, node_c],
        )

        result = _names(traversal.topological_order(graph))
        self.assertEqual(result, ["A", "B", "C"])

    def test_multiple_inputs(self):
        """Node with multiple inputs from different producers."""
        v1 = ir.Value(name="v1")
        v2 = ir.Value(name="v2")
        node_a = ir.node("Op", inputs=[v1], name="A")
        node_b = ir.node("Op", inputs=[v2], name="B")
        node_c = ir.node(
            "Op", inputs=[node_a.outputs[0], node_b.outputs[0]], name="C"
        )
        graph = ir.Graph(
            [v1, v2], node_c.outputs, nodes=[node_a, node_b, node_c]
        )

        result = _names(traversal.topological_order(graph))
        self.assertEqual(result[0], "A")
        self.assertEqual(result[1], "B")
        self.assertEqual(result[2], "C")

    def test_stability_among_peers(self):
        """Nodes at the same topological level preserve original order."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        # B and C both depend only on A
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        node_c = ir.node("Op", inputs=[node_a.outputs[0]], name="C")
        graph = ir.Graph(
            [v_in],
            [*node_b.outputs, *node_c.outputs],
            nodes=[node_a, node_b, node_c],
        )

        result = _names(traversal.topological_order(graph))
        self.assertEqual(result, ["A", "B", "C"])

    def test_stability_reversed_insertion(self):
        """Same topology as stability_among_peers but C inserted before B."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        node_c = ir.node("Op", inputs=[node_a.outputs[0]], name="C")
        # C comes before B in original graph order
        graph = ir.Graph(
            [v_in],
            [*node_c.outputs, *node_b.outputs],
            nodes=[node_a, node_c, node_b],
        )

        result = _names(traversal.topological_order(graph))
        self.assertEqual(result, ["A", "C", "B"])

    def test_cycle_raises_value_error(self):
        """A cycle in the graph should raise ValueError."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        # Create cycle: A -> B -> A
        node_a.replace_input_with(0, node_b.outputs[0])
        graph = ir.Graph(
            [v_in], node_b.outputs, nodes=[node_a, node_b]
        )

        with self.assertRaises(ValueError):
            list(traversal.topological_order(graph))

    def test_none_inputs_are_skipped(self):
        """Nodes with None inputs (optional) should not crash."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in, None], name="A")
        graph = ir.Graph([v_in], node_a.outputs, nodes=[node_a])

        result = _names(traversal.topological_order(graph))
        self.assertEqual(result, ["A"])

    def test_is_lazy_iterator(self):
        """topological_order returns an iterator, not a list."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        graph = ir.Graph([v_in], node_a.outputs, nodes=[node_a])

        result = traversal.topological_order(graph)
        # Should be an iterator, not a list or tuple
        self.assertTrue(hasattr(result, "__next__"))

    def test_does_not_mutate_graph(self):
        """The graph's node order should not change after iteration."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        node_c = ir.node("Op", inputs=[node_b.outputs[0]], name="C")
        # Insert in reverse order to verify the graph isn't re-ordered
        graph = ir.Graph(
            [v_in], node_c.outputs, nodes=[node_c, node_b, node_a]
        )
        original_order = _names(graph)

        # Consume the entire iterator
        list(traversal.topological_order(graph))

        # Graph order must be unchanged
        self.assertEqual(_names(graph), original_order)

    def test_graph_view(self):
        """topological_order works with GraphView."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        graph_view = ir.GraphView(
            [v_in], node_b.outputs, nodes=[node_b, node_a]
        )

        result = _names(traversal.topological_order(graph_view))
        self.assertEqual(result, ["A", "B"])

    def test_recursive_with_subgraph(self):
        """recursive=True yields subgraph nodes before the parent node."""
        v_in = ir.Value(name="v_in")
        # Create a subgraph with one node
        sub_input = ir.Value(name="sub_in")
        sub_node = ir.node("SubOp", inputs=[sub_input], name="SubNode")
        subgraph = ir.Graph(
            [sub_input], sub_node.outputs, nodes=[sub_node]
        )

        # Parent node has a graph attribute (like If/Loop)
        node_with_sub = ir.node(
            "If",
            inputs=[v_in],
            attributes={"then_branch": subgraph},
            name="IfNode",
        )
        graph = ir.Graph(
            [v_in], node_with_sub.outputs, nodes=[node_with_sub]
        )

        result = _names(traversal.topological_order(graph, recursive=True))
        # SubNode must come before IfNode
        sub_idx = result.index("SubNode")
        if_idx = result.index("IfNode")
        self.assertLess(sub_idx, if_idx)

    def test_non_recursive_excludes_subgraph_nodes(self):
        """recursive=False should only yield top-level nodes."""
        v_in = ir.Value(name="v_in")
        sub_input = ir.Value(name="sub_in")
        sub_node = ir.node("SubOp", inputs=[sub_input], name="SubNode")
        subgraph = ir.Graph(
            [sub_input], sub_node.outputs, nodes=[sub_node]
        )

        node_with_sub = ir.node(
            "If",
            inputs=[v_in],
            attributes={"then_branch": subgraph},
            name="IfNode",
        )
        graph = ir.Graph(
            [v_in], node_with_sub.outputs, nodes=[node_with_sub]
        )

        result = _names(traversal.topological_order(graph, recursive=False))
        self.assertEqual(result, ["IfNode"])
        self.assertNotIn("SubNode", result)

    def test_subgraph_implicit_dependency(self):
        """A subgraph referencing an outer-scope value creates an implicit dependency."""
        v_in = ir.Value(name="v_in")
        node_producer = ir.node("Op", inputs=[v_in], name="Producer")

        # The subgraph uses Producer's output (outer-scope value)
        sub_node = ir.node(
            "SubOp", inputs=[node_producer.outputs[0]], name="SubNode"
        )
        subgraph = ir.Graph([], sub_node.outputs, nodes=[sub_node])

        node_if = ir.node(
            "If",
            inputs=[v_in],
            attributes={"then_branch": subgraph},
            name="IfNode",
        )
        # Insert IfNode before Producer in graph order
        graph = ir.Graph(
            [v_in], node_if.outputs, nodes=[node_if, node_producer]
        )

        # Even with recursive=False, Producer must come before IfNode
        # because the subgraph references Producer's output
        result = _names(traversal.topological_order(graph, recursive=False))
        self.assertEqual(result, ["Producer", "IfNode"])

    def test_complex_graph_with_multiple_paths(self):
        """Complex graph with multiple paths and shared dependencies.

             v_in
            /    \\
           A      B
           |      |
           C      |
            \\    /
              D
        """
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[v_in], name="B")
        node_c = ir.node("Op", inputs=[node_a.outputs[0]], name="C")
        node_d = ir.node(
            "Op", inputs=[node_c.outputs[0], node_b.outputs[0]], name="D"
        )
        graph = ir.Graph(
            [v_in],
            node_d.outputs,
            nodes=[node_a, node_b, node_c, node_d],
        )

        result = _names(traversal.topological_order(graph))
        # A and B must come before C and D; C before D; B before D
        a_idx = result.index("A")
        b_idx = result.index("B")
        c_idx = result.index("C")
        d_idx = result.index("D")
        self.assertLess(a_idx, c_idx)
        self.assertLess(c_idx, d_idx)
        self.assertLess(b_idx, d_idx)
        # Stable ordering: A before B (both at level 0)
        self.assertLess(a_idx, b_idx)

    def test_matches_graph_sort_for_simple_graphs(self):
        """Verify topological_order matches graph.sort() for a simple graph."""
        v_in = ir.Value(name="v_in")
        node_a = ir.node("Op", inputs=[v_in], name="A")
        node_b = ir.node("Op", inputs=[node_a.outputs[0]], name="B")
        node_c = ir.node("Op", inputs=[v_in], name="C")
        node_d = ir.node(
            "Op", inputs=[node_b.outputs[0], node_c.outputs[0]], name="D"
        )
        # Insert in a mixed order
        graph = ir.Graph(
            [v_in],
            node_d.outputs,
            nodes=[node_d, node_c, node_a, node_b],
        )

        topo_result = _names(traversal.topological_order(graph))

        # Now sort the graph in-place and compare
        graph.sort()
        sort_result = _names(graph)

        self.assertEqual(topo_result, sort_result)

    def test_recursive_with_nested_subgraphs(self):
        """recursive=True handles nested subgraphs (subgraph within subgraph)."""
        v_in = ir.Value(name="v_in")

        # Inner subgraph
        inner_input = ir.Value(name="inner_in")
        inner_node = ir.node("InnerOp", inputs=[inner_input], name="Inner")
        inner_graph = ir.Graph(
            [inner_input], inner_node.outputs, nodes=[inner_node]
        )

        # Outer subgraph with a node that has its own subgraph
        outer_input = ir.Value(name="outer_in")
        outer_node = ir.node(
            "OuterOp",
            inputs=[outer_input],
            attributes={"body": inner_graph},
            name="Outer",
        )
        outer_graph = ir.Graph(
            [outer_input], outer_node.outputs, nodes=[outer_node]
        )

        # Top-level node with the outer subgraph
        top_node = ir.node(
            "TopOp",
            inputs=[v_in],
            attributes={"branch": outer_graph},
            name="Top",
        )
        graph = ir.Graph([v_in], top_node.outputs, nodes=[top_node])

        result = _names(traversal.topological_order(graph, recursive=True))
        # Inner and Outer must come before Top
        inner_idx = result.index("Inner")
        outer_idx = result.index("Outer")
        top_idx = result.index("Top")
        self.assertLess(inner_idx, outer_idx)
        self.assertLess(outer_idx, top_idx)

    def test_accessible_from_module(self):
        """topological_order is accessible as ir.traversal.topological_order."""
        self.assertTrue(hasattr(ir.traversal, "topological_order"))
        self.assertIs(ir.traversal.topological_order, traversal.topological_order)


if __name__ == "__main__":
    unittest.main()
