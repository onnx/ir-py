# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir import traversal


class RecursiveGraphIteratorTest(unittest.TestCase):
    def setUp(self):
        self.graph = ir.Graph(
            [],
            [],
            nodes=[
                ir.Node("", "Node1", []),
                ir.Node("", "Node2", []),
                ir.Node(
                    "",
                    "If",
                    [],
                    attributes=[
                        ir.AttrGraph(
                            "then_branch",
                            ir.Graph(
                                [],
                                [],
                                nodes=[ir.Node("", "Node3", []), ir.Node("", "Node4", [])],
                                name="then_graph",
                            ),
                        ),
                        ir.AttrGraph(
                            "else_branch",
                            ir.Graph(
                                [],
                                [],
                                nodes=[ir.Node("", "Node5", []), ir.Node("", "Node6", [])],
                                name="else_graph",
                            ),
                        ),
                    ],
                ),
            ],
            name="main_graph",
        )

    @parameterized.parameterized.expand(
        [
            ("forward", False, ("Node1", "Node2", "If", "Node3", "Node4", "Node5", "Node6")),
            ("reversed", True, ("If", "Node4", "Node3", "Node6", "Node5", "Node2", "Node1")),
        ]
    )
    def test_recursive_graph_iterator(self, _: str, reverse: bool, expected: tuple[str, ...]):
        iterator = traversal.RecursiveGraphIterator(self.graph)
        if reverse:
            iterator = reversed(iterator)
        nodes = list(iterator)
        self.assertEqual(tuple(node.op_type for node in nodes), expected)

    @parameterized.parameterized.expand(
        [
            ("forward", False, ("Node1", "Node2", "If")),
            ("reversed", True, ("If", "Node2", "Node1")),
        ]
    )
    def test_recursive_graph_iterator_recursive_controls_recursive_behavior(
        self, _: str, reverse: bool, expected: list[str]
    ):
        nodes = list(
            traversal.RecursiveGraphIterator(
                self.graph, recursive=lambda node: node.op_type != "If", reverse=reverse
            )
        )
        self.assertEqual(tuple(node.op_type for node in nodes), expected)


class RecursiveGraphIteratorGraphsAttrTest(unittest.TestCase):
    def test_recursive_graph_iterator_with_graphs_attribute(self):
        """Test iteration over nodes in GRAPHS (plural) attributes."""
        inner_node1 = ir.Node("", "InnerOp1", [])
        inner_node2 = ir.Node("", "InnerOp2", [])
        sub_graph1 = ir.Graph([], [], nodes=[inner_node1], name="sub1")
        sub_graph2 = ir.Graph([], [], nodes=[inner_node2], name="sub2")

        outer_node = ir.Node(
            "",
            "If",
            [],
            attributes=[ir.AttrGraphs("branches", [sub_graph1, sub_graph2])],
        )
        main_graph = ir.Graph([], [], nodes=[outer_node], name="main")

        all_nodes = list(traversal.RecursiveGraphIterator(main_graph))
        op_types = [n.op_type for n in all_nodes]
        self.assertIn("If", op_types)
        self.assertIn("InnerOp1", op_types)
        self.assertIn("InnerOp2", op_types)

    def test_recursive_graph_iterator_reverse(self):
        n1 = ir.Node("", "Op1", [])
        n2 = ir.Node("", "Op2", [])
        graph = ir.Graph([], [], nodes=[n1, n2], name="g")
        nodes = list(traversal.RecursiveGraphIterator(graph, reverse=True))
        self.assertEqual(nodes[0].op_type, "Op2")
        self.assertEqual(nodes[1].op_type, "Op1")


if __name__ == "__main__":
    unittest.main()
