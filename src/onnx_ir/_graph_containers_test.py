# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx_ir as ir


class GraphContainersTest(unittest.TestCase):
    def test_graph_inputs_set_graph(self):
        v = ir.Value(name="x")
        graph = ir.Graph([v], [], nodes=[])
        self.assertIs(v._graph, graph)

    def test_graph_inputs_cannot_be_from_different_graph(self):
        v = ir.Value(name="x")
        _ = ir.Graph([v], [], nodes=[])
        # v now belongs to graph1
        with self.assertRaises(ValueError):
            ir.Graph([v], [], nodes=[])

    def test_graph_outputs_set_graph(self):
        v = ir.Value(name="y")
        graph = ir.Graph([], [v], nodes=[])
        self.assertIs(v._graph, graph)

    def test_graph_initializers_set_and_delete(self):
        v = ir.Value(name="w", const_value=ir.tensor([1.0]))
        graph = ir.Graph([], [], nodes=[])
        graph.initializers["w"] = v
        self.assertIs(v._graph, graph)
        del graph.initializers["w"]
        # Value graph should be None after removal (if not still owned)

    def test_graph_initializers_reject_node_output(self):
        """A value produced by a node cannot be an initializer."""
        node = ir.Node("", "Op", [], num_outputs=1)
        v = node.outputs[0]
        v.name = "bad_init"
        graph = ir.Graph([], [], nodes=[])
        with self.assertRaises(ValueError):
            graph.initializers["bad_init"] = v

    def test_graph_initializers_reject_empty_name(self):
        v = ir.Value(name="", const_value=ir.tensor([1.0]))
        graph = ir.Graph([], [], nodes=[])
        with self.assertRaises(ValueError):
            graph.initializers[""] = v

    def test_graph_initializers_reject_mismatched_name(self):
        v = ir.Value(name="actual_name", const_value=ir.tensor([1.0]))
        graph = ir.Graph([], [], nodes=[])
        with self.assertRaises(ValueError):
            graph.initializers["wrong_name"] = v

    def test_graph_initializers_add(self):
        v = ir.Value(name="w", const_value=ir.tensor([1.0]))
        graph = ir.Graph([], [], nodes=[])
        graph.initializers.add(v)
        self.assertIn("w", graph.initializers)

    def test_graph_inputs_replace(self):
        v1 = ir.Value(name="x")
        v2 = ir.Value(name="y")
        graph = ir.Graph([v1], [], nodes=[])
        graph.inputs[0] = v2
        self.assertEqual(graph.inputs[0].name, "y")

    def test_graph_inputs_slice_replace(self):
        v1 = ir.Value(name="x")
        v2 = ir.Value(name="y")
        v3 = ir.Value(name="z")
        graph = ir.Graph([v1, v2], [], nodes=[])
        graph.inputs[0:2] = [v3]
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs[0].name, "z")

    def test_graph_io_unimplemented_methods(self):
        v = ir.Value(name="x")
        graph = ir.Graph([v], [], nodes=[])
        with self.assertRaises(RuntimeError):
            graph.inputs + graph.inputs
        with self.assertRaises(RuntimeError):
            graph.inputs * 2

    def test_graph_io_copy(self):
        v1 = ir.Value(name="x")
        v2 = ir.Value(name="y")
        graph = ir.Graph([v1, v2], [], nodes=[])
        copied = graph.inputs.copy()
        self.assertEqual(len(copied), 2)

    def test_attributes_set_and_add(self):
        node = ir.Node("", "Op", [], attributes=[ir.AttrInt64("a", 1)])
        node.attributes["b"] = ir.AttrFloat32("b", 2.0)
        self.assertEqual(node.attributes["b"].as_float(), 2.0)
        node.attributes.add(ir.AttrString("c", "hello"))
        self.assertEqual(node.attributes["c"].as_string(), "hello")

    def test_attributes_reject_non_string_key(self):
        node = ir.Node("", "Op", [])
        with self.assertRaises(TypeError):
            node.attributes[42] = ir.AttrInt64("a", 1)  # type: ignore[index]

    def test_attributes_reject_non_attr_value(self):
        node = ir.Node("", "Op", [])
        with self.assertRaises(TypeError):
            node.attributes["a"] = "not an attr"  # type: ignore[assignment]

    def test_attributes_get_helpers(self):
        node = ir.Node(
            "",
            "Op",
            [],
            attributes=[
                ir.AttrInt64("i", 42),
                ir.AttrFloat32("f", 3.14),
                ir.AttrString("s", "hi"),
                ir.AttrInt64s("is_", [1, 2]),
                ir.AttrFloat32s("fs", [1.0]),
                ir.AttrStrings("ss", ["a"]),
            ],
        )
        self.assertEqual(node.attributes.get_int("i"), 42)
        self.assertAlmostEqual(node.attributes.get_float("f"), 3.14, places=2)
        self.assertEqual(node.attributes.get_string("s"), "hi")
        self.assertEqual(list(node.attributes.get_ints("is_")), [1, 2])
        self.assertEqual(list(node.attributes.get_floats("fs")), [1.0])
        self.assertEqual(list(node.attributes.get_strings("ss")), ["a"])
        # Defaults
        self.assertIsNone(node.attributes.get_int("missing"))
        self.assertIsNone(node.attributes.get_float("missing"))
        self.assertIsNone(node.attributes.get_string("missing"))

    def test_graph_initializers_type_check(self):
        graph = ir.Graph([], [], nodes=[])
        with self.assertRaises(TypeError):
            graph.initializers["k"] = "not a value"  # type: ignore[assignment]
        with self.assertRaises(TypeError):
            graph.initializers[42] = ir.Value(name="v")  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
