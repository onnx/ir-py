# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx

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
        self.assertNotIn("w", graph.initializers)
        self.assertIsNone(v._graph)

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

    def test_graph_io_slice_replace_with_iterable(self):
        for kind in ("inputs", "outputs"):
            for iterable_type in (list, tuple, iter, lambda values: (v for v in values)):
                for index, replacement_count in (
                    (slice(None), 2),
                    (slice(1, 3), 1),
                    (slice(None, None, 2), 2),
                    (slice(None, None, -1), 4),
                    (slice(1, 1, 2), 0),
                    (slice(None), 0),
                ):
                    with self.subTest(kind=kind, iterable_type=iterable_type, index=index):
                        old = [ir.Value(name=f"old_{i}") for i in range(4)]
                        new = [ir.Value(name=f"new_{i}") for i in range(replacement_count)]
                        graph = ir.Graph([], [], nodes=[])
                        values = getattr(graph, kind)
                        values.extend(old)
                        expected = list(old)
                        expected[index] = new
                        values[index] = iterable_type(new)
                        self.assertEqual(list(values), expected)
                        for value in old + new:
                            is_member = value in expected
                            self.assertIs(value.graph, graph if is_member else None)
                            self.assertEqual(
                                value.is_graph_input(), kind == "inputs" and is_member
                            )
                            self.assertEqual(
                                value.is_graph_output(), kind == "outputs" and is_member
                            )
                        values.clear()
                        for value in old + new:
                            self.assertIsNone(value.graph)

    def test_graph_io_slice_replace_from_self(self):
        for kind in ("inputs", "outputs"):
            for use_generator in (False, True):
                with self.subTest(kind=kind, use_generator=use_generator):
                    old = [ir.Value(name=f"value_{i}") for i in range(3)]
                    graph = ir.Graph([], [], nodes=[])
                    values = getattr(graph, kind)
                    values.extend(old)
                    replacement = (value for value in values) if use_generator else values
                    expected = list(old)
                    expected[1:2] = old
                    values[1:2] = replacement
                    self.assertEqual(list(values), expected)
                    # Values repeated by the replacement stay owned until their last use.
                    while values:
                        value = values.pop()
                        self.assertIs(value.graph, graph if value in values else None)
                        self.assertEqual(
                            value.is_graph_input(), kind == "inputs" and value in values
                        )
                        self.assertEqual(
                            value.is_graph_output(), kind == "outputs" and value in values
                        )

    def test_graph_io_extended_slice_rejects_wrong_length(self):
        for kind in ("inputs", "outputs"):
            for replacement_count in (0, 1, 3):
                with self.subTest(kind=kind, replacement_count=replacement_count):
                    old = [ir.Value(name=f"old_{i}") for i in range(4)]
                    new = [ir.Value(name=f"new_{i}") for i in range(replacement_count)]
                    graph = ir.Graph([], [], nodes=[])
                    values = getattr(graph, kind)
                    values.extend(old)
                    expected = list(old)
                    with self.assertRaises(ValueError):
                        expected[::2] = iter(new)
                    with self.assertRaises(ValueError):
                        values[::2] = iter(new)
                    self.assertEqual(list(values), expected)

    def test_model_clone_keeps_graph_io_after_generator_slice(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["": 18]>
            identity (float[2] x) => (float[2] y) {
                y = Identity(x)
            }
            """
        )
        original = ir.to_proto(model)
        for kind in ("inputs", "outputs"):
            with self.subTest(kind=kind):
                clone = model.clone()
                values = getattr(clone.graph, kind)
                expected = list(values)
                values[:] = (value for value in values)
                self.assertEqual(list(values), expected)
                serialized = ir.to_proto(clone)
                self.assertEqual(serialized, original)
                onnx.checker.check_model(serialized, full_check=True)
                self.assertEqual(ir.to_proto(model), original)

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
