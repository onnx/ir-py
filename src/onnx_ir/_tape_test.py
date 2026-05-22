# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir._tape import Builder


class TestTape(unittest.TestCase):
    def test_op(self):
        # Create a simple ONNX model with shape inference
        # Define the model
        inputs = [
            ir.Value(
                name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
            ir.Value(
                name="input_b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
        ]

        tape = ir.tape.Tape()

        _ = tape.op("Add", inputs=inputs)

        self.assertEqual([n.op_type for n in tape.nodes], ["Add"])

    def test_initializers(self):
        inputs = [
            ir.Value(
                name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
            ir.Value(
                name="input_b",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((2, 1)),
                const_value=ir.tensor([[42]] * 2, dtype=ir.DataType.FLOAT),
            ),
        ]

        tape = ir.tape.Tape()

        # Shape and type are not explicitly set for the initializer but it should still work
        initializer = tape.initializer(
            ir.tensor([[2, 3]], dtype=ir.DataType.FLOAT), name="initializer"
        )
        val_add = tape.op("Add", inputs=inputs)
        _ = tape.op("Mul", inputs=[val_add, initializer])

        self.assertEqual([n.op_type for n in tape.nodes], ["Add", "Mul"])
        self.assertEqual(tape.initializers, (initializer,))

    def test_op_multi_out(self):
        inputs = [
            ir.Value(
                name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
            ir.Value(
                name="input_b",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((2, 1)),
                const_value=ir.tensor([[42]] * 2, dtype=ir.DataType.FLOAT),
            ),
        ]

        tape = ir.tape.Tape()

        out1, out2, out3 = tape.op_multi_out("SomeOp", inputs=inputs, num_outputs=3)  # pylint: disable=unbalanced-tuple-unpacking
        _ = tape.op("SomeOtherOp", inputs=[out1, out2, out3])

        self.assertEqual([n.op_type for n in tape.nodes], ["SomeOp", "SomeOtherOp"])


class TestTapeOpMultiOutValidation(unittest.TestCase):
    def test_op_multi_out_raises_when_both_num_outputs_and_outputs_provided(self):
        tape = ir.tape.Tape()
        inputs = [ir.Value(name="x")]
        outputs = [ir.Value(name="o1")]
        with self.assertRaises(ValueError):
            tape.op_multi_out("Op", inputs, num_outputs=1, outputs=outputs)

    def test_op_multi_out_raises_when_neither_provided(self):
        tape = ir.tape.Tape()
        inputs = [ir.Value(name="x")]
        with self.assertRaises(ValueError):
            tape.op_multi_out("Op", inputs)

    def test_op_multi_out_with_outputs_param(self):
        tape = ir.tape.Tape()
        inputs = [ir.Value(name="x")]
        o1 = ir.Value(name="o1")
        o2 = ir.Value(name="o2")
        results = tape.op_multi_out("Op", inputs, outputs=[o1, o2])
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], o1)
        self.assertIs(results[1], o2)

    def test_op_with_output_param(self):
        tape = ir.tape.Tape()
        inputs = [ir.Value(name="x")]
        output = ir.Value(name="y")
        result = tape.op("Add", inputs, output=output)
        self.assertIs(result, output)

    def test_op_with_attributes(self):
        tape = ir.tape.Tape()
        inputs = [ir.Value(name="x")]
        result = tape.op("Elu", inputs, attributes={"alpha": 2.0})
        node = tape.nodes[0]
        self.assertEqual(node.op_type, "Elu")
        self.assertEqual(node.attributes["alpha"].as_float(), 2.0)
        self.assertIsNotNone(result)

    def test_op_multi_out_with_attributes(self):
        tape = ir.tape.Tape()
        inputs = [ir.Value(name="x")]
        results = tape.op_multi_out(
            "Split", inputs, attributes={"num_outputs": 3}, num_outputs=3
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(tape.nodes[0].attributes["num_outputs"].as_int(), 3)


class TestTapeWithGraph(unittest.TestCase):
    def test_tape_with_graph_sets_node_graph(self):
        graph = ir.Graph([], [], nodes=[], name="test_graph")
        tape = ir.tape.Tape(graph_like=graph)
        inputs = [ir.Value(name="x")]
        _ = tape.op("Relu", inputs)
        self.assertIs(tape.nodes[0].graph, graph)

    def test_initializer_with_graph_registers(self):
        graph = ir.Graph([], [], nodes=[], name="test_graph")
        tape = ir.tape.Tape(graph_like=graph)
        tensor = ir.tensor([1.0, 2.0], name="w")
        val = tape.initializer(tensor)
        self.assertEqual(val.name, "w")
        self.assertIn("w", graph.initializers)

    def test_initializer_raises_without_name(self):
        tape = ir.tape.Tape()
        tensor = ir.tensor([1.0, 2.0])
        with self.assertRaises(ValueError):
            tape.initializer(tensor)


class TestTapeRepr(unittest.TestCase):
    def test_repr_shows_nodes_and_initializers(self):
        tape = ir.tape.Tape()
        r = repr(tape)
        self.assertIn("Tape", r)
        self.assertIn("nodes=", r)
        self.assertIn("initializers=", r)


class TestTapeUsedOpsets(unittest.TestCase):
    def test_used_opsets_tracked(self):
        tape = ir.tape.Tape()
        inputs = [ir.Value(name="x")]
        _ = tape.op("Add", inputs)
        _ = tape.op("CustomOp", inputs, domain="custom.domain", version=1)
        self.assertIn(("", None), tape.used_opsets)
        self.assertIn(("custom.domain", 1), tape.used_opsets)


class TestBuilder(unittest.TestCase):
    def test_builder_single_output(self):
        builder = Builder()
        x = ir.Value(name="x")
        y = builder.Add(x, x)
        self.assertEqual(len(builder.nodes), 1)
        self.assertEqual(builder.nodes[0].op_type, "Add")
        self.assertIsInstance(y, ir.Value)

    def test_builder_with_named_output(self):
        builder = Builder()
        x = ir.Value(name="x")
        y = builder.Relu(x, _outputs=["relu_out"])
        self.assertEqual(y.name, "relu_out")

    def test_builder_multi_output(self):
        builder = Builder()
        x = ir.Value(name="x")
        results = builder.Split(x, _outputs=3)
        self.assertEqual(len(results), 3)

    def test_builder_multi_output_with_names(self):
        builder = Builder()
        x = ir.Value(name="x")
        o1, o2 = builder.Split(x, _outputs=["a", "b"])
        self.assertEqual(o1.name, "a")
        self.assertEqual(o2.name, "b")

    def test_builder_with_domain_and_version(self):
        builder = Builder()
        x = ir.Value(name="x")
        _ = builder.MyOp(x, _domain="custom", _version=1)
        self.assertEqual(builder.nodes[0].domain, "custom")

    def test_builder_with_kwargs_attributes(self):
        builder = Builder()
        x = ir.Value(name="x")
        _ = builder.Elu(x, alpha=2.0)
        self.assertEqual(builder.nodes[0].attributes["alpha"].as_float(), 2.0)


if __name__ == "__main__":
    unittest.main()
