# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
import onnx

import onnx_ir as ir
from onnx_ir.passes.common import _c_api_utils, shape_inference


class TestShapeInferencePass(unittest.TestCase):
    def test_pass_is_in_place(self):
        self.assertTrue(shape_inference.ShapeInferencePass().in_place)

    def test_pass(self):
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

        output = tape.op("Add", inputs=inputs)

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[output],
                nodes=tape.nodes,
                opset_imports={"": 20},
            ),
            ir_version=10,
        )
        self.assertIsNone(output.shape)
        self.assertIsNone(output.dtype)

        # Perform shape inference
        result = shape_inference.ShapeInferencePass()(model)
        self.assertTrue(result.modified)
        self.assertEqual(result.model.graph.node(0).outputs[0].shape, ir.Shape((1, 2)))
        self.assertEqual(result.model.graph.node(0).outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(result.model.graph.outputs[0].shape, ir.Shape((1, 2)))
        self.assertEqual(result.model.graph.outputs[0].dtype, ir.DataType.FLOAT)

    def test_pass_preserves_inferred_container_types(self):
        tensor_type = ir.TensorType(ir.DataType.FLOAT)
        sequence_type = ir.SequenceType(tensor_type)
        for input_type in (tensor_type, sequence_type):
            with self.subTest(input_type=input_type):
                input_value = ir.Value(name="x", type=input_type, shape=ir.Shape([2]))
                tape = ir.tape.Tape()
                output = tape.op("Optional", inputs=[input_value])
                model = ir.Model(
                    ir.Graph(
                        inputs=[input_value],
                        outputs=[output],
                        nodes=tape.nodes,
                        opset_imports={"": 18},
                        name="optional_graph",
                    ),
                    ir_version=10,
                )
                self.assertIsNone(output.type)

                result = shape_inference.ShapeInferencePass()(model)

                self.assertTrue(result.modified)
                self.assertEqual(output.type, ir.OptionalType(input_type))
                self.assertFalse(shape_inference.ShapeInferencePass()(model).modified)
                onnx.checker.check_model(ir.to_proto(model), full_check=True)

    def test_pass_preserves_loop_carried_sequence_type(self):
        proto = onnx.parser.parse_model("""
            <ir_version: 10, opset_import: ["" : 18]>
            graph (float[2] x, int64 trip_count, bool cond) => (float[?] y) {
                empty = SequenceEmpty <dtype: int = 1> ()
                sequence = Loop (trip_count, cond, empty) <
                    body: graph = body (
                        int64 iteration, bool cond_in, seq(float[]) sequence_in
                    ) => (bool cond_out, seq(float[]) sequence_out) {
                        cond_out = Identity (cond_in)
                        sequence_out = SequenceInsert (sequence_in, x)
                    }
                >
                y = ConcatFromSequence <axis: int = 0> (sequence)
            }
        """)
        model = ir.from_proto(proto)
        empty = model.graph.node(0).outputs[0]
        sequence = model.graph.node(1).outputs[0]
        self.assertIsNone(empty.type)
        self.assertIsNone(sequence.type)

        result = shape_inference.ShapeInferencePass()(model)

        self.assertTrue(result.modified)
        sequence_type = ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
        self.assertEqual(empty.type, sequence_type)
        self.assertEqual(sequence.type, sequence_type)
        body = model.graph.node(1).attributes["body"].as_graph()
        self.assertEqual(body.outputs[1].type, sequence_type)
        onnx.checker.check_model(ir.to_proto(model), full_check=True)

    def test_pass_with_initializers(self):
        # _BIG_TENSOR_SIZE_LIMIT is in bytes, but we create big_dim as size
        # of a tensor. This is fine as we just need to create a big tensor whose size
        # passes _BIG_TENSOR_SIZE_LIMIT
        big_dim = _c_api_utils._BIG_TENSOR_SIZE_LIMIT * 2  # pylint: disable=protected-access
        inputs = [
            ir.Value(
                name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
            ir.Value(
                name="input_b",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((big_dim, 1)),
                const_value=ir.tensor([[42]] * big_dim, dtype=ir.DataType.FLOAT),
            ),
        ]

        tape = ir.tape.Tape()

        # Shape and type are not explicitly set for the initializer but it should still work
        initializer = ir.Value(
            name="initializer", const_value=ir.tensor([[2, 3]], dtype=ir.DataType.FLOAT)
        )
        val_add = tape.op("Add", inputs=inputs)
        val_mul = tape.op("Mul", inputs=[val_add, initializer])

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[val_mul],
                nodes=tape.nodes,
                opset_imports={"": 20},
                initializers=[inputs[1], initializer],
            ),
            ir_version=10,
        )

        self.assertIsNone(val_add.shape)
        self.assertIsNone(val_add.dtype)
        self.assertIsNone(val_mul.shape)
        self.assertIsNone(val_mul.dtype)
        self.assertIsNone(initializer.shape)
        self.assertIsNone(initializer.dtype)

        # Perform shape inference
        result = shape_inference.ShapeInferencePass()(model)
        self.assertTrue(result.modified)
        self.assertEqual(result.model.graph.node(0).outputs[0].shape, ir.Shape((big_dim, 2)))
        self.assertEqual(result.model.graph.node(0).outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(result.model.graph.node(1).outputs[0].shape, ir.Shape((big_dim, 2)))
        self.assertEqual(result.model.graph.node(1).outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(
            result.model.graph.initializers["initializer"].shape, ir.Shape((1, 2))
        )
        self.assertEqual(
            result.model.graph.initializers["initializer"].dtype, ir.DataType.FLOAT
        )
        self.assertEqual(result.model.graph.outputs[0].shape, ir.Shape((big_dim, 2)))
        self.assertEqual(result.model.graph.outputs[0].dtype, ir.DataType.FLOAT)

        # Check that the initializer correctly appears in the result
        self.assertEqual(len(result.model.graph.inputs), 2)
        self.assertEqual(len(result.model.graph.initializers), 2)
        np.testing.assert_array_equal(
            result.model.graph.initializers["input_b"].const_value.numpy(),
            np.array([[42]] * big_dim, dtype=np.float32),
            strict=True,
        )
        self.assertEqual(
            result.model.graph.initializers["input_b"].const_value.dtype,
            ir.DataType.FLOAT,
        )
        np.testing.assert_array_equal(
            result.model.graph.initializers["initializer"].const_value.numpy(),
            np.array([[2.0, 3.0]], dtype=np.float32),
            strict=True,
        )
        self.assertEqual(
            result.model.graph.initializers["initializer"].const_value.dtype,
            ir.DataType.FLOAT,
        )

    def test_pass_with_initializers_without_const_value(self):
        """ShapeInferencePass should handle initializers with const_value=None.

        This is a common pattern where shape inference is needed before weights
        are loaded. The initializer has shape and type set but no tensor data.
        """
        float_type = ir.TensorType(ir.DataType.FLOAT)

        # Input
        input_x = ir.Value(name="X", shape=ir.Shape([2, 3]), type=float_type)

        # Initializer (weight) — no const_value: weights not loaded yet
        weight = ir.Value(name="W", shape=ir.Shape([3, 4]), type=float_type)

        # Node: output = MatMul(input_x, weight)
        node = ir.Node("", "MatMul", inputs=[input_x, weight], num_outputs=1)
        output = node.outputs[0]
        output.name = "Y"

        graph = ir.Graph(
            inputs=[input_x],
            outputs=[output],
            nodes=[node],
            initializers=[weight],
            name="test_graph",
            opset_imports={"": 20},
        )
        model = ir.Model(graph, ir_version=10)

        self.assertIsNone(weight.const_value)
        self.assertIsNone(output.type)

        # Perform shape inference — should not raise AssertionError
        result = shape_inference.ShapeInferencePass()(model)
        self.assertTrue(result.modified)
        self.assertEqual(output.shape, ir.Shape([2, 4]))
        self.assertEqual(output.dtype, ir.DataType.FLOAT)

        # Verify the model is restored properly
        self.assertEqual(len(result.model.graph.inputs), 1)
        self.assertNotIn(weight, result.model.graph.inputs)
        self.assertEqual(len(result.model.graph.initializers), 1)
        self.assertIn("W", result.model.graph.initializers)
        self.assertIs(result.model.graph.initializers["W"], weight)
        self.assertIsNone(result.model.graph.initializers["W"].const_value)


if __name__ == "__main__":
    unittest.main()
