# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Attention and RotaryEmbedding shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class AttentionTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 64])])

    def test_q_shape_propagated(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [2, 8, 32]), ts(FLOAT, [2, 8, 32]), ts(FLOAT, [2, 8, 32])],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 32])])

    def test_missing_q_shape_falls_back_to_v(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 64])])

    def test_missing_all_shapes(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT), ts(FLOAT), ts(FLOAT)],
            opset_version=23,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Attention", [], opset_version=23)

    def test_none_input(self):
        v = ir.Value(name="q", type=ir.TensorType(FLOAT), shape=ir.Shape([1, 10, 64]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Attention",
                [v, None, None],
                opset_version=23,
            )

    def test_present_key_value_outputs(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[2], ts(FLOAT, [1, 10, 64]))


class RotaryEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "RotaryEmbedding",
            [ts(FLOAT, [1, 10, 64])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 64])])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "RotaryEmbedding",
            [ts(FLOAT)],
            opset_version=24,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "RotaryEmbedding", [], opset_version=24)

    def test_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "RotaryEmbedding",
                [None],
                opset_version=24,
            )

    def test_rotary_embedding_with_position_ids_two_outputs(self):
        input_val = ir.Value(
            name="input", shape=ir.Shape([1, 10, 64]), type=ir.TensorType(FLOAT)
        )
        position_ids = ir.Value(
            name="position_ids", shape=ir.Shape([1, 10]), type=ir.TensorType(INT64)
        )
        actual = run_shape_inference_with_values(
            "",
            "RotaryEmbedding",
            [input_val, position_ids],
            opset_version=24,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[1], ts(INT64, [1, 10]))


class AttentionSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_dims(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", "S", "D"]),
                ts(FLOAT, ["B", "S", "D"]),
                ts(FLOAT, ["B", "S", "D"]),
            ],
            opset_version=23,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[2], ir.SymbolicDim)

    def test_symbolic_with_concrete_embed_dim(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", "S", 64]),
                ts(FLOAT, ["B", "S", 64]),
                ts(FLOAT, ["B", "S", 64]),
            ],
            opset_version=23,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[2], 64)

    def test_4d_concrete(self):
        """4D inputs: [batch, num_heads, seq_len, head_size]."""
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, [2, 8, 10, 64]),
                ts(FLOAT, [2, 8, 10, 64]),
                ts(FLOAT, [2, 8, 10, 64]),
            ],
            opset_version=23,
            num_outputs=3,
        )
        self.assertEqual(list(actual[0].shape), [2, 8, 10, 64])
        self.assertEqual(list(actual[1].shape), [2, 8, 10, 64])
        self.assertEqual(list(actual[2].shape), [2, 8, 10, 64])

    def test_4d_symbolic(self):
        """4D inputs with symbolic batch and seq dims."""
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", 8, "S", 64]),
                ts(FLOAT, ["B", 8, "Sk", 64]),
                ts(FLOAT, ["B", 8, "Sk", 64]),
            ],
            opset_version=23,
            num_outputs=3,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 8)
        self.assertIsInstance(actual[0].shape[2], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[3], 64)
        # present_key follows K shape
        self.assertEqual(actual[1].shape.rank(), 4)
        # present_value follows V shape
        self.assertEqual(actual[2].shape.rank(), 4)


if __name__ == "__main__":
    unittest.main()
