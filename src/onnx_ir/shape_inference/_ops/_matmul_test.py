# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MatMul shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class MatMulTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("2d", [3, 4], [4, 5], [3, 5]),
            ("batch", [2, 3, 4], [2, 4, 5], [2, 3, 5]),
            ("broadcast_batch", [1, 3, 4], [2, 4, 5], [2, 3, 5]),
            ("1d_1d", [4], [4], []),
            ("2d_1d", [3, 4], [4], [3]),
            ("1d_2d", [4], [4, 5], [5]),
            ("symbolic", ["batch", "M", 64], ["batch", 64, "N"], ["batch", "M", "N"]),
            ("high_rank", [2, 3, 4, 5], [5, 6], [2, 3, 4, 6]),
        ]
    )
    def test_matmul(self, _name, shape_a, shape_b, expected_shape):
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT, shape_a), ts(FLOAT, shape_b)],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_symbolic_dims(self):
        """MatMul with symbolic: ["M", "K"] @ ["K", "N"] → ["M", "N"], both SymbolicDim."""
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT, ["M", "K"]), ts(FLOAT, ["K", "N"])],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)
        self.assertIsInstance(result.shape[0], ir.SymbolicDim)
        self.assertIsInstance(result.shape[1], ir.SymbolicDim)

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT), ts(FLOAT, [4, 5])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_matmul_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "MatMul", [], opset_version=17)

    def test_matmul_none_input(self):
        v = ir.Value(name="a", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "MatMul",
                [v, None],
                opset_version=17,
            )

    def test_matmul_1d_times_2d(self):
        """1D @ 2D: [K] @ [K, N] → [N]."""
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT, [4]), ts(FLOAT, [4, 5])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [5])])


INT8 = ir.DataType.INT8
INT32 = ir.DataType.INT32
UINT8 = ir.DataType.UINT8


class MatMulIntegerTest(unittest.TestCase):
    def test_matmul_integer_basic(self):
        actual = run_shape_inference(
            "",
            "MatMulInteger",
            [ts(INT8, [3, 4]), ts(INT8, [4, 5])],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [3, 5])])

    def test_matmul_integer_batch(self):
        actual = run_shape_inference(
            "",
            "MatMulInteger",
            [ts(INT8, [2, 3, 4]), ts(INT8, [2, 4, 5])],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [2, 3, 5])])


class QLinearMatMulTest(unittest.TestCase):
    def test_qlinear_matmul_basic(self):
        actual = run_shape_inference(
            "",
            "QLinearMatMul",
            [
                ts(UINT8, [3, 4]),  # a
                ts(FLOAT, []),  # a_scale
                ts(UINT8, []),  # a_zero_point
                ts(UINT8, [4, 5]),  # b
                ts(FLOAT, []),  # b_scale
                ts(UINT8, []),  # b_zero_point
                ts(FLOAT, []),  # y_scale
                ts(UINT8, []),  # y_zero_point
            ],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(UINT8, [3, 5])])

    def test_qlinear_matmul_int8_output(self):
        actual = run_shape_inference(
            "",
            "QLinearMatMul",
            [
                ts(INT8, [3, 4]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [4, 5]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),  # y_zero_point dtype determines output
            ],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT8, [3, 5])])


if __name__ == "__main__":
    unittest.main()
