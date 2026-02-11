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


if __name__ == "__main__":
    unittest.main()
