# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for binary element-wise shape inference."""

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
BOOL = ir.DataType.BOOL
INT64 = ir.DataType.INT64

# All arithmetic binary ops share the same broadcast logic
_ARITHMETIC_OPS = ["Add", "Sub", "Mul", "Div", "Pow"]
_COMPARISON_OPS = ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"]
_LOGICAL_OPS = ["And", "Or", "Xor"]


class BinaryElementwiseTest(unittest.TestCase):
    """Tests for binary element-wise shape inference."""

    @parameterized.parameterized.expand([(op,) for op in _ARITHMETIC_OPS])
    def test_arithmetic_broadcast(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 1, 5]), ts(FLOAT, [1, 4, 5])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, 5])])

    @parameterized.parameterized.expand([(op,) for op in _ARITHMETIC_OPS])
    def test_arithmetic_symbolic(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, ["batch", 128]), ts(FLOAT, [1, 128])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 128])])

    @parameterized.parameterized.expand([(op,) for op in _ARITHMETIC_OPS])
    def test_arithmetic_missing_shape(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT), ts(FLOAT, [2, 3])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT)])

    @parameterized.parameterized.expand([(op,) for op in _COMPARISON_OPS])
    def test_comparison_output_bool(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 4]), ts(FLOAT, [3, 4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(BOOL, [3, 4])])

    @parameterized.parameterized.expand([(op,) for op in _COMPARISON_OPS])
    def test_comparison_broadcast(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, ["batch", 1]), ts(FLOAT, [1, 128])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(BOOL, ["batch", 128])])

    @parameterized.parameterized.expand([(op,) for op in _LOGICAL_OPS])
    def test_logical_output_bool(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(BOOL, [2, 3]), ts(BOOL, [2, 3])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(BOOL, [2, 3])])

    def test_mod(self):
        actual = run_shape_inference(
            "",
            "Mod",
            [ts(INT64, [3, 4]), ts(INT64, [3, 4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [3, 4])])

    def test_add_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Add", [], opset_version=17)

    def test_add_none_input(self):
        v = ir.Value(name="a", type=ir.TensorType(FLOAT), shape=ir.Shape([3]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Add",
                [v, None],
                opset_version=17,
            )


if __name__ == "__main__":
    unittest.main()
