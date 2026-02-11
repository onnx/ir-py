# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for unary element-wise shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import InvalidOpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
BOOL = ir.DataType.BOOL

_UNARY_OPS = [
    "Identity",
    "Neg",
    "Abs",
    "Ceil",
    "Floor",
    "Round",
    "Reciprocal",
    "Sqrt",
    "Exp",
    "Log",
    "Sigmoid",
    "Relu",
    "Tanh",
    "Erf",
    "Sign",
    "Sin",
    "Cos",
]


class UnaryTest(unittest.TestCase):
    """Tests for unary passthrough shape inference."""

    @parameterized.parameterized.expand([(op,) for op in _UNARY_OPS])
    def test_shape_passthrough(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, ["batch", 128])],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 128])])

    @parameterized.parameterized.expand([(op,) for op in _UNARY_OPS])
    def test_concrete_shape(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 4, 5])],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, 5])])

    @parameterized.parameterized.expand([(op,) for op in _UNARY_OPS])
    def test_missing_shape(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT)],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT)])

    def test_not_output_bool(self):
        actual = run_shape_inference(
            "",
            "Not",
            [ts(BOOL, [3, 4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(BOOL, [3, 4])])

    def test_isnan_output_bool(self):
        actual = run_shape_inference(
            "",
            "IsNaN",
            [ts(FLOAT, [2, 3])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(BOOL, [2, 3])])

    def test_unary_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference("", "Abs", [], opset_version=17)

    def test_unary_none_input(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "Relu",
                [None],
                opset_version=17,
            )

    def test_logical_unary_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference("", "Not", [], opset_version=17)

    def test_logical_unary_none_input(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "IsNaN",
                [None],
                opset_version=17,
            )


if __name__ == "__main__":
    unittest.main()
