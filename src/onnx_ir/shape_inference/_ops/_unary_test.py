# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for unary element-wise shape inference."""

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

    def test_relu_symbolic_dims(self):
        """Relu on ["N", "C"] → same shape ["N", "C"]."""
        actual = run_shape_inference(
            "",
            "Relu",
            [ts(FLOAT, ["N", "C"])],
            opset_version=20,
        )
        self.assertEqual(actual[0].shape, ir.Shape(["N", "C"]))
        self.assertEqual(actual[0].type.dtype, FLOAT)

    @parameterized.parameterized.expand(
        [
            ("not", "Not", BOOL, [3, 4], BOOL),
            ("isnan", "IsNaN", FLOAT, [2, 3], BOOL),
            ("isinf", "IsInf", FLOAT, [4, 5], BOOL),
        ]
    )
    def test_bool_output(self, _name, op_type, input_dtype, input_shape, output_dtype):
        actual = run_shape_inference(
            "", op_type, [ts(input_dtype, input_shape)], opset_version=17
        )
        self.assertEqual(actual, [ts(output_dtype, input_shape)])

    def test_unary_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Abs", [], opset_version=17)

    def test_unary_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Relu",
                [None],
                opset_version=17,
            )

    def test_logical_unary_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Not", [], opset_version=17)

    def test_logical_unary_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "IsNaN",
                [None],
                opset_version=17,
            )


if __name__ == "__main__":
    unittest.main()
