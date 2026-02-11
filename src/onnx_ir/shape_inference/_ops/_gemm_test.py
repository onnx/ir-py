# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gemm shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import InvalidOpUsageError, ShapeInferenceError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class GemmTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("basic", [7, 5], [5, 11], 0, 0, [7, 11]),
            ("trans_a", [5, 7], [5, 11], 1, 0, [7, 11]),
            ("trans_b", [7, 5], [11, 5], 0, 1, [7, 11]),
            ("both_trans", [5, 7], [11, 5], 1, 1, [7, 11]),
            ("symbolic", ["M", 64], [64, "N"], 0, 0, ["M", "N"]),
        ]
    )
    def test_gemm(self, _name, shape_a, shape_b, trans_a, trans_b, expected_shape):
        attrs = {}
        if trans_a:
            attrs["transA"] = ir.Attr("transA", ir.AttributeType.INT, trans_a)
        if trans_b:
            attrs["transB"] = ir.Attr("transB", ir.AttributeType.INT, trans_b)
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT, shape_a), ts(FLOAT, shape_b)],
            attrs or None,
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_no_bias(self):
        """From ONNX test_gemm_no_bias: C is optional."""
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT, [7, 5]), ts(FLOAT, [5, 11])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [7, 11])])

    def test_with_bias(self):
        """Gemm with bias (C) — does not affect output shape."""
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT, [7, 5]), ts(FLOAT, [5, 11]), ts(FLOAT, [11])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [7, 11])])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "Gemm",
            [ts(FLOAT), ts(FLOAT, [5, 11])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_gemm_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference("", "Gemm", [], opset_version=17)

    def test_gemm_none_input(self):
        v = ir.Value(name="a", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4]))
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "Gemm",
                [v, None],
                opset_version=17,
            )

    def test_gemm_wrong_rank_a(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "Gemm",
                [ts(FLOAT, [2, 3, 4]), ts(FLOAT, [4, 5])],
                opset_version=17,
            )

    def test_gemm_wrong_rank_b(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "Gemm",
                [ts(FLOAT, [2, 3]), ts(FLOAT, [3])],
                opset_version=17,
            )


if __name__ == "__main__":
    unittest.main()
