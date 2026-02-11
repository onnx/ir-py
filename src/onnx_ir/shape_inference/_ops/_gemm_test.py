# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gemm shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT


class GemmTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        ("basic", [3, 4], [4, 5], 0, 0, [3, 5]),
        ("transA", [4, 3], [4, 5], 1, 0, [3, 5]),
        ("transB", [3, 4], [5, 4], 0, 1, [3, 5]),
        ("both_trans", [4, 3], [5, 4], 1, 1, [3, 5]),
        ("symbolic", ["M", 64], [64, "N"], 0, 0, ["M", "N"]),
    ])
    def test_gemm(self, _name, shape_a, shape_b, transA, transB, expected_shape):
        attrs = {}
        if transA:
            attrs["transA"] = ir.Attr("transA", ir.AttributeType.INT, transA)
        if transB:
            attrs["transB"] = ir.Attr("transB", ir.AttributeType.INT, transB)
        actual = run_shape_inference(
            "", "Gemm",
            [ts(FLOAT, shape_a), ts(FLOAT, shape_b)],
            attrs or None,
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])


if __name__ == "__main__":
    unittest.main()
