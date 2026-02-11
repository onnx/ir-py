# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cast shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
FLOAT16 = ir.DataType.FLOAT16
INT64 = ir.DataType.INT64


class CastTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        ("float_to_half", FLOAT, [3, 4], FLOAT16),
        ("float_to_int64", FLOAT, ["batch", 128], INT64),
        ("int64_to_float", INT64, [2, 3], FLOAT),
    ])
    def test_cast(self, _name, src_dtype, shape, target_dtype):
        actual = run_shape_inference(
            "", "Cast", [ts(src_dtype, shape)],
            {"to": ir.Attr("to", ir.AttributeType.INT, target_dtype)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(target_dtype, shape)])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "", "Cast", [ts(FLOAT)],
            {"to": ir.Attr("to", ir.AttributeType.INT, INT64)},
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, INT64)


if __name__ == "__main__":
    unittest.main()
