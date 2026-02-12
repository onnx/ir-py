# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Range shape inference."""

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
INT64 = ir.DataType.INT64


class RangeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("float", FLOAT),
            ("int64", INT64),
        ]
    )
    def test_basic(self, _name, dtype):
        actual = run_shape_inference(
            "", "Range", [ts(dtype, []), ts(dtype, []), ts(dtype, [])], opset_version=21
        )
        self.assertEqual(actual, [ts(dtype, ["_d0"])])

    def test_none_input_raises(self):
        v1 = ir.Value(name="limit", type=ir.TensorType(FLOAT), shape=ir.Shape([]))
        v2 = ir.Value(name="delta", type=ir.TensorType(FLOAT), shape=ir.Shape([]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "Range", [None, v1, v2], opset_version=21)


if __name__ == "__main__":
    unittest.main()
