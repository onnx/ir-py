# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ConstantOfShape shape inference."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir.shape_inference._ops import _testing

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class ConstantOfShapeTest(unittest.TestCase):
    def test_basic(self):
        shape_val = _testing.const_value([3, 4, 5], name="shape")
        actual = _testing.run_shape_inference_with_values(
            "", "ConstantOfShape", [shape_val], opset_version=17,
        )
        # Default dtype is FLOAT
        self.assertEqual(actual, [_testing.ts(FLOAT, [3, 4, 5])])

    def test_int64_value(self):
        shape_val = _testing.const_value([2, 3], name="shape")
        tensor = ir.Tensor(np.array([0], dtype=np.int64))
        actual = _testing.run_shape_inference_with_values(
            "", "ConstantOfShape", [shape_val],
            {"value": ir.Attr("value", ir.AttributeType.TENSOR, tensor)},
            opset_version=17,
        )
        self.assertEqual(actual, [_testing.ts(INT64, [2, 3])])

    def test_dynamic_shape(self):
        """When shape input is not const, output shape is unknown."""
        shape_val = ir.Value(
            name="shape", shape=ir.Shape([3]),
            type=ir.TensorType(INT64),
        )
        actual = _testing.run_shape_inference_with_values(
            "", "ConstantOfShape", [shape_val], opset_version=17,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)


if __name__ == "__main__":
    unittest.main()
