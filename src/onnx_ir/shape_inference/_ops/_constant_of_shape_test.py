# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ConstantOfShape shape inference."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class ConstantOfShapeTest(unittest.TestCase):
    def test_basic(self):
        shape_val = const_value([3, 4, 5], name="shape")
        actual = run_shape_inference_with_values(
            "",
            "ConstantOfShape",
            [shape_val],
            opset_version=17,
        )
        # Default dtype is FLOAT
        self.assertEqual(actual, [ts(FLOAT, [3, 4, 5])])

    def test_int64_value(self):
        shape_val = const_value([2, 3], name="shape")
        tensor = ir.Tensor(np.array([0], dtype=np.int64))
        actual = run_shape_inference_with_values(
            "",
            "ConstantOfShape",
            [shape_val],
            {"value": ir.Attr("value", ir.AttributeType.TENSOR, tensor)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [2, 3])])

    def test_dynamic_shape(self):
        """When shape input is not const, output shape is unknown."""
        shape_val = ir.Value(
            name="shape",
            shape=ir.Shape([3]),
            type=ir.TensorType(INT64),
        )
        actual = run_shape_inference_with_values(
            "",
            "ConstantOfShape",
            [shape_val],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_scalar_output(self):
        """From ONNX test_constantofshape_without_input_shape_scalar: shape=[] → scalar."""
        shape_val = const_value([], name="shape")
        actual = run_shape_inference_with_values(
            "",
            "ConstantOfShape",
            [shape_val],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [])])

    def test_zero_size_tensor(self):
        """From ONNX test_constantofshape_with_shape_zero: shape=[0] → (0,)."""
        shape_val = const_value([0], name="shape")
        actual = run_shape_inference_with_values(
            "",
            "ConstantOfShape",
            [shape_val],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [0])])

    def test_constant_of_shape_no_inputs(self):
        actual = run_shape_inference_with_values(
            "",
            "ConstantOfShape",
            [],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_constant_of_shape_none_input(self):
        actual = run_shape_inference_with_values(
            "",
            "ConstantOfShape",
            [None],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)


if __name__ == "__main__":
    unittest.main()
