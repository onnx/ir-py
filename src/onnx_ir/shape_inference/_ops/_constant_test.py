# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Constant shape inference."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class ConstantTest(unittest.TestCase):
    def test_tensor_value(self):
        tensor = ir.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        actual = run_shape_inference(
            "", "Constant", [],
            {"value": ir.Attr("value", ir.AttributeType.TENSOR, tensor)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 2])])

    def test_value_float(self):
        actual = run_shape_inference(
            "", "Constant", [],
            {"value_float": ir.Attr("value_float", ir.AttributeType.FLOAT, 3.14)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [])])

    def test_value_int(self):
        actual = run_shape_inference(
            "", "Constant", [],
            {"value_int": ir.Attr("value_int", ir.AttributeType.INT, 42)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])


if __name__ == "__main__":
    unittest.main()
