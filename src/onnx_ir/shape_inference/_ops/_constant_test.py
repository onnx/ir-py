# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Constant shape inference."""

from __future__ import annotations

import unittest

import numpy as np
import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64
STRING = ir.DataType.STRING


class ConstantTest(unittest.TestCase):
    def test_tensor_value(self):
        tensor = ir.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        actual = run_shape_inference(
            "",
            "Constant",
            [],
            {"value": ir.Attr("value", ir.AttributeType.TENSOR, tensor)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 2])])

    @parameterized.parameterized.expand(
        [
            ("value_float", "value_float", ir.AttributeType.FLOAT, 3.14, FLOAT, []),
            ("value_int", "value_int", ir.AttributeType.INT, 42, INT64, []),
            ("value_string", "value_string", ir.AttributeType.STRING, "hello", STRING, []),
            (
                "value_floats",
                "value_floats",
                ir.AttributeType.FLOATS,
                [1.0, 2.0, 3.0],
                FLOAT,
                [3],
            ),
            ("value_ints", "value_ints", ir.AttributeType.INTS, [10, 20], INT64, [2]),
            (
                "value_strings",
                "value_strings",
                ir.AttributeType.STRINGS,
                ["a", "b"],
                STRING,
                [2],
            ),
        ]
    )
    def test_scalar_and_vector(self, _name, attr_name, attr_type, value, dtype, shape):
        actual = run_shape_inference(
            "",
            "Constant",
            [],
            {attr_name: ir.Attr(attr_name, attr_type, value)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(dtype, shape)])


if __name__ == "__main__":
    unittest.main()
