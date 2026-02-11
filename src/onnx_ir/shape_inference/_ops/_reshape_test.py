# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Reshape shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops import _testing

FLOAT = ir.DataType.FLOAT


class ReshapeTest(unittest.TestCase):
    def _run(self, input_ts, shape_data, expected):
        data = ir.Value(name="data", shape=input_ts.shape, type=input_ts.type)
        shape_val = _testing.const_value(shape_data, name="shape")
        actual = _testing.run_shape_inference_with_values(
            "", "Reshape", [data, shape_val], opset_version=17,
        )
        self.assertEqual(actual, expected)

    @parameterized.parameterized.expand([
        ("simple", [2, 3, 4], [6, 4], [_testing.ts(FLOAT, [6, 4])]),
        ("with_neg_one", [2, 3, 4], [2, -1], [_testing.ts(FLOAT, [2, 12])]),
        ("with_zero", [2, 3, 4], [0, -1], [_testing.ts(FLOAT, [2, 12])]),
        ("flatten_all", [2, 3, 4], [-1], [_testing.ts(FLOAT, [24])]),
        ("add_dim", [6], [2, 3], [_testing.ts(FLOAT, [2, 3])]),
    ])
    def test_reshape(self, _name, input_shape, target_shape, expected):
        self._run(_testing.ts(FLOAT, input_shape), target_shape, expected)

    def test_dynamic_shape(self):
        """When shape input is not const, output rank can still be inferred."""
        data = ir.Value(
            name="data", shape=ir.Shape([2, 3]),
            type=ir.TensorType(FLOAT),
        )
        shape_val = ir.Value(
            name="shape", shape=ir.Shape([3]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        actual = _testing.run_shape_inference_with_values(
            "", "Reshape", [data, shape_val], opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)


if __name__ == "__main__":
    unittest.main()
