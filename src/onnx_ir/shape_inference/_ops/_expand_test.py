# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Expand shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import InvalidOpUsageError
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class ExpandTest(unittest.TestCase):
    def _run(self, input_ts, target_shape):
        data = ir.Value(name="data", shape=input_ts.shape, type=input_ts.type)
        shape_val = const_value(target_shape, name="shape")
        return run_shape_inference_with_values(
            "",
            "Expand",
            [data, shape_val],
            opset_version=17,
        )

    @parameterized.parameterized.expand(
        [
            ("basic", [1, 4], [3, 4], [3, 4]),
            ("broadcast_rank", [4], [2, 3, 4], [2, 3, 4]),
            ("broadcast_ones", [3, 1], [3, 4], [3, 4]),
            ("noop", [3, 4], [3, 4], [3, 4]),
            # From ONNX test_expand: (3,1) → (2,3,6)
            ("rank_increase", [3, 1], [2, 3, 6], [2, 3, 6]),
        ]
    )
    def test_expand(self, _name, input_shape, target, expected_shape):
        actual = self._run(ts(FLOAT, input_shape), target)
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_scalar_input(self):
        """From ONNX test_expand_scalar_input: () → (4,8)."""
        actual = self._run(ts(FLOAT, []), [4, 8])
        self.assertEqual(actual, [ts(FLOAT, [4, 8])])

    def test_dynamic_shape(self):
        """When shape input is not const, output shape is unknown."""
        data = ir.Value(name="data", shape=ir.Shape([1, 2, 3]), type=ir.TensorType(FLOAT))
        shape_val = ir.Value(name="shape", type=ir.TensorType(ir.DataType.INT64))
        actual = run_shape_inference_with_values(
            "",
            "Expand",
            [data, shape_val],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_expand_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "Expand",
                [],
                opset_version=17,
            )

    def test_expand_none_input(self):
        shape_val = const_value([3, 4])
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "Expand",
                [None, shape_val],
                opset_version=17,
            )

    def test_expand_no_const_shape(self):
        """When the shape input is not constant, output shape is unknown."""
        data = ir.Value(name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 1]))
        shape_val = ir.Value(
            name="shape", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([2])
        )
        actual = run_shape_inference_with_values(
            "",
            "Expand",
            [data, shape_val],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)


if __name__ == "__main__":
    unittest.main()
