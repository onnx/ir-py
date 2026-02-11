# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Where shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import ShapeInferenceError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
BOOL = ir.DataType.BOOL


class WhereTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "same_shape",
                [3, 4],
                [3, 4],
                [3, 4],
                [3, 4],
            ),
            (
                "broadcast_condition",
                [1, 4],
                [3, 1],
                [3, 4],
                [3, 4],
            ),
            (
                "broadcast_all",
                [1],
                [3, 1],
                [1, 4],
                [3, 4],
            ),
            (
                "symbolic",
                ["batch", 1],
                [1, 128],
                ["batch", 128],
                ["batch", 128],
            ),
        ]
    )
    def test_where(self, _name, cond_shape, x_shape, y_shape, expected_shape):
        actual = run_shape_inference(
            "",
            "Where",
            [ts(BOOL, cond_shape), ts(FLOAT, x_shape), ts(FLOAT, y_shape)],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "Where",
            [ts(BOOL, [3, 4]), ts(FLOAT), ts(FLOAT, [3, 4])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_where_no_inputs(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference("", "Where", [ts(BOOL, [3])], opset_version=17)

    def test_where_none_input(self):
        cond = ir.Value(name="cond", type=ir.TensorType(BOOL), shape=ir.Shape([3]))
        actual = run_shape_inference_with_values(
            "",
            "Where",
            [cond, None, None],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)


if __name__ == "__main__":
    unittest.main()
