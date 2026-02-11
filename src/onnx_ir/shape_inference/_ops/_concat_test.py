# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Concat shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT


class ConcatTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        (
            "axis_1",
            [ts(FLOAT, [2, 3]), ts(FLOAT, [2, 5])],
            1,
            [ts(FLOAT, [2, 8])],
        ),
        (
            "symbolic_batch",
            [ts(FLOAT, ["batch", 3]), ts(FLOAT, ["batch", 5])],
            1,
            [ts(FLOAT, ["batch", 8])],
        ),
        (
            "axis_0",
            [ts(FLOAT, [2, 4]), ts(FLOAT, [3, 4])],
            0,
            [ts(FLOAT, [5, 4])],
        ),
        (
            "negative_axis",
            [ts(FLOAT, [2, 3]), ts(FLOAT, [2, 5])],
            -1,
            [ts(FLOAT, [2, 8])],
        ),
        (
            "three_inputs",
            [ts(FLOAT, [2, 3]), ts(FLOAT, [2, 4]), ts(FLOAT, [2, 5])],
            1,
            [ts(FLOAT, [2, 12])],
        ),
    ])
    def test_concat(self, _name, inputs, axis, expected):
        actual = run_shape_inference(
            "", "Concat", inputs,
            {"axis": ir.Attr("axis", ir.AttributeType.INT, axis)},
            opset_version=17,
        )
        self.assertEqual(actual, expected)

    def test_missing_input_shape(self):
        actual = run_shape_inference(
            "", "Concat",
            [ts(FLOAT, [2, 3]), ts(FLOAT)],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        # dtype should still propagate even if shape is unknown
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)


if __name__ == "__main__":
    unittest.main()
