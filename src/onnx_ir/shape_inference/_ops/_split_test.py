# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Split shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT


class SplitTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        (
            "equal_split_3",
            [6, 4], 0, 3,
            [ts(FLOAT, [2, 4]), ts(FLOAT, [2, 4]), ts(FLOAT, [2, 4])],
        ),
        (
            "equal_split_2",
            [10, 4], 0, 2,
            [ts(FLOAT, [5, 4]), ts(FLOAT, [5, 4])],
        ),
    ])
    def test_equal_split(self, _name, shape, axis, num_outputs, expected):
        actual = run_shape_inference(
            "", "Split", [ts(FLOAT, shape)],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, axis)},
            opset_version=17, num_outputs=num_outputs,
        )
        self.assertEqual(actual, expected)

    def test_explicit_split_attr(self):
        actual = run_shape_inference(
            "", "Split", [ts(FLOAT, [10, 4])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0),
             "split": ir.Attr("split", ir.AttributeType.INTS, [3, 7])},
            opset_version=11, num_outputs=2,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4]), ts(FLOAT, [7, 4])])

    def test_axis_1(self):
        actual = run_shape_inference(
            "", "Split", [ts(FLOAT, [4, 6])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
            opset_version=17, num_outputs=2,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 3]), ts(FLOAT, [4, 3])])


if __name__ == "__main__":
    unittest.main()
