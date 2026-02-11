# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Shape, Size, and Flatten shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class ShapeOpTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        ("rank_3", [3, 4, 5], [3]),
        ("rank_2", [3, 4], [2]),
        ("symbolic", ["batch", "seq", 256], [3]),
        ("scalar", [], [0]),
    ])
    def test_shape(self, _name, input_shape, expected_shape):
        actual = run_shape_inference(
            "", "Shape", [ts(FLOAT, input_shape)], opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, expected_shape)])

    def test_shape_with_start_end(self):
        actual = run_shape_inference(
            "", "Shape", [ts(FLOAT, [2, 3, 4, 5])],
            {"start": ir.Attr("start", ir.AttributeType.INT, 1),
             "end": ir.Attr("end", ir.AttributeType.INT, 3)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [2])])


class SizeTest(unittest.TestCase):
    def test_size(self):
        actual = run_shape_inference(
            "", "Size", [ts(FLOAT, [3, 4, 5])], opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])

    def test_size_symbolic(self):
        actual = run_shape_inference(
            "", "Size", [ts(FLOAT, ["batch", 128])], opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])


class FlattenTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        ("default_axis", [2, 3, 4], None, [2, 12]),
        ("axis_0", [2, 3, 4], 0, [1, 24]),
        ("axis_2", [2, 3, 4], 2, [6, 4]),
        ("axis_3", [2, 3, 4], 3, [24, 1]),
    ])
    def test_flatten(self, _name, input_shape, axis, expected_shape):
        attrs = {}
        if axis is not None:
            attrs["axis"] = ir.Attr("axis", ir.AttributeType.INT, axis)
        actual = run_shape_inference(
            "", "Flatten", [ts(FLOAT, input_shape)], attrs or None, opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])


if __name__ == "__main__":
    unittest.main()
