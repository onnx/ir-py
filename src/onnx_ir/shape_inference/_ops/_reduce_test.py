# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Reduce* shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT

_REDUCE_OPS = [
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
]


class ReduceTest(unittest.TestCase):
    """Tests for Reduce* shape inference."""

    @parameterized.parameterized.expand(
        [(op,) for op in _REDUCE_OPS]
    )
    def test_keepdims_with_axes_attr(self, op):
        actual = run_shape_inference(
            "", op,
            [ts(FLOAT, ["batch", 10, 20])],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, [1]),
             "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1)},
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 1, 20])])

    @parameterized.parameterized.expand(
        [(op,) for op in _REDUCE_OPS]
    )
    def test_no_keepdims(self, op):
        actual = run_shape_inference(
            "", op,
            [ts(FLOAT, [3, 4, 5])],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, [1]),
             "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 0)},
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 5])])

    def test_negative_axis(self):
        actual = run_shape_inference(
            "", "ReduceSum",
            [ts(FLOAT, [3, 4, 5])],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, [-1]),
             "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1)},
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, 1])])

    def test_reduce_all_axes(self):
        actual = run_shape_inference(
            "", "ReduceSum",
            [ts(FLOAT, [3, 4, 5])],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, [0, 1, 2]),
             "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 0)},
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [])])

    def test_reduce_all_keepdims(self):
        actual = run_shape_inference(
            "", "ReduceSum",
            [ts(FLOAT, [3, 4, 5])],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, [0, 1, 2]),
             "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1)},
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 1])])

    def test_missing_input_shape(self):
        actual = run_shape_inference(
            "", "ReduceSum",
            [ts(FLOAT)],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, [0])},
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT)])

    def test_multiple_axes(self):
        actual = run_shape_inference(
            "", "ReduceMean",
            [ts(FLOAT, ["batch", 10, 20, 30])],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, [1, 3]),
             "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 0)},
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 20])])


if __name__ == "__main__":
    unittest.main()
