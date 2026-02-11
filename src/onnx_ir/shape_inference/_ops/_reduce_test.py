# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Reduce* shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops import _testing

FLOAT = ir.DataType.FLOAT

_REDUCE_OPS = [
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
]


class ReduceTest(unittest.TestCase):
    """Tests for Reduce* shape inference."""

    @parameterized.parameterized.expand([(op,) for op in _REDUCE_OPS])
    def test_keepdims_with_axes_attr(self, op):
        actual = _testing.run_shape_inference(
            "",
            op,
            [_testing.ts(FLOAT, ["batch", 10, 20])],
            {
                "axes": ir.Attr("axes", ir.AttributeType.INTS, [1]),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1),
            },
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, ["batch", 1, 20])])

    @parameterized.parameterized.expand([(op,) for op in _REDUCE_OPS])
    def test_no_keepdims(self, op):
        actual = _testing.run_shape_inference(
            "",
            op,
            [_testing.ts(FLOAT, [3, 4, 5])],
            {
                "axes": ir.Attr("axes", ir.AttributeType.INTS, [1]),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 0),
            },
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, [3, 5])])

    def test_negative_axis(self):
        actual = _testing.run_shape_inference(
            "",
            "ReduceSum",
            [_testing.ts(FLOAT, [3, 4, 5])],
            {
                "axes": ir.Attr("axes", ir.AttributeType.INTS, [-1]),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1),
            },
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, [3, 4, 1])])

    def test_reduce_all_axes(self):
        actual = _testing.run_shape_inference(
            "",
            "ReduceSum",
            [_testing.ts(FLOAT, [3, 4, 5])],
            {
                "axes": ir.Attr("axes", ir.AttributeType.INTS, [0, 1, 2]),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 0),
            },
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, [])])

    def test_reduce_all_keepdims(self):
        actual = _testing.run_shape_inference(
            "",
            "ReduceSum",
            [_testing.ts(FLOAT, [3, 4, 5])],
            {
                "axes": ir.Attr("axes", ir.AttributeType.INTS, [0, 1, 2]),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1),
            },
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, [1, 1, 1])])

    def test_missing_input_shape(self):
        actual = _testing.run_shape_inference(
            "",
            "ReduceSum",
            [_testing.ts(FLOAT)],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, [0])},
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT)])

    def test_multiple_axes(self):
        """From ONNX test_reduce_op_shape_2_axis."""
        actual = _testing.run_shape_inference(
            "",
            "ReduceMean",
            [_testing.ts(FLOAT, [24, 4, 11])],
            {
                "axes": ir.Attr("axes", ir.AttributeType.INTS, [1, 2]),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1),
            },
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, [24, 1, 1])])

    def test_noop_with_empty_axes(self):
        """noop_with_empty_axes=1 + empty axes → identity."""
        actual = _testing.run_shape_inference(
            "",
            "ReduceSum",
            [_testing.ts(FLOAT, [3, 4, 5])],
            {
                "axes": ir.Attr("axes", ir.AttributeType.INTS, []),
                "noop_with_empty_axes": ir.Attr("noop_with_empty_axes", ir.AttributeType.INT, 1),
            },
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, [3, 4, 5])])

    def test_default_no_axes_reduces_all(self):
        """No axes attribute + noop_with_empty_axes=0 (default) → reduce all."""
        actual = _testing.run_shape_inference(
            "",
            "ReduceSum",
            [_testing.ts(FLOAT, [3, 4, 5])],
            {
                "axes": ir.Attr("axes", ir.AttributeType.INTS, []),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 0),
            },
            opset_version=13,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, [])])

    def test_reduce_opset18_axes_input(self):
        """Opset 18+: axes come from input[1]."""
        data = ir.Value(name="data", shape=ir.Shape([24, 4, 11]), type=ir.TensorType(FLOAT))
        axes = _testing.const_value([1, 2], "axes")
        actual = _testing.run_shape_inference_with_values(
            "",
            "ReduceMean",
            [data, axes],
            {"keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1)},
            opset_version=18,
        )
        self.assertEqual(actual, [_testing.ts(FLOAT, [24, 1, 1])])


if __name__ == "__main__":
    unittest.main()
