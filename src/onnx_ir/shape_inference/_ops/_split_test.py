# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Split shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class SplitTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "equal_split_3",
                [6, 4],
                0,
                3,
                [ts(FLOAT, [2, 4])] * 3,
            ),
            (
                "equal_split_2",
                [10, 4],
                0,
                2,
                [ts(FLOAT, [5, 4])] * 2,
            ),
        ]
    )
    def test_equal_split(self, _name, shape, axis, num_outputs, expected):
        actual = run_shape_inference(
            "",
            "Split",
            [ts(FLOAT, shape)],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, axis)},
            opset_version=17,
            num_outputs=num_outputs,
        )
        self.assertEqual(actual, expected)

    def test_explicit_split_attr(self):
        actual = run_shape_inference(
            "",
            "Split",
            [ts(FLOAT, [10, 4])],
            {
                "axis": ir.Attr("axis", ir.AttributeType.INT, 0),
                "split": ir.Attr("split", ir.AttributeType.INTS, [3, 7]),
            },
            opset_version=11,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4]), ts(FLOAT, [7, 4])])

    def test_axis_1(self):
        actual = run_shape_inference(
            "",
            "Split",
            [ts(FLOAT, [4, 6])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
            opset_version=17,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 3])] * 2)

    def test_negative_axis(self):
        """From ONNX test_split_negative_axis."""
        actual = run_shape_inference(
            "",
            "Split",
            [ts(FLOAT, [2, 4])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, -1)},
            opset_version=17,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 2])] * 2)

    def test_uneven_split(self):
        """From ONNX test_split_uneven_split_2d: 8 / 3 → [3, 3, 2]."""
        actual = run_shape_inference(
            "",
            "Split",
            [ts(FLOAT, [8, 2])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=18,
            num_outputs=3,
        )
        self.assertEqual(
            actual,
            [ts(FLOAT, [3, 2]), ts(FLOAT, [3, 2]), ts(FLOAT, [2, 2])],
        )

    def test_split_input_opset13(self):
        """Opset 13+: split sizes come from input[1]."""
        data = ir.Value(name="data", shape=ir.Shape([2, 4]), type=ir.TensorType(FLOAT))
        split = const_value([3, 1], "split")
        actual = run_shape_inference_with_values(
            "",
            "Split",
            [data, split],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, -1)},
            opset_version=13,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3]), ts(FLOAT, [2, 1])])

    def test_unknown_split_dim(self):
        """When split axis dim is symbolic, can't compute sizes."""
        actual = run_shape_inference(
            "",
            "Split",
            [ts(FLOAT, ["N", 4])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
            num_outputs=2,
        )
        # Shape should be None since we can't determine split sizes
        self.assertIsNone(actual[0].shape)


if __name__ == "__main__":
    unittest.main()
