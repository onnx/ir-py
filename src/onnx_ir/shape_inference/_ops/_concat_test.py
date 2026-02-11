# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Concat shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class ConcatTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
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
            # From ONNX test_concat: (2,4,3) + (7,4,3) → (9,4,3)
            (
                "onnx_basic",
                [ts(FLOAT, [2, 4, 3]), ts(FLOAT, [7, 4, 3])],
                0,
                [ts(FLOAT, [9, 4, 3])],
            ),
            # From ONNX test_concat_3d_axis_2: (2,2,2) + (2,2,2) → (2,2,4)
            (
                "3d_axis_2",
                [ts(FLOAT, [2, 2, 2]), ts(FLOAT, [2, 2, 2])],
                2,
                [ts(FLOAT, [2, 2, 4])],
            ),
        ]
    )
    def test_concat(self, _name, inputs, axis, expected):
        actual = run_shape_inference(
            "",
            "Concat",
            inputs,
            {"axis": ir.Attr("axis", ir.AttributeType.INT, axis)},
            opset_version=17,
        )
        self.assertEqual(actual, expected)

    def test_missing_input_shape(self):
        actual = run_shape_inference(
            "",
            "Concat",
            [ts(FLOAT, [2, 3]), ts(FLOAT)],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        # dtype should still propagate even if shape is unknown
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_symbolic_concat_dim(self):
        """When the concat dim is symbolic, result should also be symbolic."""
        actual = run_shape_inference(
            "",
            "Concat",
            [ts(FLOAT, [2, "a"]), ts(FLOAT, [2, "b"])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
            opset_version=17,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape[0], 2)
        # Concat of symbolic dims should be symbolic
        self.assertNotIsInstance(result.shape[1], int)

    def test_single_input(self):
        """Single input concat is identity."""
        actual = run_shape_inference(
            "",
            "Concat",
            [ts(FLOAT, [3, 4, 5])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, 5])])

    def test_rank_mismatch_records_error(self):
        actual = run_shape_inference(
            "",
            "Concat",
            [ts(FLOAT, [2, 3]), ts(FLOAT, [2, 3, 4])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_axis_out_of_range_records_error(self):
        actual = run_shape_inference(
            "",
            "Concat",
            [ts(FLOAT, [2, 3])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 5)},
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs_records_error(self):
        actual = run_shape_inference(
            "",
            "Concat",
            [],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_missing_axis_records_error(self):
        actual = run_shape_inference(
            "",
            "Concat",
            [ts(FLOAT, [2, 3])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_none_input_returns_early(self):
        """A None input in the middle means we can't compute output shape."""
        v1 = ir.Value(name="v1", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 3]))
        actual = run_shape_inference_with_values(
            "",
            "Concat",
            [v1, None],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)


if __name__ == "__main__":
    unittest.main()
