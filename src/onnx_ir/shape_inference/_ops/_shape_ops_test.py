# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Shape, Size, and Flatten shape inference."""

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
INT64 = ir.DataType.INT64


class ShapeOpTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("rank_3", [3, 4, 5], [3]),
            ("rank_2", [3, 4], [2]),
            ("symbolic", ["batch", "seq", 256], [3]),
            ("scalar", [], [0]),
        ]
    )
    def test_shape(self, _name, input_shape, expected_shape):
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, input_shape)],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, expected_shape)])

    def test_shape_with_start_end(self):
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 3, 4, 5])],
            {
                "start": ir.Attr("start", ir.AttributeType.INT, 1),
                "end": ir.Attr("end", ir.AttributeType.INT, 3),
            },
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [2])])

    def test_shape_start_only(self):
        """From ONNX test_shape_start_1."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"start": ir.Attr("start", ir.AttributeType.INT, 1)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [2])])

    def test_shape_end_only(self):
        """From ONNX test_shape_end_1."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"end": ir.Attr("end", ir.AttributeType.INT, 1)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [1])])

    def test_shape_negative_start(self):
        """From ONNX test_shape_negative_start."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"start": ir.Attr("start", ir.AttributeType.INT, -1)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [1])])

    def test_shape_clip_start(self):
        """From ONNX test_shape_clip1: start=-5 clipped to 0."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"start": ir.Attr("start", ir.AttributeType.INT, -5)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [3])])

    def test_shape_clip_end(self):
        """From ONNX test_shape_clip2: end=10 clipped to rank."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"end": ir.Attr("end", ir.AttributeType.INT, 10)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [3])])


class SizeTest(unittest.TestCase):
    def test_size(self):
        actual = run_shape_inference(
            "",
            "Size",
            [ts(FLOAT, [3, 4, 5])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])

    def test_size_symbolic(self):
        actual = run_shape_inference(
            "",
            "Size",
            [ts(FLOAT, ["batch", 128])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])


class FlattenTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            # From ONNX test_flatten: axis=2
            ("axis_2", [2, 3, 4, 5], 2, [6, 20]),
            # From ONNX test_flatten_default_axis: axis=1 default
            ("default_axis", [2, 3, 4], None, [2, 12]),
            # From ONNX test_flatten_zero_axis: axis=0
            ("axis_0", [2, 3, 4], 0, [1, 24]),
            ("axis_end", [2, 3, 4], 3, [24, 1]),
        ]
    )
    def test_flatten(self, _name, input_shape, axis, expected_shape):
        attrs = {}
        if axis is not None:
            attrs["axis"] = ir.Attr("axis", ir.AttributeType.INT, axis)
        actual = run_shape_inference(
            "",
            "Flatten",
            [ts(FLOAT, input_shape)],
            attrs or None,
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_flatten_unknown_dim(self):
        """From ONNX test_flatten_unknown_dim: symbolic dims → unknown dims."""
        actual = run_shape_inference(
            "",
            "Flatten",
            [ts(FLOAT, [2, "N", 4, 5])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 2)},
            opset_version=17,
        )
        # Both output dims should be unknown (symbolic input)
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)

    def test_shape_no_inputs(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference("", "Shape", [], opset_version=17)

    def test_shape_none_input(self):
        actual = run_shape_inference_with_values(
            "",
            "Shape",
            [None],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_size_no_inputs(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference("", "Size", [], opset_version=17)

    def test_size_none_input(self):
        actual = run_shape_inference_with_values(
            "",
            "Size",
            [None],
            opset_version=17,
        )
        self.assertEqual(actual[0].shape, ir.Shape([]))

    def test_flatten_no_inputs(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference("", "Flatten", [], opset_version=17)

    def test_flatten_none_input(self):
        actual = run_shape_inference_with_values(
            "",
            "Flatten",
            [None],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_flatten_missing_shape(self):
        """Flatten with unknown input shape → dtype only."""
        actual = run_shape_inference("", "Flatten", [ts(FLOAT)], opset_version=17)
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)


if __name__ == "__main__":
    unittest.main()
