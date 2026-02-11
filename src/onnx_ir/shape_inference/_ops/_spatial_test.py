# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for spatial operator shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class GridSampleTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "GridSample",
            [ts(FLOAT, [1, 3, 4, 5]), ts(FLOAT, [1, 6, 7, 2])],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 6, 7])])

    def test_missing_x_shape(self):
        actual = run_shape_inference(
            "",
            "GridSample",
            [ts(FLOAT), ts(FLOAT, [1, 6, 7, 2])],
            opset_version=20,
        )
        self.assertIsNone(actual[0].shape)

    def test_missing_grid_shape(self):
        actual = run_shape_inference(
            "",
            "GridSample",
            [ts(FLOAT, [1, 3, 4, 5]), ts(FLOAT)],
            opset_version=20,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "GridSample", [], opset_version=20)

    def test_none_input(self):
        v = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([1, 3, 4, 5]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "GridSample",
                [v, None],
                opset_version=20,
            )


class RoiAlignTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "output_height": ir.Attr("output_height", ir.AttributeType.INT, 2),
            "output_width": ir.Attr("output_width", ir.AttributeType.INT, 2),
        }
        actual = run_shape_inference(
            "",
            "RoiAlign",
            [
                ts(FLOAT, [1, 3, 4, 5]),
                ts(FLOAT, [5, 4]),
                ts(INT64, [5]),
            ],
            attrs,
            opset_version=16,
        )
        self.assertEqual(actual, [ts(FLOAT, [5, 3, 2, 2])])

    def test_default_output_size(self):
        actual = run_shape_inference(
            "",
            "RoiAlign",
            [
                ts(FLOAT, [1, 3, 4, 5]),
                ts(FLOAT, [5, 4]),
                ts(INT64, [5]),
            ],
            opset_version=16,
        )
        # Default output_height=1, output_width=1
        self.assertEqual(actual, [ts(FLOAT, [5, 3, 1, 1])])

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "RoiAlign", [], opset_version=16)

    def test_none_input(self):
        v = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([1, 3, 4, 5]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "RoiAlign",
                [v, None, None],
                opset_version=16,
            )


class MaxRoiPoolTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "pooled_shape": ir.Attr("pooled_shape", ir.AttributeType.INTS, [3, 3]),
        }
        actual = run_shape_inference(
            "",
            "MaxRoiPool",
            [ts(FLOAT, [1, 3, 8, 8]), ts(FLOAT, [5, 5])],
            attrs,
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [5, 3, 3, 3])])

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "",
                "MaxRoiPool",
                [],
                {"pooled_shape": ir.Attr("pooled_shape", ir.AttributeType.INTS, [3, 3])},
                opset_version=1,
            )


class AffineGridTest(unittest.TestCase):
    def test_2d(self):
        theta = ir.Value(name="theta", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 2, 3]))
        size = const_value([2, 3, 4, 5], name="size")
        actual = run_shape_inference_with_values(
            "",
            "AffineGrid",
            [theta, size],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 4, 5, 2])])

    def test_3d(self):
        theta = ir.Value(name="theta", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 3, 4]))
        size = const_value([2, 3, 4, 5, 6], name="size")
        actual = run_shape_inference_with_values(
            "",
            "AffineGrid",
            [theta, size],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 4, 5, 6, 3])])

    def test_no_const_size(self):
        theta = ir.Value(name="theta", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 2, 3]))
        size = ir.Value(name="size", type=ir.TensorType(INT64), shape=ir.Shape([4]))
        actual = run_shape_inference_with_values(
            "",
            "AffineGrid",
            [theta, size],
            opset_version=20,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "AffineGrid", [], opset_version=20)

    def test_none_input(self):
        theta = ir.Value(name="theta", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 2, 3]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "AffineGrid",
                [theta, None],
                opset_version=20,
            )


class GridSampleSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_dims(self):
        actual = run_shape_inference(
            "",
            "GridSample",
            [ts(FLOAT, ["N", "C", "H", "W"]), ts(FLOAT, ["N", 6, 7, 2])],
            opset_version=20,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[2], 6)
        self.assertEqual(actual[0].shape[3], 7)


class Col2ImSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_batch(self):
        input_val = ir.Value(
            name="input", type=ir.TensorType(FLOAT), shape=ir.Shape(["N", 27, 4])
        )
        image_shape = const_value([4, 4], name="image_shape")
        block_shape = const_value([3, 3], name="block_shape")
        actual = run_shape_inference_with_values(
            "",
            "Col2Im",
            [input_val, image_shape, block_shape],
            opset_version=18,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[2], 4)
        self.assertEqual(actual[0].shape[3], 4)


class RoiAlignSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_dims(self):
        attrs = {
            "output_height": ir.Attr("output_height", ir.AttributeType.INT, 3),
            "output_width": ir.Attr("output_width", ir.AttributeType.INT, 3),
        }
        actual = run_shape_inference(
            "",
            "RoiAlign",
            [
                ts(FLOAT, ["N", "C", "H", "W"]),
                ts(FLOAT, ["R", 4]),
                ts(INT64, ["R"]),
            ],
            attrs,
            opset_version=16,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[2], 3)
        self.assertEqual(actual[0].shape[3], 3)


if __name__ == "__main__":
    unittest.main()
