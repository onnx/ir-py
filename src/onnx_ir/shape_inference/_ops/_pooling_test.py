# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pooling shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class GlobalAveragePoolTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "", "GlobalAveragePool", [ts(FLOAT, [1, 3, 4, 5])], opset_version=21
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 1, 1])])

    def test_3d(self):
        actual = run_shape_inference(
            "", "GlobalAveragePool", [ts(FLOAT, [2, 8, 6, 7, 9])], opset_version=21
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 1, 1, 1])])

    def test_missing_shape(self):
        actual = run_shape_inference("", "GlobalAveragePool", [ts(FLOAT)], opset_version=21)
        self.assertIsNone(actual[0].shape)

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "GlobalAveragePool", [None], opset_version=21)


class GlobalMaxPoolTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "", "GlobalMaxPool", [ts(FLOAT, [1, 3, 4, 5])], opset_version=21
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 1, 1])])


class AveragePoolTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [1, 1]),
            "pads": ir.Attr("pads", ir.AttributeType.INTS, [0, 0, 0, 0]),
        }
        actual = run_shape_inference(
            "",
            "AveragePool",
            [ts(FLOAT, [1, 1, 5, 5])],
            attrs,
            opset_version=21,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 3, 3])])

    def test_missing_shape(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
        }
        actual = run_shape_inference("", "AveragePool", [ts(FLOAT)], attrs, opset_version=21)
        self.assertIsNone(actual[0].shape)

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "AveragePool", [None], opset_version=21)


class MaxPoolTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [1, 1]),
            "pads": ir.Attr("pads", ir.AttributeType.INTS, [0, 0, 0, 0]),
        }
        actual = run_shape_inference(
            "",
            "MaxPool",
            [ts(FLOAT, [1, 1, 5, 5])],
            attrs,
            opset_version=21,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 3, 3])])

    def test_two_outputs(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [1, 1]),
            "pads": ir.Attr("pads", ir.AttributeType.INTS, [0, 0, 0, 0]),
        }
        actual = run_shape_inference(
            "",
            "MaxPool",
            [ts(FLOAT, [1, 1, 5, 5])],
            attrs,
            opset_version=21,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 1, 3, 3]))
        self.assertEqual(actual[1].shape, ir.Shape([1, 1, 3, 3]))
        self.assertEqual(actual[1].type, ir.TensorType(ir.DataType.INT64))

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "MaxPool", [None], opset_version=21)


class LpPoolTest(unittest.TestCase):
    def test_lp_pool_basic(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [1, 1]),
            "pads": ir.Attr("pads", ir.AttributeType.INTS, [0, 0, 0, 0]),
        }
        actual = run_shape_inference(
            "",
            "LpPool",
            [ts(FLOAT, [1, 1, 5, 5])],
            attrs,
            opset_version=18,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 3, 3])])

    def test_lp_pool_missing_shape(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
        }
        actual = run_shape_inference("", "LpPool", [ts(FLOAT)], attrs, opset_version=18)
        self.assertIsNone(actual[0].shape)

    def test_lp_pool_missing_kernel_shape_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "", "LpPool", [ts(FLOAT, [1, 1, 5, 5])], opset_version=18
            )

    def test_lp_pool_auto_pad_raises(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
            "auto_pad": ir.Attr("auto_pad", ir.AttributeType.STRING, "SAME_UPPER"),
        }
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "", "LpPool", [ts(FLOAT, [1, 1, 5, 5])], attrs, opset_version=18
            )


class GlobalLpPoolTest(unittest.TestCase):
    def test_global_lp_pool_basic(self):
        actual = run_shape_inference(
            "", "GlobalLpPool", [ts(FLOAT, [1, 3, 4, 5])], opset_version=21
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 1, 1])])

    def test_global_lp_pool_missing_shape(self):
        actual = run_shape_inference("", "GlobalLpPool", [ts(FLOAT)], opset_version=21)
        self.assertIsNone(actual[0].shape)


class MaxPoolWithIndicesTest(unittest.TestCase):
    def test_max_pool_two_outputs_missing_shape(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
        }
        actual = run_shape_inference(
            "", "MaxPool", [ts(FLOAT)], attrs, opset_version=21, num_outputs=2
        )
        self.assertIsNone(actual[0].shape)
        self.assertIsNone(actual[1].shape)
        self.assertEqual(actual[1].type.dtype, ir.DataType.INT64)


class PoolingSymbolicDimsTest(unittest.TestCase):
    def test_average_pool_symbolic_spatial_dims(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [1, 1]),
            "pads": ir.Attr("pads", ir.AttributeType.INTS, [0, 0, 0, 0]),
        }
        actual = run_shape_inference(
            "",
            "AveragePool",
            [ts(FLOAT, ["N", "C", "H", "W"])],
            attrs,
            opset_version=21,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)

    def test_max_pool_symbolic_spatial_dims(self):
        attrs = {
            "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [2, 2]),
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [2, 2]),
            "pads": ir.Attr("pads", ir.AttributeType.INTS, [0, 0, 0, 0]),
        }
        actual = run_shape_inference(
            "",
            "MaxPool",
            [ts(FLOAT, ["N", 3, "H", "W"])],
            attrs,
            opset_version=21,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 3)

    def test_global_average_pool_symbolic_dims(self):
        actual = run_shape_inference(
            "", "GlobalAveragePool", [ts(FLOAT, ["N", "C", "H", "W"])], opset_version=21
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[2], 1)
        self.assertEqual(actual[0].shape[3], 1)

    def test_global_max_pool_symbolic_dims(self):
        actual = run_shape_inference(
            "", "GlobalMaxPool", [ts(FLOAT, ["N", 8, "H", "W"])], opset_version=21
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 8)
        self.assertEqual(actual[0].shape[2], 1)
        self.assertEqual(actual[0].shape[3], 1)


if __name__ == "__main__":
    unittest.main()
