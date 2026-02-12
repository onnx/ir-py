# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for miscellaneous operator shape inference."""

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
INT64 = ir.DataType.INT64
UINT8 = ir.DataType.UINT8


class EyeLikeTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference("", "EyeLike", [ts(FLOAT, [3, 4])], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_dtype_attr(self):
        attrs = {
            "dtype": ir.Attr("dtype", ir.AttributeType.INT, ir.DataType.DOUBLE.value),
        }
        actual = run_shape_inference(
            "", "EyeLike", [ts(FLOAT, [3, 4])], attrs, opset_version=21
        )
        self.assertEqual(actual, [ts(ir.DataType.DOUBLE, [3, 4])])

    def test_symbolic_dims(self):
        actual = run_shape_inference(
            "", "EyeLike", [ts(FLOAT, ["N", "M"])], opset_version=21
        )
        self.assertEqual(actual[0].shape, ir.Shape(["N", "M"]))
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "EyeLike", [None], opset_version=21)


class ImageDecoderTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference("", "ImageDecoder", [ts(UINT8, [100])], opset_version=20)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[2], 3)
        self.assertEqual(actual[0].type.dtype, UINT8)

    def test_grayscale(self):
        attrs = {
            "pixel_format": ir.Attr("pixel_format", ir.AttributeType.STRING, "Grayscale"),
        }
        actual = run_shape_inference(
            "", "ImageDecoder", [ts(UINT8, [100])], attrs, opset_version=20
        )
        self.assertEqual(actual[0].shape[2], 1)


class MelWeightMatrixTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "MelWeightMatrix",
            [ts(INT64, []), ts(INT64, []), ts(FLOAT, []), ts(FLOAT, []), ts(INT64, [])],
            opset_version=17,
        )
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].type.dtype, FLOAT)


class TfIdfVectorizerTest(unittest.TestCase):
    def test_1d_input(self):
        actual = run_shape_inference("", "TfIdfVectorizer", [ts(INT64, [10])], opset_version=9)
        self.assertEqual(actual[0].shape.rank(), 1)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_2d_input(self):
        actual = run_shape_inference(
            "", "TfIdfVectorizer", [ts(INT64, [4, 10])], opset_version=9
        )
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertEqual(actual[0].shape[0], 4)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)


if __name__ == "__main__":
    unittest.main()
