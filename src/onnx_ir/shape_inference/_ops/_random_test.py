# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for random op shape inference."""

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


class RandomNormalTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "shape": ir.Attr("shape", ir.AttributeType.INTS, [2, 3]),
        }
        actual = run_shape_inference("", "RandomNormal", [], attrs, opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, [2, 3])])

    def test_dtype_attr(self):
        attrs = {
            "shape": ir.Attr("shape", ir.AttributeType.INTS, [4, 5]),
            "dtype": ir.Attr("dtype", ir.AttributeType.INT, ir.DataType.DOUBLE.value),
        }
        actual = run_shape_inference("", "RandomNormal", [], attrs, opset_version=21)
        self.assertEqual(actual, [ts(ir.DataType.DOUBLE, [4, 5])])


class RandomUniformTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "shape": ir.Attr("shape", ir.AttributeType.INTS, [2, 3]),
        }
        actual = run_shape_inference("", "RandomUniform", [], attrs, opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, [2, 3])])


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

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "EyeLike", [None], opset_version=21)


class RandomNormalLikeTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "", "RandomNormalLike", [ts(FLOAT, [2, 3])], opset_version=21
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3])])


class BernoulliTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference("", "Bernoulli", [ts(FLOAT, [2, 3])], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, [2, 3])])


class MultinomialTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "sample_size": ir.Attr("sample_size", ir.AttributeType.INT, 10),
        }
        actual = run_shape_inference(
            "", "Multinomial", [ts(FLOAT, [2, 5])], attrs, opset_version=21
        )
        self.assertEqual(actual, [ts(ir.DataType.INT32, [2, 10])])

    def test_default_sample_size(self):
        actual = run_shape_inference("", "Multinomial", [ts(FLOAT, [2, 5])], opset_version=21)
        self.assertEqual(actual, [ts(ir.DataType.INT32, [2, 1])])

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "Multinomial", [None], opset_version=21)


class BlackmanWindowTest(unittest.TestCase):
    def test_basic(self):
        size = const_value([8])
        actual = run_shape_inference_with_values(
            "", "BlackmanWindow", [size], opset_version=21
        )
        self.assertEqual(actual, [ts(FLOAT, [8])])


class HannWindowTest(unittest.TestCase):
    def test_basic(self):
        size = const_value([16])
        actual = run_shape_inference_with_values("", "HannWindow", [size], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, [16])])


class HammingWindowTest(unittest.TestCase):
    def test_basic(self):
        size = const_value([10])
        actual = run_shape_inference_with_values("", "HammingWindow", [size], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, [10])])


class MultinomialSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_batch(self):
        attrs = {
            "sample_size": ir.Attr("sample_size", ir.AttributeType.INT, 5),
        }
        actual = run_shape_inference(
            "", "Multinomial", [ts(FLOAT, ["N", 10])], attrs, opset_version=21
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 5)


if __name__ == "__main__":
    unittest.main()
