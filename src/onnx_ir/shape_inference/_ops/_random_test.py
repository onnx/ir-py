# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for random/distribution operator shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class RandomShapeAttrTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("random_normal", "RandomNormal"),
            ("random_uniform", "RandomUniform"),
        ]
    )
    def test_basic(self, _name, op_type):
        attrs = {
            "shape": ir.Attr("shape", ir.AttributeType.INTS, [2, 3]),
        }
        actual = run_shape_inference("", op_type, [], attrs, opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, [2, 3])])

    def test_random_normal_dtype_attr(self):
        attrs = {
            "shape": ir.Attr("shape", ir.AttributeType.INTS, [4, 5]),
            "dtype": ir.Attr("dtype", ir.AttributeType.INT, ir.DataType.DOUBLE.value),
        }
        actual = run_shape_inference("", "RandomNormal", [], attrs, opset_version=21)
        self.assertEqual(actual, [ts(ir.DataType.DOUBLE, [4, 5])])


class RandomLikeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("random_normal_like", "RandomNormalLike"),
            ("bernoulli", "Bernoulli"),
        ]
    )
    def test_basic(self, _name, op_type):
        actual = run_shape_inference("", op_type, [ts(FLOAT, [2, 3])], opset_version=21)
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
