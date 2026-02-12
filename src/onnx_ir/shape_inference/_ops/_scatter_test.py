# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ScatterElements and ScatterND shape inference."""

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


class ScatterElementsTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "ScatterElements",
            [ts(FLOAT, [3, 4])],
            opset_version=18,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_symbolic_dims(self):
        """Verify symbolic dims are SymbolicDim instances."""
        actual = run_shape_inference(
            "",
            "ScatterElements",
            [ts(FLOAT, ["N", "C"])],
            opset_version=18,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)
        self.assertIsInstance(result.shape[0], ir.SymbolicDim)
        self.assertIsInstance(result.shape[1], ir.SymbolicDim)

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "ScatterElements",
                [None],
                opset_version=18,
            )


class ScatterNDTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "ScatterND",
            [ts(FLOAT, [4, 5, 6])],
            opset_version=18,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 5, 6])])

    def test_symbolic_dims(self):
        """Verify symbolic dims are SymbolicDim instances."""
        actual = run_shape_inference(
            "",
            "ScatterND",
            [ts(FLOAT, ["N", "C", "D"])],
            opset_version=18,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 3)
        self.assertIsInstance(result.shape[0], ir.SymbolicDim)
        self.assertIsInstance(result.shape[1], ir.SymbolicDim)
        self.assertIsInstance(result.shape[2], ir.SymbolicDim)

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "ScatterND",
                [None],
                opset_version=18,
            )


class TensorScatterTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "TensorScatter",
            [ts(FLOAT, [5, 3]), ts(INT64, [2, 1]), ts(FLOAT, [2, 3])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, [5, 3])])


if __name__ == "__main__":
    unittest.main()
