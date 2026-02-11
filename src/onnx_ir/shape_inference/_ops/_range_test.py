# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Range shape inference."""

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


class RangeTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "Range",
            [ts(FLOAT, []), ts(FLOAT, []), ts(FLOAT, [])],
            opset_version=21,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 1)
        self.assertEqual(actual[0].type, ir.TensorType(FLOAT))

    def test_int64_dtype(self):
        actual = run_shape_inference(
            "",
            "Range",
            [ts(INT64, []), ts(INT64, []), ts(INT64, [])],
            opset_version=21,
        )
        self.assertEqual(actual[0].type, ir.TensorType(INT64))

    def test_none_input_raises(self):
        v1 = ir.Value(name="limit", type=ir.TensorType(FLOAT), shape=ir.Shape([]))
        v2 = ir.Value(name="delta", type=ir.TensorType(FLOAT), shape=ir.Shape([]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "Range", [None, v1, v2], opset_version=21)

    def test_symbolic_scalar_inputs(self):
        """Range always produces 1-D output even with symbolic scalar inputs."""
        actual = run_shape_inference(
            "",
            "Range",
            [ts(FLOAT, []), ts(FLOAT, []), ts(FLOAT, [])],
            opset_version=21,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 1)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)


if __name__ == "__main__":
    unittest.main()
