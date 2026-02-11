# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TopK shape inference."""

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


class TopKTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "axis": ir.Attr("axis", ir.AttributeType.INT, -1),
        }
        actual = run_shape_inference(
            "",
            "TopK",
            [ts(FLOAT, [3, 4, 5]), ts(INT64, [1])],
            attrs,
            opset_version=21,
            num_outputs=2,
        )
        # The axis dim becomes symbolic
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertEqual(actual[0].shape[0], 3)
        self.assertEqual(actual[0].shape[1], 4)
        self.assertEqual(actual[0].type, ir.TensorType(FLOAT))
        # Second output is INT64 indices
        self.assertIsNotNone(actual[1].shape)
        self.assertEqual(actual[1].shape.rank(), 3)
        self.assertEqual(actual[1].type, ir.TensorType(INT64))

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "TopK",
            [ts(FLOAT), ts(INT64, [1])],
            opset_version=21,
            num_outputs=2,
        )
        self.assertIsNone(actual[0].shape)

    def test_none_input_raises(self):
        v = ir.Value(name="k", type=ir.TensorType(INT64), shape=ir.Shape([1]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "TopK", [None, v], opset_version=21, num_outputs=2
            )

    def test_const_k(self):
        """When K is a constant, the axis dim should be the concrete K value."""
        from onnx_ir.shape_inference._ops._testing import const_value

        x = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4, 5]))
        k = const_value([3])
        actual = run_shape_inference_with_values(
            "", "TopK", [x, k], opset_version=21, num_outputs=2
        )
        self.assertEqual(list(actual[0].shape), [3, 4, 3])
        self.assertEqual(list(actual[1].shape), [3, 4, 3])

    def test_symbolic_dims(self):
        """TopK with symbolic input dims preserves them on non-axis dims."""
        actual = run_shape_inference(
            "",
            "TopK",
            [ts(FLOAT, ["N", "M", 5]), ts(INT64, [1])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, -1)},
            opset_version=21,
            num_outputs=2,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)


if __name__ == "__main__":
    unittest.main()
