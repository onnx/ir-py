# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Resize shape inference."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class ResizeTest(unittest.TestCase):
    def test_no_shape_sets_dtype(self):
        actual = run_shape_inference(
            "",
            "Resize",
            [ts(FLOAT)],
            opset_version=19,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_fallback_symbolic_dims(self):
        actual = run_shape_inference(
            "",
            "Resize",
            [ts(FLOAT, [1, 3, 4, 4])],
            opset_version=19,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)

    def test_symbolic_fallback(self):
        """Resize with symbolic input: ["N", 3, 4, 4] → rank 4, batch is SymbolicDim."""
        actual = run_shape_inference(
            "",
            "Resize",
            [ts(FLOAT, ["N", 3, 4, 4])],
            opset_version=19,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 4)
        self.assertIsInstance(result.shape[0], ir.SymbolicDim)

    def test_with_sizes_const(self):
        x = ir.Value(name="X", shape=ir.Shape([1, 3, 4, 4]), type=ir.TensorType(FLOAT))
        roi = ir.Value(name="roi", type=ir.TensorType(FLOAT))
        scales = ir.Value(name="scales", type=ir.TensorType(FLOAT))
        sizes = const_value([1, 3, 8, 8], name="sizes")
        actual = run_shape_inference_with_values(
            "",
            "Resize",
            [x, roi, scales, sizes],
            opset_version=19,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 8, 8])])

    def test_with_scales_const(self):
        x = ir.Value(name="X", shape=ir.Shape([1, 3, 4, 4]), type=ir.TensorType(FLOAT))
        roi = ir.Value(name="roi", type=ir.TensorType(FLOAT))
        scales_tensor = ir.Tensor(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
        scales = ir.Value(
            name="scales",
            const_value=scales_tensor,
            type=ir.TensorType(FLOAT),
        )
        scales.shape = ir.Shape([4])
        actual = run_shape_inference_with_values(
            "",
            "Resize",
            [x, roi, scales],
            opset_version=19,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 8, 8])])

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Resize", [], opset_version=19)

    def test_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Resize",
                [None],
                opset_version=19,
            )


if __name__ == "__main__":
    unittest.main()
