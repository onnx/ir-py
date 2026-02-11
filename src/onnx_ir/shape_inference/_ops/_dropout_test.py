# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Dropout shape inference."""

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
BOOL = ir.DataType.BOOL


class DropoutTest(unittest.TestCase):
    def test_output_shape_passthrough(self):
        actual = run_shape_inference(
            "",
            "Dropout",
            [ts(FLOAT, [3, 4])],
            opset_version=13,
            num_outputs=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_mask_output(self):
        actual = run_shape_inference(
            "",
            "Dropout",
            [ts(FLOAT, ["batch", 128])],
            opset_version=13,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, ["batch", 128]))
        self.assertEqual(actual[1], ts(BOOL, ["batch", 128]))

    def test_symbolic_dims(self):
        """Dropout on ["N", "C"] → same shape, both SymbolicDim."""
        actual = run_shape_inference(
            "",
            "Dropout",
            [ts(FLOAT, ["N", "C"])],
            opset_version=13,
            num_outputs=1,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)
        self.assertIsInstance(result.shape[0], ir.SymbolicDim)
        self.assertIsInstance(result.shape[1], ir.SymbolicDim)

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "Dropout",
            [ts(FLOAT)],
            opset_version=13,
            num_outputs=2,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)
        self.assertIsNone(actual[1].shape)
        self.assertEqual(actual[1].type.dtype, BOOL)

    def test_single_output_only(self):
        actual = run_shape_inference(
            "",
            "Dropout",
            [ts(FLOAT, [2, 3, 4])],
            opset_version=13,
            num_outputs=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3, 4])])

    def test_dropout_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "",
                "Dropout",
                [],
                opset_version=17,
                num_outputs=2,
            )

    def test_dropout_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Dropout",
                [None],
                opset_version=17,
                num_outputs=2,
            )


if __name__ == "__main__":
    unittest.main()
