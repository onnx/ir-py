# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for signal processing shape inference (DFT, STFT)."""

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


class DFTTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "DFT",
            [ts(FLOAT, [1, 10, 2])],
            opset_version=20,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "DFT",
            [ts(FLOAT)],
            opset_version=20,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "DFT", [], opset_version=20)

    def test_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "DFT",
                [None],
                opset_version=20,
            )


class STFTTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "STFT",
            [ts(FLOAT, [1, 128, 1]), ts(ir.DataType.INT64, [])],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertEqual(actual[0].shape[0], 1)  # batch from signal
        self.assertEqual(actual[0].shape[3], 2)  # real/imag
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_missing_signal_shape(self):
        actual = run_shape_inference(
            "",
            "STFT",
            [ts(FLOAT), ts(ir.DataType.INT64, [])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "STFT", [], opset_version=17)

    def test_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "STFT",
                [None, None],
                opset_version=17,
            )


class DFTSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_dims(self):
        actual = run_shape_inference(
            "",
            "DFT",
            [ts(FLOAT, ["N", "L", 2])],
            opset_version=20,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertEqual(actual[0].type.dtype, FLOAT)


class STFTSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_batch(self):
        actual = run_shape_inference(
            "",
            "STFT",
            [ts(FLOAT, ["N", 128, 1]), ts(ir.DataType.INT64, [])],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[3], 2)


if __name__ == "__main__":
    unittest.main()
