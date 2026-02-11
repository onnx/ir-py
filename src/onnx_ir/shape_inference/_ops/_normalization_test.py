# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for normalization shape inference."""

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
UINT8 = ir.DataType.UINT8


class BatchNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "BatchNormalization",
            [
                ts(FLOAT, [2, 3, 4, 5]),
                ts(FLOAT, [3]),
                ts(FLOAT, [3]),
                ts(FLOAT, [3]),
                ts(FLOAT, [3]),
            ],
            opset_version=15,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3, 4, 5])])

    def test_symbolic_dims(self):
        actual = run_shape_inference(
            "",
            "BatchNormalization",
            [
                ts(FLOAT, ["N", "C", "H", "W"]),
                ts(FLOAT, ["C"]),
                ts(FLOAT, ["C"]),
                ts(FLOAT, ["C"]),
                ts(FLOAT, ["C"]),
            ],
            opset_version=15,
        )
        self.assertEqual(actual, [ts(FLOAT, ["N", "C", "H", "W"])])

    def test_none_input_raises(self):
        v = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 3, 4, 5]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "BatchNormalization",
                [v, None],
                opset_version=15,
            )


class LayerNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "LayerNormalization",
            [ts(FLOAT, [2, 3, 4]), ts(FLOAT, [4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3, 4])])

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "LayerNormalization",
                [None],
                opset_version=17,
            )


class GroupNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "GroupNormalization",
            [ts(FLOAT, [2, 6, 4, 4])],
            opset_version=21,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 6, 4, 4])])


class RMSNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "RMSNormalization",
            [ts(FLOAT, [2, 3, 4])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3, 4])])


class InstanceNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "InstanceNormalization",
            [ts(FLOAT, [2, 3, 4, 5])],
            opset_version=6,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3, 4, 5])])


class LRNTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "LRN",
            [ts(FLOAT, [2, 3, 4, 5])],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3, 4, 5])])


class DequantizeLinearTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "DequantizeLinear",
            [ts(UINT8, [2, 3])],
            opset_version=21,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3])])


class QuantizeLinearTest(unittest.TestCase):
    def test_default_uint8(self):
        actual = run_shape_inference(
            "",
            "QuantizeLinear",
            [ts(FLOAT, [2, 3])],
            opset_version=21,
        )
        self.assertEqual(actual, [ts(UINT8, [2, 3])])


if __name__ == "__main__":
    unittest.main()
