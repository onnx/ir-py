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

    def test_training_mode_outputs(self):
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
            {"training_mode": ir.Attr("training_mode", ir.AttributeType.INT, 1)},
            opset_version=15,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 3, 4, 5]))
        self.assertEqual(actual[1], ts(FLOAT, [3]))
        self.assertEqual(actual[2], ts(FLOAT, [3]))


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

    def test_layer_normalization_mean_invstddev_outputs(self):
        actual = run_shape_inference(
            "",
            "LayerNormalization",
            [ts(FLOAT, [2, 3, 4]), ts(FLOAT, [4])],
            opset_version=17,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 3, 4]))
        # Mean and InvStdDev have reduced shape (axis=-1 → [2, 3])
        self.assertEqual(actual[1], ts(FLOAT, [2, 3]))
        self.assertEqual(actual[2], ts(FLOAT, [2, 3]))

    def test_layer_normalization_custom_axis(self):
        actual = run_shape_inference(
            "",
            "LayerNormalization",
            [ts(FLOAT, [2, 3, 4, 5]), ts(FLOAT, [4, 5])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 2)},
            opset_version=17,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 3, 4, 5]))
        # Reduced shape at axis=2 → [2, 3]
        self.assertEqual(actual[1], ts(FLOAT, [2, 3]))
        self.assertEqual(actual[2], ts(FLOAT, [2, 3]))


class GroupNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "GroupNormalization",
            [ts(FLOAT, [2, 6, 4, 4])],
            opset_version=21,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 6, 4, 4])])

    def test_group_normalization_with_scale_bias(self):
        actual = run_shape_inference(
            "",
            "GroupNormalization",
            [ts(FLOAT, [2, 6, 4, 4]), ts(FLOAT, [3]), ts(FLOAT, [3])],
            {"num_groups": ir.Attr("num_groups", ir.AttributeType.INT, 3)},
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

    def test_rms_normalization_with_inv_std_dev_output(self):
        actual = run_shape_inference(
            "",
            "RMSNormalization",
            [ts(FLOAT, [2, 3, 4]), ts(FLOAT, [4])],
            opset_version=24,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 3, 4]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 3]))


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

    def test_quantize_linear_with_zero_point_dtype(self):
        x = ir.Value(name="x", shape=ir.Shape([3, 4]), type=ir.TensorType(FLOAT))
        x_scale = ir.Value(name="x_scale", shape=ir.Shape([]), type=ir.TensorType(FLOAT))
        x_zp = ir.Value(
            name="x_zp", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.INT8)
        )
        actual = run_shape_inference_with_values(
            "",
            "QuantizeLinear",
            [x, x_scale, x_zp],
            opset_version=21,
        )
        self.assertEqual(actual, [ts(ir.DataType.INT8, [3, 4])])


class DynamicQuantizeLinearTest(unittest.TestCase):
    def test_dynamic_quantize_linear(self):
        actual = run_shape_inference(
            "",
            "DynamicQuantizeLinear",
            [ts(FLOAT, [3, 4])],
            opset_version=11,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(UINT8, [3, 4]))
        self.assertEqual(actual[1], ts(FLOAT, []))
        self.assertEqual(actual[2], ts(UINT8, []))


if __name__ == "__main__":
    unittest.main()
