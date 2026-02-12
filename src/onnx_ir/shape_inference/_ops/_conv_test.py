# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Conv shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError, ShapeInferenceError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT8 = ir.DataType.INT8
INT32 = ir.DataType.INT32
UINT8 = ir.DataType.UINT8


def _attrs(**kwargs) -> dict[str, ir.Attr]:
    """Build Conv attributes dict from keyword arguments."""
    attrs = {}
    for name, value in kwargs.items():
        if isinstance(value, list):
            attrs[name] = ir.Attr(name, ir.AttributeType.INTS, value)
        elif isinstance(value, int):
            attrs[name] = ir.Attr(name, ir.AttributeType.INT, value)
        elif isinstance(value, str):
            attrs[name] = ir.Attr(name, ir.AttributeType.STRING, value)
    return attrs


class ConvTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            # Basic cases
            (
                "basic_no_pad",
                [1, 3, 28, 28],
                [16, 3, 3, 3],
                {},
                [1, 16, 26, 26],
            ),
            (
                "with_padding",
                [1, 3, 28, 28],
                [16, 3, 3, 3],
                _attrs(pads=[1, 1, 1, 1]),
                [1, 16, 28, 28],
            ),
            (
                "stride_2",
                [1, 3, 28, 28],
                [16, 3, 3, 3],
                _attrs(strides=[2, 2]),
                [1, 16, 13, 13],
            ),
            # ResNet first layer
            (
                "resnet_first_layer",
                [1, 3, 224, 224],
                [64, 3, 7, 7],
                _attrs(pads=[3, 3, 3, 3], strides=[2, 2]),
                [1, 64, 112, 112],
            ),
            # 1D convolution
            (
                "1d_conv",
                [1, 3, 100],
                [16, 3, 5],
                {},
                [1, 16, 96],
            ),
            # 3D convolution with dilations, strides, and pads (from ONNX test_conv)
            (
                "3d_conv",
                [3, 4, 5, 6, 7],
                [5, 4, 2, 4, 3],
                _attrs(pads=[0, 1, 1, 0, 0, 1], dilations=[1, 2, 2], strides=[1, 1, 2]),
                [3, 5, 4, 1, 3],
            ),
            # 1D conv simple (from ONNX test_conv_1d_simple)
            (
                "1d_conv_simple",
                [30, 4, 5],
                [50, 4, 2],
                _attrs(dilations=[1]),
                [30, 50, 4],
            ),
            # 3D conv with dilations only (from ONNX test_conv_dilations)
            (
                "3d_dilations",
                [30, 4, 8, 8, 8],
                [50, 4, 3, 3, 3],
                _attrs(dilations=[1, 2, 3]),
                [30, 50, 6, 4, 2],
            ),
            # 3D conv with strides only (from ONNX test_conv_strides)
            (
                "3d_strides",
                [30, 4, 8, 8, 8],
                [50, 4, 3, 3, 3],
                _attrs(strides=[1, 2, 3]),
                [30, 50, 6, 3, 2],
            ),
            # 3D conv with pads (from ONNX test_conv_pads)
            (
                "3d_pads",
                [30, 4, 7, 6, 4],
                [50, 4, 3, 3, 3],
                _attrs(pads=[1, 1, 2, 0, 1, 2]),
                [30, 50, 6, 6, 6],
            ),
            # Grouped conv (from ONNX test_conv_group)
            (
                "grouped",
                [30, 4, 8, 8, 8],
                [4, 1, 8, 8, 8],
                _attrs(group=4),
                [30, 4, 1, 1, 1],
            ),
            # Conv resulting in 1 output pos (from ONNX test_conv_only_one_pos)
            (
                "only_one_pos",
                [30, 4, 5],
                [50, 4, 5],
                _attrs(strides=[2]),
                [30, 50, 1],
            ),
            # Symbolic batch dimension
            (
                "symbolic_batch",
                ["batch", 3, 224, 224],
                [64, 3, 7, 7],
                _attrs(pads=[3, 3, 3, 3], strides=[2, 2]),
                ["batch", 64, 112, 112],
            ),
        ]
    )
    def test_conv(self, _name, x_shape, w_shape, attrs, expected_shape):
        actual = run_shape_inference(
            "",
            "Conv",
            [ts(FLOAT, x_shape), ts(FLOAT, w_shape)],
            attrs or None,
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    @parameterized.parameterized.expand(
        [
            # auto_pad="SAME_UPPER" (from ONNX test_conv_auto_pad)
            (
                "same_upper",
                [30, 4, 7, 6, 4],
                [50, 4, 4, 3, 2],
                _attrs(auto_pad="SAME_UPPER"),
                [30, 50, 7, 6, 4],
            ),
            # auto_pad="SAME_UPPER" with strides (from ONNX test_conv_auto_pads)
            (
                "same_upper_strides",
                [30, 4, 7, 6, 4],
                [50, 4, 4, 3, 2],
                _attrs(auto_pad="SAME_UPPER", strides=[2, 2, 1]),
                [30, 50, 4, 3, 4],
            ),
            # auto_pad="SAME_UPPER" with dilations (from ONNX test_conv_auto_pad_dilation)
            (
                "same_upper_dilations",
                [30, 4, 65, 64, 63],
                [50, 4, 4, 3, 2],
                _attrs(auto_pad="SAME_UPPER", dilations=[2, 3, 4]),
                [30, 50, 65, 64, 63],
            ),
            # auto_pad="VALID"
            (
                "valid",
                [1, 3, 28, 28],
                [16, 3, 3, 3],
                _attrs(auto_pad="VALID"),
                [1, 16, 26, 26],
            ),
            # auto_pad="SAME_LOWER"
            (
                "same_lower",
                [1, 1, 5],
                [1, 1, 3],
                _attrs(auto_pad="SAME_LOWER", strides=[2]),
                [1, 1, 3],
            ),
        ]
    )
    def test_conv_auto_pad(self, _name, x_shape, w_shape, attrs, expected_shape):
        actual = run_shape_inference(
            "",
            "Conv",
            [ts(FLOAT, x_shape), ts(FLOAT, w_shape)],
            attrs,
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_symbolic_batch_dim(self):
        """Conv with symbolic batch: ["N", 3, "H", "W"] → rank 4, batch is SymbolicDim."""
        actual = run_shape_inference(
            "",
            "Conv",
            [ts(FLOAT, ["N", 3, "H", "W"]), ts(FLOAT, [16, 3, 3, 3])],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 4)
        self.assertEqual(result.shape[0], ir.SymbolicDim("N"))
        self.assertEqual(result.shape[1], 16)

    def test_partial_missing_input_shape(self):
        """When a spatial dim is unknown, output dim should also be unknown."""
        actual = run_shape_inference(
            "",
            "Conv",
            [ts(FLOAT, [30, 4, None, 6, 4]), ts(FLOAT, [50, 4, 3, 3, 3])],
            _attrs(pads=[1, 1, 2, 0, 1, 2]),
            opset_version=17,
        )
        result = actual[0]
        self.assertEqual(result.shape[0], 30)
        self.assertEqual(result.shape[1], 50)
        self.assertNotIsInstance(result.shape[2], int)  # Unknown
        self.assertEqual(result.shape[3], 6)
        self.assertEqual(result.shape[4], 6)

    def test_partial_missing_weight_shape(self):
        """When kernel dims are unknown and no kernel_shape attr, spatial dims unknown."""
        actual = run_shape_inference(
            "",
            "Conv",
            [ts(FLOAT, [30, 4, 7, 6, 4]), ts(FLOAT, [50, 4, None, 3, 3])],
            _attrs(pads=[1, 1, 2, 0, 1, 2]),
            opset_version=17,
        )
        result = actual[0]
        self.assertEqual(result.shape[0], 30)
        self.assertEqual(result.shape[1], 50)
        # First spatial dim unknown because kernel is unknown
        self.assertNotIsInstance(result.shape[2], int)

    def test_kernel_shape_attribute_overrides_weight(self):
        """kernel_shape attr should be used when weight spatial dims are unknown."""
        actual = run_shape_inference(
            "",
            "Conv",
            [ts(FLOAT, [25, 48, 16, 16]), ts(FLOAT, [32, 48, None, None])],
            _attrs(kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 2, 2]),
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [25, 32, 9, 9])])

    def test_missing_x_shape(self):
        """When x shape is entirely unknown, output shape should be None."""
        actual = run_shape_inference(
            "",
            "Conv",
            [ts(FLOAT), ts(FLOAT, [16, 3, 3, 3])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_with_bias(self):
        """Conv with optional bias input (3rd input)."""
        actual = run_shape_inference(
            "",
            "Conv",
            [ts(FLOAT, [1, 3, 28, 28]), ts(FLOAT, [16, 3, 3, 3]), ts(FLOAT, [16])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 16, 26, 26])])

    def test_conv_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Conv", [], opset_version=17)

    def test_conv_none_input(self):
        w = ir.Value(name="w", type=ir.TensorType(FLOAT), shape=ir.Shape([16, 3, 3, 3]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Conv",
                [None, w],
                {"kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3])},
                opset_version=17,
            )

    def test_conv_missing_input_spatial(self):
        """Conv with unknown spatial dims but known batch/channels."""
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "Conv",
                [ts(FLOAT, [1, 3]), ts(FLOAT, [16, 3, 3, 3])],
                {"kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3])},
                opset_version=17,
            )


class ConvIntegerTest(unittest.TestCase):
    def test_conv_integer_basic(self):
        actual = run_shape_inference(
            "",
            "ConvInteger",
            [ts(INT8, [1, 1, 5, 5]), ts(INT8, [1, 1, 3, 3])],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [1, 1, 3, 3])])

    def test_conv_integer_with_zero_points(self):
        actual = run_shape_inference(
            "",
            "ConvInteger",
            [
                ts(INT8, [1, 1, 5, 5]),
                ts(INT8, [1, 1, 3, 3]),
                ts(INT8, []),
                ts(INT8, []),
            ],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [1, 1, 3, 3])])

    def test_conv_integer_missing_shape(self):
        actual = run_shape_inference(
            "",
            "ConvInteger",
            [ts(INT8), ts(INT8, [1, 1, 3, 3])],
            opset_version=10,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, INT32)

    def test_conv_integer_with_padding(self):
        actual = run_shape_inference(
            "",
            "ConvInteger",
            [ts(INT8, [1, 1, 5, 5]), ts(INT8, [1, 1, 3, 3])],
            _attrs(pads=[1, 1, 1, 1]),
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [1, 1, 5, 5])])

    def test_conv_integer_auto_pad_same_upper(self):
        actual = run_shape_inference(
            "",
            "ConvInteger",
            [ts(INT8, [1, 1, 5, 5]), ts(INT8, [1, 1, 3, 3])],
            _attrs(auto_pad="SAME_UPPER"),
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [1, 1, 5, 5])])

    def test_conv_integer_auto_pad_valid(self):
        actual = run_shape_inference(
            "",
            "ConvInteger",
            [ts(INT8, [1, 1, 5, 5]), ts(INT8, [1, 1, 3, 3])],
            _attrs(auto_pad="VALID"),
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [1, 1, 3, 3])])

    def test_conv_integer_kernel_shape_attr(self):
        actual = run_shape_inference(
            "",
            "ConvInteger",
            [ts(INT8, [1, 1, 5, 5]), ts(INT8, [1, 1, None, None])],
            _attrs(kernel_shape=[3, 3]),
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [1, 1, 3, 3])])

    def test_conv_integer_symbolic_spatial(self):
        actual = run_shape_inference(
            "",
            "ConvInteger",
            [ts(INT8, [1, 1, "H", "W"]), ts(INT8, [1, 1, 3, 3])],
            opset_version=10,
        )
        result = actual[0]
        self.assertEqual(result.shape.rank(), 4)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 1)
        # Spatial dims are derived from symbolic H/W, so they're new symbolic dims
        self.assertIsInstance(result.shape[2], ir.SymbolicDim)
        self.assertIsInstance(result.shape[3], ir.SymbolicDim)


class QLinearConvTest(unittest.TestCase):
    def test_qlinear_conv_basic(self):
        actual = run_shape_inference(
            "",
            "QLinearConv",
            [
                ts(UINT8, [1, 1, 5, 5]),  # x
                ts(FLOAT, []),  # x_scale
                ts(UINT8, []),  # x_zero_point
                ts(UINT8, [1, 1, 3, 3]),  # w
                ts(FLOAT, []),  # w_scale
                ts(UINT8, []),  # w_zero_point
                ts(FLOAT, []),  # y_scale
                ts(UINT8, []),  # y_zero_point
            ],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(UINT8, [1, 1, 3, 3])])

    def test_qlinear_conv_int8_output(self):
        actual = run_shape_inference(
            "",
            "QLinearConv",
            [
                ts(INT8, [1, 1, 5, 5]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [1, 1, 3, 3]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),  # y_zero_point dtype determines output dtype
            ],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT8, [1, 1, 3, 3])])

    def test_qlinear_conv_missing_shape(self):
        actual = run_shape_inference(
            "",
            "QLinearConv",
            [
                ts(UINT8),
                ts(FLOAT, []),
                ts(UINT8, []),
                ts(UINT8, [1, 1, 3, 3]),
                ts(FLOAT, []),
                ts(UINT8, []),
                ts(FLOAT, []),
                ts(UINT8, []),
            ],
            opset_version=10,
        )
        self.assertIsNone(actual[0].shape)

    def test_qlinear_conv_too_few_inputs(self):
        from onnx_ir.shape_inference import OpUsageError

        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "",
                "QLinearConv",
                [ts(UINT8, [1, 1, 5, 5]), ts(FLOAT, [])],
                opset_version=10,
            )


if __name__ == "__main__":
    unittest.main()
