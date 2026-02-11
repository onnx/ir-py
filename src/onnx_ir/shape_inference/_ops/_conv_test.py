# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Conv shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT


class ConvTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        (
            "basic_no_pad",
            [1, 3, 28, 28], [16, 3, 3, 3],
            {}, {},
            [1, 16, 26, 26],
        ),
        (
            "with_padding",
            [1, 3, 28, 28], [16, 3, 3, 3],
            {"pads": [1, 1, 1, 1]}, {},
            [1, 16, 28, 28],
        ),
        (
            "stride_2",
            [1, 3, 28, 28], [16, 3, 3, 3],
            {}, {"strides": [2, 2]},
            [1, 16, 13, 13],
        ),
        (
            "resnet_first_layer",
            [1, 3, 224, 224], [64, 3, 7, 7],
            {"pads": [3, 3, 3, 3]}, {"strides": [2, 2]},
            [1, 64, 112, 112],
        ),
        (
            "symbolic_batch",
            ["batch", 3, 224, 224], [64, 3, 7, 7],
            {"pads": [3, 3, 3, 3]}, {"strides": [2, 2]},
            ["batch", 64, 112, 112],
        ),
        (
            "1d_conv",
            [1, 3, 100], [16, 3, 5],
            {}, {},
            [1, 16, 96],
        ),
    ])
    def test_conv(self, _name, x_shape, w_shape, pad_kw, stride_kw, expected_shape):
        attrs = {}
        if "pads" in pad_kw:
            attrs["pads"] = ir.Attr("pads", ir.AttributeType.INTS, pad_kw["pads"])
        if "strides" in stride_kw:
            attrs["strides"] = ir.Attr("strides", ir.AttributeType.INTS, stride_kw["strides"])
        actual = run_shape_inference(
            "", "Conv",
            [ts(FLOAT, x_shape), ts(FLOAT, w_shape)],
            attrs or None,
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])


if __name__ == "__main__":
    unittest.main()
