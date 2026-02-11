# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Squeeze and Unsqueeze shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT


class SqueezeTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        (
            "explicit_axes",
            [1, 3, 1, 5],
            [0, 2],
            [3, 5],
        ),
        (
            "no_axes_removes_ones",
            [1, 3, 1, 5],
            None,
            [3, 5],
        ),
        (
            "symbolic",
            [1, "batch", 1],
            [0, 2],
            ["batch"],
        ),
        (
            "negative_axis",
            [3, 1, 5],
            [-2],
            [3, 5],
        ),
    ])
    def test_squeeze(self, _name, input_shape, axes, expected_shape):
        attrs = {}
        if axes is not None:
            attrs["axes"] = ir.Attr("axes", ir.AttributeType.INTS, axes)
        actual = run_shape_inference(
            "", "Squeeze", [ts(FLOAT, input_shape)], attrs, opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])


class UnsqueezeTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        (
            "basic",
            [3, 4],
            [0, 3],
            [1, 3, 4, 1],
        ),
        (
            "symbolic",
            ["batch", 128],
            [0],
            [1, "batch", 128],
        ),
        (
            "negative_axis",
            [3, 4],
            [-1],
            [3, 4, 1],
        ),
        (
            "middle",
            [3, 4],
            [1],
            [3, 1, 4],
        ),
    ])
    def test_unsqueeze(self, _name, input_shape, axes, expected_shape):
        actual = run_shape_inference(
            "", "Unsqueeze", [ts(FLOAT, input_shape)],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, axes)},
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])


if __name__ == "__main__":
    unittest.main()
