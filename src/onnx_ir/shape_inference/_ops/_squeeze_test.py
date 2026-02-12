# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Squeeze and Unsqueeze shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class SqueezeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
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
            # From ONNX test_squeeze: removes all 1s
            (
                "many_ones",
                [1, 3, 1, 1, 2, 1],
                [0, 2, 3, 5],
                [3, 2],
            ),
        ]
    )
    def test_squeeze_attr(self, _name, input_shape, axes, expected_shape):
        attrs = {}
        if axes is not None:
            attrs["axes"] = ir.Attr("axes", ir.AttributeType.INTS, axes)
        actual = run_shape_inference(
            "",
            "Squeeze",
            [ts(FLOAT, input_shape)],
            attrs,
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_squeeze_opset13_axes_input(self):
        """Opset 13+: axes come from input[1]."""
        data = ir.Value(name="data", shape=ir.Shape([1, 3, 1, 5]), type=ir.TensorType(FLOAT))
        axes = const_value([0, 2], "axes")
        actual = run_shape_inference_with_values(
            "",
            "Squeeze",
            [data, axes],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 5])])

    def test_squeeze_no_axes_removes_all_ones(self):
        """No axes specified: remove all dims that are statically 1."""
        actual = run_shape_inference(
            "",
            "Squeeze",
            [ts(FLOAT, [1, 3, 1, 1, 2, 1])],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 2])])


class UnsqueezeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
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
            # From ONNX test_unsqueeze_regular: complex multi-axis insertion
            (
                "multi_axis",
                [3, 2],
                [0, 1, 3, 5],
                [1, 1, 3, 1, 2, 1],
            ),
        ]
    )
    def test_unsqueeze_attr(self, _name, input_shape, axes, expected_shape):
        actual = run_shape_inference(
            "",
            "Unsqueeze",
            [ts(FLOAT, input_shape)],
            {"axes": ir.Attr("axes", ir.AttributeType.INTS, axes)},
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_unsqueeze_opset13_axes_input(self):
        """Opset 13+: axes come from input[1]."""
        data = ir.Value(name="data", shape=ir.Shape([3, 4, 5]), type=ir.TensorType(FLOAT))
        axes = const_value([0, 4], "axes")
        actual = run_shape_inference_with_values(
            "",
            "Unsqueeze",
            [data, axes],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 4, 5, 1])])

    def test_unsqueeze_opset13_symbolic(self):
        """Opset 13+: Unsqueeze ["N", "C"] with axes [0, 3] → [1, "N", "C", 1]."""
        data = ir.Value(
            name="data", shape=ir.Shape(["N", "C"]), type=ir.TensorType(FLOAT)
        )
        axes = const_value([0, 3], "axes")
        actual = run_shape_inference_with_values(
            "",
            "Unsqueeze",
            [data, axes],
            opset_version=13,
        )
        self.assertEqual(actual[0].shape, ir.Shape([1, "N", "C", 1]))
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_unsqueeze_scalar(self):
        """From ONNX test_unsqueeze_scalar: scalar input with axis=-1."""
        data = ir.Value(name="data", shape=ir.Shape([]), type=ir.TensorType(FLOAT))
        axes = const_value([-1], "axes")
        actual = run_shape_inference_with_values(
            "",
            "Unsqueeze",
            [data, axes],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [1])])

    def test_unsqueeze_negative_axes(self):
        """From ONNX test_unsqueeze_negative_axes."""
        data = ir.Value(name="data", shape=ir.Shape([3, 4, 5]), type=ir.TensorType(FLOAT))
        # -5 normalizes to 0 in output_rank=5, giving axes [0, 4]
        axes = const_value([-5, 4], "axes")
        actual = run_shape_inference_with_values(
            "",
            "Unsqueeze",
            [data, axes],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 4, 5, 1])])

    def test_squeeze_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Squeeze", [], opset_version=17)

    def test_squeeze_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Squeeze",
                [None],
                opset_version=17,
            )

    def test_unsqueeze_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Unsqueeze", [], opset_version=17)

    def test_unsqueeze_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Unsqueeze",
                [None],
                opset_version=17,
            )


if __name__ == "__main__":
    unittest.main()
