# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Transpose shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import InvalidOpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
FLOAT16 = ir.DataType.FLOAT16


class InferTransposeTest(unittest.TestCase):
    """Tests for Transpose shape inference function."""

    @parameterized.parameterized.expand(
        [
            (
                "explicit_perm",
                [ts(FLOAT, ["batch", "seq", 256])],
                {"perm": ir.Attr("perm", ir.AttributeType.INTS, [2, 0, 1])},
                [ts(FLOAT, [256, "batch", "seq"])],
            ),
            (
                "swap_last_two",
                [ts(FLOAT, ["batch", "seq", 256])],
                {"perm": ir.Attr("perm", ir.AttributeType.INTS, [0, 2, 1])},
                [ts(FLOAT, ["batch", 256, "seq"])],
            ),
            (
                "2d_transpose",
                [ts(FLOAT, ["batch", 128])],
                {"perm": ir.Attr("perm", ir.AttributeType.INTS, [1, 0])},
                [ts(FLOAT, [128, "batch"])],
            ),
            (
                "concrete_dims",
                [ts(FLOAT, [2, 3, 4])],
                {"perm": ir.Attr("perm", ir.AttributeType.INTS, [2, 0, 1])},
                [ts(FLOAT, [4, 2, 3])],
            ),
            (
                "default_perm_reverse_3d",
                [ts(FLOAT, [2, 3, 4])],
                None,
                [ts(FLOAT, [4, 3, 2])],
            ),
            (
                "default_perm_reverse_2d",
                [ts(FLOAT, [5, 7])],
                None,
                [ts(FLOAT, [7, 5])],
            ),
            (
                "default_perm_reverse_symbolic",
                [ts(FLOAT, ["batch", "seq"])],
                None,
                [ts(FLOAT, ["seq", "batch"])],
            ),
            (
                "dtype_propagated",
                [ts(FLOAT16, [2, 3])],
                None,
                [ts(FLOAT16, [3, 2])],
            ),
            (
                "no_shape_when_input_shape_missing",
                [ts(FLOAT)],
                None,
                [ts(FLOAT)],
            ),
        ]
    )
    def test_transpose(self, _name, inputs, attributes, expected_outputs):
        actual = run_shape_inference("", "Transpose", inputs, attributes, opset_version=17)
        self.assertEqual(actual, expected_outputs)

    def test_transpose_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference("", "Transpose", [], opset_version=17)

    def test_transpose_none_input(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "Transpose",
                [None],
                opset_version=17,
            )

    def test_transpose_missing_shape(self):
        actual = run_shape_inference(
            "",
            "Transpose",
            [ts(FLOAT)],
            {"perm": ir.Attr("perm", ir.AttributeType.INTS, [1, 0])},
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)


if __name__ == "__main__":
    unittest.main()
