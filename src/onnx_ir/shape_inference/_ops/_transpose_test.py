# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Transpose shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference


class InferTransposeTest(unittest.TestCase):
    """Tests for Transpose shape inference function."""

    @parameterized.parameterized.expand(
        [
            (
                "explicit_perm",
                ["batch", "seq", 256],
                [2, 0, 1],
                "[256,batch,seq]",
            ),
            (
                "swap_last_two",
                ["batch", "seq", 256],
                [0, 2, 1],
                "[batch,256,seq]",
            ),
            (
                "2d_transpose",
                ["batch", 128],
                [1, 0],
                "[128,batch]",
            ),
            (
                "concrete_dims",
                [2, 3, 4],
                [2, 0, 1],
                "[4,2,3]",
            ),
        ]
    )
    def test_output_shape_with_perm(self, _name, input_shape, perm, expected_shape_str):
        x = ir.val("x", ir.DataType.FLOAT, input_shape)
        attrs = {"perm": ir.Attr("perm", ir.AttributeType.INTS, perm)}
        node, _ctx = run_shape_inference("", "Transpose", inputs=[x], attributes=attrs)

        output = node.outputs[0]
        self.assertEqual(str(output.shape), expected_shape_str)

    @parameterized.parameterized.expand(
        [
            (
                "reverse_3d",
                [2, 3, 4],
                "[4,3,2]",
            ),
            (
                "reverse_2d",
                [5, 7],
                "[7,5]",
            ),
            (
                "reverse_symbolic",
                ["batch", "seq"],
                "[seq,batch]",
            ),
        ]
    )
    def test_default_perm_reverses(self, _name, input_shape, expected_shape_str):
        """When no perm attribute is given, dimensions are reversed."""
        x = ir.val("x", ir.DataType.FLOAT, input_shape)
        node, _ctx = run_shape_inference("", "Transpose", inputs=[x])

        self.assertEqual(str(node.outputs[0].shape), expected_shape_str)

    def test_output_dtype_propagated(self):
        x = ir.val("x", ir.DataType.FLOAT16, [2, 3])
        node, _ctx = run_shape_inference("", "Transpose", inputs=[x])

        self.assertEqual(node.outputs[0].dtype, ir.DataType.FLOAT16)

    def test_no_shape_when_input_shape_missing(self):
        x = ir.val("x", ir.DataType.FLOAT)
        node, _ctx = run_shape_inference("", "Transpose", inputs=[x])

        self.assertIsNone(node.outputs[0].shape)


if __name__ == "__main__":
    unittest.main()
