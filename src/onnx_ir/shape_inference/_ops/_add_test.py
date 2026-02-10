# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Add shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference


class InferAddTest(unittest.TestCase):
    """Tests for Add shape inference function."""

    @parameterized.parameterized.expand(
        [
            (
                "broadcast_with_symbolic",
                ["batch", 128],
                [1, 128],
                "[batch,128]",
            ),
            (
                "same_shape",
                [3, 4, 5],
                [3, 4, 5],
                "[3,4,5]",
            ),
            (
                "broadcast_ones",
                [3, 1, 5],
                [1, 4, 5],
                "[3,4,5]",
            ),
            (
                "broadcast_different_ranks",
                [128],
                ["batch", 128],
                "[batch,128]",
            ),
            (
                "scalar_broadcast",
                [1],
                ["batch", "seq", 256],
                "[batch,seq,256]",
            ),
        ]
    )
    def test_output_shape(self, _name, shape_a, shape_b, expected_shape_str):
        x = ir.val("x", ir.DataType.FLOAT, shape_a)
        y = ir.val("y", ir.DataType.FLOAT, shape_b)
        node, _ctx = run_shape_inference("", "Add", inputs=[x, y])

        output = node.outputs[0]
        self.assertEqual(str(output.shape), expected_shape_str)

    def test_output_dtype_propagated(self):
        x = ir.val("x", ir.DataType.FLOAT, [2, 3])
        y = ir.val("y", ir.DataType.FLOAT, [2, 3])
        node, _ctx = run_shape_inference("", "Add", inputs=[x, y])

        self.assertEqual(node.outputs[0].dtype, ir.DataType.FLOAT)

    def test_output_dtype_from_second_input(self):
        """When first input has no dtype, dtype comes from the second."""
        x = ir.val("x", shape=[2, 3])
        y = ir.val("y", ir.DataType.DOUBLE, [2, 3])
        node, _ctx = run_shape_inference("", "Add", inputs=[x, y])

        self.assertEqual(node.outputs[0].dtype, ir.DataType.DOUBLE)

    def test_no_shape_when_input_shape_missing(self):
        """When an input has no shape, output shape is None."""
        x = ir.val("x", ir.DataType.FLOAT)
        y = ir.val("y", ir.DataType.FLOAT, [2, 3])
        node, _ctx = run_shape_inference("", "Add", inputs=[x, y])

        self.assertIsNone(node.outputs[0].shape)

    def test_incompatible_shapes_produce_no_shape(self):
        x = ir.val("x", ir.DataType.FLOAT, [3, 4])
        y = ir.val("y", ir.DataType.FLOAT, [5, 4])
        node, _ctx = run_shape_inference("", "Add", inputs=[x, y])

        self.assertIsNone(node.outputs[0].shape)


if __name__ == "__main__":
    unittest.main()
