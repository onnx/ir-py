# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Add shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
DOUBLE = ir.DataType.DOUBLE


class InferAddTest(unittest.TestCase):
    """Tests for Add shape inference function."""

    @parameterized.parameterized.expand(
        [
            (
                "broadcast_with_symbolic",
                [ts(FLOAT, ["batch", 128]), ts(FLOAT, [1, 128])],
                [ts(FLOAT, ["batch", 128])],
            ),
            (
                "same_shape",
                [ts(FLOAT, [3, 4, 5]), ts(FLOAT, [3, 4, 5])],
                [ts(FLOAT, [3, 4, 5])],
            ),
            (
                "broadcast_ones",
                [ts(FLOAT, [3, 1, 5]), ts(FLOAT, [1, 4, 5])],
                [ts(FLOAT, [3, 4, 5])],
            ),
            (
                "broadcast_different_ranks",
                [ts(FLOAT, [128]), ts(FLOAT, ["batch", 128])],
                [ts(FLOAT, ["batch", 128])],
            ),
            (
                "scalar_broadcast",
                [ts(FLOAT, [1]), ts(FLOAT, ["batch", "seq", 256])],
                [ts(FLOAT, ["batch", "seq", 256])],
            ),
            (
                "dtype_from_second_input",
                [ts(shape=[2, 3]), ts(DOUBLE, [2, 3])],
                [ts(DOUBLE, [2, 3])],
            ),
            (
                "no_shape_when_input_shape_missing",
                [ts(FLOAT), ts(FLOAT, [2, 3])],
                [ts(FLOAT)],
            ),
            (
                "incompatible_shapes",
                [ts(FLOAT, [3, 4]), ts(FLOAT, [5, 4])],
                [ts(FLOAT)],
            ),
        ]
    )
    def test_add(self, _name, inputs, expected_outputs):
        actual = run_shape_inference("", "Add", inputs, opset_version=17)
        self.assertEqual(actual, expected_outputs)


if __name__ == "__main__":
    unittest.main()
