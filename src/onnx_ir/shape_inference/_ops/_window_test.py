# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for window function shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class WindowTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("blackman", "BlackmanWindow", 8),
            ("hann", "HannWindow", 16),
            ("hamming", "HammingWindow", 10),
        ]
    )
    def test_window_const_size(self, _name, op_type, size):
        actual = run_shape_inference_with_values(
            "", op_type, [const_value([size])], opset_version=21
        )
        self.assertEqual(actual, [ts(FLOAT, [size])])

    @parameterized.parameterized.expand(
        [
            ("blackman", "BlackmanWindow"),
            ("hann", "HannWindow"),
            ("hamming", "HammingWindow"),
        ]
    )
    def test_window_dynamic_size(self, _name, op_type):
        size = ir.Value(name="size", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([]))
        actual = run_shape_inference_with_values("", op_type, [size], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, ["_d0"])])


if __name__ == "__main__":
    unittest.main()
