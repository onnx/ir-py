# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Softmax / LogSoftmax / Hardmax shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT


class SoftmaxTest(unittest.TestCase):
    @parameterized.parameterized.expand([
        ("Softmax",),
        ("LogSoftmax",),
        ("Hardmax",),
    ])
    def test_passthrough(self, op):
        actual = run_shape_inference(
            "", op, [ts(FLOAT, ["batch", 10])], opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 10])])

    @parameterized.parameterized.expand([
        ("Softmax",),
        ("LogSoftmax",),
        ("Hardmax",),
    ])
    def test_concrete(self, op):
        actual = run_shape_inference(
            "", op, [ts(FLOAT, [3, 4, 5])], opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, 5])])

    @parameterized.parameterized.expand([
        ("Softmax",),
        ("LogSoftmax",),
        ("Hardmax",),
    ])
    def test_missing_shape(self, op):
        actual = run_shape_inference(
            "", op, [ts(FLOAT)], opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT)])


if __name__ == "__main__":
    unittest.main()
