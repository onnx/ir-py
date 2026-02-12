# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for string operator shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

STRING = ir.DataType.STRING
INT64 = ir.DataType.INT64


class StringSplitTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "", "StringSplit", [ts(STRING, [3])], opset_version=22, num_outputs=2
        )
        self.assertEqual(actual[0], ts(STRING, [3, "_d0"]))
        self.assertEqual(actual[1], ts(INT64, [3]))

    def test_no_shape(self):
        actual = run_shape_inference(
            "", "StringSplit", [ts(STRING)], opset_version=22, num_outputs=2
        )
        self.assertIsNone(actual[0].shape)


class StringNormalizerTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "", "StringNormalizer", [ts(STRING, [5])], opset_version=10
        )
        self.assertEqual(actual, [ts(STRING, [5])])


if __name__ == "__main__":
    unittest.main()
