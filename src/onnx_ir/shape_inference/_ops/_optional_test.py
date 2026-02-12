# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Optional operator shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
BOOL = ir.DataType.BOOL


class OptionalTest(unittest.TestCase):
    def test_optional(self):
        actual = run_shape_inference("", "Optional", [ts(FLOAT, [3, 4])], opset_version=15)
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_optional_get_element(self):
        actual = run_shape_inference(
            "", "OptionalGetElement", [ts(FLOAT, [3, 4])], opset_version=18
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_optional_has_element(self):
        actual = run_shape_inference(
            "", "OptionalHasElement", [ts(FLOAT, [3, 4])], opset_version=18
        )
        self.assertEqual(actual, [ts(BOOL, [])])


if __name__ == "__main__":
    unittest.main()
