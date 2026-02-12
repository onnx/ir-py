# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Optional operator shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
BOOL = ir.DataType.BOOL


class OptionalTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("optional", "Optional", 15),
            ("optional_get_element", "OptionalGetElement", 18),
        ]
    )
    def test_passthrough(self, _name, op_type, opset_version):
        actual = run_shape_inference(
            "", op_type, [ts(FLOAT, [3, 4])], opset_version=opset_version
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_optional_has_element(self):
        actual = run_shape_inference(
            "", "OptionalHasElement", [ts(FLOAT, [3, 4])], opset_version=18
        )
        self.assertEqual(actual, [ts(BOOL, [])])


if __name__ == "__main__":
    unittest.main()
