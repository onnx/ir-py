# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for data-dependent output shape operators (NonZero, Compress, Unique)."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class NonZeroTest(unittest.TestCase):
    def test_known_rank(self):
        actual = run_shape_inference("", "NonZero", [ts(FLOAT, [3, 4])], opset_version=17)
        self.assertEqual(actual, [ts(INT64, [2, "_d0"])])

    def test_symbolic_input(self):
        actual = run_shape_inference("", "NonZero", [ts(FLOAT, ["N", 3])], opset_version=17)
        self.assertEqual(actual, [ts(INT64, [2, "_d0"])])

    def test_unknown_rank(self):
        actual = run_shape_inference("", "NonZero", [ts(FLOAT)], opset_version=17)
        self.assertEqual(actual, [ts(INT64, ["_d0", "_d1"])])


class CompressTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference("", "Compress", [ts(FLOAT, ["N", 3])], opset_version=17)
        self.assertEqual(actual, [ts(FLOAT, ["_d0"])])


class UniqueTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "", "Unique", [ts(FLOAT, [5])], opset_version=17, num_outputs=4
        )
        self.assertEqual(actual[0], ts(FLOAT, ["_d0"]))
        self.assertEqual(actual[1].type.dtype, INT64)
        self.assertEqual(actual[2].type.dtype, INT64)
        self.assertEqual(actual[3].type.dtype, INT64)

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "Unique", [None], opset_version=17, num_outputs=4
            )


if __name__ == "__main__":
    unittest.main()
