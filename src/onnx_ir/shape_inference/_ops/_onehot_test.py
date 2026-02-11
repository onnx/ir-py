# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OneHot shape inference."""

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


class OneHotTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "OneHot",
            [ts(INT64, [3, 4]), ts(INT64, []), ts(FLOAT, [2])],
            opset_version=21,
        )
        self.assertIsNotNone(actual[0].shape)
        # indices [3,4] -> output rank 3, default axis=-1 inserts at end
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertEqual(actual[0].shape[0], 3)
        self.assertEqual(actual[0].shape[1], 4)
        # depth dim is symbolic
        self.assertEqual(actual[0].type, ir.TensorType(FLOAT))

    def test_none_input_raises(self):
        v_depth = ir.Value(name="depth", type=ir.TensorType(INT64), shape=ir.Shape([]))
        v_values = ir.Value(name="values", type=ir.TensorType(FLOAT), shape=ir.Shape([2]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "OneHot", [None, v_depth, v_values], opset_version=21
            )


if __name__ == "__main__":
    unittest.main()
