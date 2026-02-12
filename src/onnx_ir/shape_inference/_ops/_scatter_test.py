# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ScatterElements, ScatterND, and TensorScatter shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class ScatterTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("scatter_elements", "ScatterElements", [3, 4]),
            ("scatter_nd", "ScatterND", [4, 5, 6]),
        ]
    )
    def test_basic(self, _name, op_type, input_shape):
        actual = run_shape_inference("", op_type, [ts(FLOAT, input_shape)], opset_version=18)
        self.assertEqual(actual, [ts(FLOAT, input_shape)])

    @parameterized.parameterized.expand(
        [
            ("scatter_elements", "ScatterElements", ["N", "C"]),
            ("scatter_nd", "ScatterND", ["N", "C", "D"]),
        ]
    )
    def test_symbolic_dims(self, _name, op_type, input_shape):
        actual = run_shape_inference("", op_type, [ts(FLOAT, input_shape)], opset_version=18)
        self.assertEqual(actual, [ts(FLOAT, input_shape)])

    @parameterized.parameterized.expand(
        [
            ("scatter_elements", "ScatterElements"),
            ("scatter_nd", "ScatterND"),
        ]
    )
    def test_none_input_raises(self, _name, op_type):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", op_type, [None], opset_version=18)


class TensorScatterTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "TensorScatter",
            [ts(FLOAT, [5, 3]), ts(INT64, [2, 1]), ts(FLOAT, [2, 3])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, [5, 3])])


if __name__ == "__main__":
    unittest.main()
