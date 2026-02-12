# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Tile shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class TileTest(unittest.TestCase):
    def test_basic(self):
        data = ir.Value(name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 3]))
        repeats = const_value([2, 3], name="repeats")
        actual = run_shape_inference_with_values(
            "",
            "Tile",
            [data, repeats],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 9])])

    def test_unknown_repeats(self):
        data = ir.Value(name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 3]))
        repeats = ir.Value(
            name="repeats",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([2]),
        )
        actual = run_shape_inference_with_values(
            "",
            "Tile",
            [data, repeats],
            opset_version=13,
        )
        # Without const repeats, dtype is preserved but shape is None
        self.assertEqual(actual, [ts(FLOAT)])

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Tile",
                [None],
                opset_version=13,
            )

    def test_symbolic_input_with_const_repeats(self):
        data = ir.Value(
            name="data",
            type=ir.TensorType(FLOAT),
            shape=ir.Shape(["N", "M"]),
        )
        repeats = const_value([2, 3], name="repeats")
        actual = run_shape_inference_with_values(
            "",
            "Tile",
            [data, repeats],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, ["2*N", "3*M"])])


if __name__ == "__main__":
    unittest.main()
