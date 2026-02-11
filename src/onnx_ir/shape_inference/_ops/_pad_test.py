# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Pad shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class PadTest(unittest.TestCase):
    def test_basic_rank_preserved(self):
        """Without const pads, rank is still preserved with symbolic dims."""
        actual = run_shape_inference(
            "",
            "Pad",
            [ts(FLOAT, [2, 3]), ts(ir.DataType.INT64, [4])],
            opset_version=13,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)

    def test_symbolic_input_rank_preserved(self):
        """Pad with symbolic input: ["N", "C"] → rank 2, dims are symbolic."""
        actual = run_shape_inference(
            "",
            "Pad",
            [ts(FLOAT, ["N", "C"]), ts(ir.DataType.INT64, [4])],
            opset_version=13,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)

    def test_const_pads_concrete_dims(self):
        """Pad with const pads on concrete dims: [3, 4] + pads [1, 0, 1, 0] → [5, 4]."""
        data_val = ir.Value(
            name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4])
        )
        pads_val = const_value([1, 0, 1, 0], name="pads")
        actual = run_shape_inference_with_values(
            "", "Pad", [data_val, pads_val], opset_version=13
        )
        self.assertEqual(actual, [ts(FLOAT, [5, 4])])

    def test_const_pads_with_symbolic_dim(self):
        """Pad with const pads but symbolic input dim produces symbolic output dim."""
        data_val = ir.Value(
            name="data", type=ir.TensorType(FLOAT), shape=ir.Shape(["N", 4])
        )
        pads_val = const_value([1, 0, 1, 0], name="pads")
        actual = run_shape_inference_with_values(
            "", "Pad", [data_val, pads_val], opset_version=13
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 4)

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Pad",
                [None],
                opset_version=13,
            )


if __name__ == "__main__":
    unittest.main()
