# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Pad shape inference."""

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
