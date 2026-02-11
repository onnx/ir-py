# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DepthToSpace and SpaceToDepth shape inference."""

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


class DepthToSpaceTest(unittest.TestCase):
    def test_basic(self):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        actual = run_shape_inference(
            "",
            "DepthToSpace",
            [ts(FLOAT, [1, 12, 2, 3])],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 3, 4, 6])])

    def test_symbolic_dims(self):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        actual = run_shape_inference(
            "",
            "DepthToSpace",
            [ts(FLOAT, ["N", "C", "H", "W"])],
            attrs,
            opset_version=13,
        )
        # Symbolic dims produce new symbolic dims for C, H, W
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 4)

    def test_none_input_raises(self):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "DepthToSpace",
                [None],
                attrs,
                opset_version=13,
            )


class SpaceToDepthTest(unittest.TestCase):
    def test_basic(self):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        actual = run_shape_inference(
            "",
            "SpaceToDepth",
            [ts(FLOAT, [1, 3, 4, 6])],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 12, 2, 3])])

    def test_symbolic_dims(self):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        actual = run_shape_inference(
            "",
            "SpaceToDepth",
            [ts(FLOAT, ["N", "C", "H", "W"])],
            attrs,
            opset_version=13,
        )
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 4)

    def test_none_input_raises(self):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "SpaceToDepth",
                [None],
                attrs,
                opset_version=13,
            )


if __name__ == "__main__":
    unittest.main()
