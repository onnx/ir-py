# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DepthToSpace and SpaceToDepth shape inference."""

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


class DepthSpaceTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("depth_to_space", "DepthToSpace", [1, 12, 2, 3], [1, 3, 4, 6]),
            ("space_to_depth", "SpaceToDepth", [1, 3, 4, 6], [1, 12, 2, 3]),
        ]
    )
    def test_basic(self, _name, op_type, input_shape, expected_shape):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        actual = run_shape_inference(
            "", op_type, [ts(FLOAT, input_shape)], attrs, opset_version=13
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    @parameterized.parameterized.expand(
        [
            ("depth_to_space", "DepthToSpace"),
            ("space_to_depth", "SpaceToDepth"),
        ]
    )
    def test_symbolic_dims(self, _name, op_type):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        actual = run_shape_inference(
            "", op_type, [ts(FLOAT, ["N", "C", "H", "W"])], attrs, opset_version=13
        )
        result = actual[0]
        self.assertEqual(result.shape[0], ir.SymbolicDim("N"))
        self.assertEqual(result.shape.rank(), 4)

    @parameterized.parameterized.expand(
        [
            ("depth_to_space", "DepthToSpace"),
            ("space_to_depth", "SpaceToDepth"),
        ]
    )
    def test_none_input_raises(self, _name, op_type):
        attrs = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", op_type, [None], attrs, opset_version=13)


if __name__ == "__main__":
    unittest.main()
