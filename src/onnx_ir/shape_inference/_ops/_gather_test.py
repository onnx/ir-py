# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gather shape inference."""

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


class GatherTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "axis_0",
                [5, 4, 3],
                [2],
                0,
                [2, 4, 3],
            ),
            (
                "axis_1",
                [5, 4, 3],
                [2, 2],
                1,
                [5, 2, 2, 3],
            ),
            (
                "scalar_indices",
                [5, 4, 3],
                [],
                0,
                [4, 3],
            ),
            (
                "symbolic",
                ["vocab", 256],
                ["batch", "seq"],
                0,
                ["batch", "seq", 256],
            ),
            (
                "negative_axis",
                [5, 4, 3],
                [2],
                -1,
                [5, 4, 2],
            ),
        ]
    )
    def test_gather(self, _name, data_shape, indices_shape, axis, expected_shape):
        actual = run_shape_inference(
            "",
            "Gather",
            [ts(FLOAT, data_shape), ts(INT64, indices_shape)],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, axis)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_default_axis(self):
        actual = run_shape_inference(
            "",
            "Gather",
            [ts(FLOAT, [5, 4]), ts(INT64, [3])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_gather_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "",
                "Gather",
                [],
                opset_version=17,
            )

    def test_gather_none_data(self):
        indices = ir.Value(name="idx", type=ir.TensorType(INT64), shape=ir.Shape([3]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Gather",
                [None, indices],
                {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
                opset_version=17,
            )

    def test_gather_missing_shapes(self):
        actual = run_shape_inference(
            "",
            "Gather",
            [ts(FLOAT), ts(INT64, [3])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)


if __name__ == "__main__":
    unittest.main()
