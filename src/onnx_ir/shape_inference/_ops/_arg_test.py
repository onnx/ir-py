# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ArgMax and ArgMin shape inference."""

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


class ArgMaxTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("basic_keepdims", [3, 4, 5], 1, 1, [3, 1, 5]),
            ("no_keepdims", [3, 4, 5], 1, 0, [3, 5]),
            ("default_axis", [3, 4], 0, 1, [1, 4]),
        ]
    )
    def test_argmax(self, _name, input_shape, axis, keepdims, expected_shape):
        attrs = {
            "axis": ir.Attr("axis", ir.AttributeType.INT, axis),
            "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, keepdims),
        }
        actual = run_shape_inference(
            "",
            "ArgMax",
            [ts(FLOAT, input_shape)],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual, [ts(INT64, expected_shape)])

    def test_symbolic_dims(self):
        attrs = {
            "axis": ir.Attr("axis", ir.AttributeType.INT, 1),
            "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1),
        }
        actual = run_shape_inference(
            "",
            "ArgMax",
            [ts(FLOAT, ["N", 4, "D"])],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual, [ts(INT64, ["N", 1, "D"])])

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "ArgMax",
                [None],
                opset_version=13,
            )


class ArgMinTest(unittest.TestCase):
    def test_basic(self):
        attrs = {
            "axis": ir.Attr("axis", ir.AttributeType.INT, 1),
            "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1),
        }
        actual = run_shape_inference(
            "",
            "ArgMin",
            [ts(FLOAT, [3, 4, 5])],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual, [ts(INT64, [3, 1, 5])])

    def test_no_keepdims(self):
        attrs = {
            "axis": ir.Attr("axis", ir.AttributeType.INT, 1),
            "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 0),
        }
        actual = run_shape_inference(
            "",
            "ArgMin",
            [ts(FLOAT, [3, 4, 5])],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual, [ts(INT64, [3, 5])])


if __name__ == "__main__":
    unittest.main()
