# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cast and CastLike shape inference."""

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
FLOAT16 = ir.DataType.FLOAT16
INT64 = ir.DataType.INT64


class CastTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("float_to_half", FLOAT, [3, 4], FLOAT16),
            ("float_to_int64", FLOAT, ["batch", 128], INT64),
            ("int64_to_float", INT64, [2, 3], FLOAT),
        ]
    )
    def test_cast(self, _name, src_dtype, shape, target_dtype):
        actual = run_shape_inference(
            "",
            "Cast",
            [ts(src_dtype, shape)],
            {"to": ir.Attr("to", ir.AttributeType.INT, target_dtype)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(target_dtype, shape)])

    def test_symbolic_dims(self):
        """Cast ["N", "C"] FLOAT → INT64, same shape with SymbolicDim preserved."""
        actual = run_shape_inference(
            "",
            "Cast",
            [ts(FLOAT, ["N", "C"])],
            {"to": ir.Attr("to", ir.AttributeType.INT, INT64)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, ["N", "C"])])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "Cast",
            [ts(FLOAT)],
            {"to": ir.Attr("to", ir.AttributeType.INT, INT64)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64)])

    def test_cast_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "",
                "Cast",
                [],
                {"to": ir.Attr("to", ir.AttributeType.INT, FLOAT)},
                opset_version=17,
            )

    def test_cast_missing_to(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "",
                "Cast",
                [ts(FLOAT, [3])],
                opset_version=17,
            )

    def test_cast_none_input(self):
        """Cast with a None (missing optional) input."""
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Cast",
                [None],
                {"to": ir.Attr("to", ir.AttributeType.INT, INT64)},
                opset_version=17,
            )


class CastLikeTest(unittest.TestCase):
    def test_cast_like_dtype_from_target(self):
        actual = run_shape_inference(
            "",
            "CastLike",
            [ts(FLOAT, [3, 4]), ts(FLOAT16, [1])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT16, [3, 4])])

    def test_cast_like_preserves_shape(self):
        actual = run_shape_inference(
            "",
            "CastLike",
            [ts(INT64, ["batch", 128]), ts(FLOAT, [2, 3])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 128])])

    def test_cast_like_missing_shape(self):
        actual = run_shape_inference(
            "",
            "CastLike",
            [ts(FLOAT), ts(INT64, [1])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64)])

    def test_cast_like_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                "",
                "CastLike",
                [ts(FLOAT, [3])],
                opset_version=17,
            )

    def test_cast_like_none_target(self):
        """CastLike with None target input."""
        with self.assertRaises(OpUsageError):
            data = ir.Value(name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([3]))
            run_shape_inference_with_values(
                "",
                "CastLike",
                [data, None],
                opset_version=17,
            )


if __name__ == "__main__":
    unittest.main()
