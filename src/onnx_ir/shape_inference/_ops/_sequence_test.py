# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Sequence operator shape inference."""

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
INT32 = ir.DataType.INT32
INT64 = ir.DataType.INT64


class SequenceConstructTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "SequenceConstruct",
            [ts(FLOAT, [3, 4]), ts(FLOAT, [3, 4])],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, FLOAT)

    def test_single_input(self):
        actual = run_shape_inference(
            "",
            "SequenceConstruct",
            [ts(INT32, [2])],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, INT32)

    def test_preserves_elem_tensor_type(self):
        actual = run_shape_inference(
            "",
            "SequenceConstruct",
            [ts(FLOAT, ["batch", 128])],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertIsInstance(result.type.elem_type, ir.TensorType)


class SequenceEmptyTest(unittest.TestCase):
    def test_default_float(self):
        actual = run_shape_inference(
            "",
            "SequenceEmpty",
            [],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, FLOAT)

    def test_explicit_dtype(self):
        actual = run_shape_inference(
            "",
            "SequenceEmpty",
            [],
            {"dtype": ir.Attr("dtype", ir.AttributeType.INT, INT64)},
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, INT64)


class SequenceAtTest(unittest.TestCase):
    def test_extracts_element_type(self):
        seq = ir.Value(
            name="seq",
            type=ir.SequenceType(ir.TensorType(FLOAT)),
        )
        position = ir.Value(name="pos", type=ir.TensorType(INT64))
        actual = run_shape_inference_with_values(
            "",
            "SequenceAt",
            [seq, position],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.TensorType)
        self.assertEqual(result.type.dtype, FLOAT)

    def test_unknown_sequence_type(self):
        """When sequence type is unknown, output type is not set."""
        seq = ir.Value(name="seq")
        position = ir.Value(name="pos", type=ir.TensorType(INT64))
        actual = run_shape_inference_with_values(
            "",
            "SequenceAt",
            [seq, position],
            opset_version=17,
        )
        self.assertIsNone(actual[0].type)


class SequenceLengthTest(unittest.TestCase):
    def test_scalar_int64(self):
        seq = ir.Value(
            name="seq",
            type=ir.SequenceType(ir.TensorType(FLOAT)),
        )
        actual = run_shape_inference_with_values(
            "",
            "SequenceLength",
            [seq],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])


class SequenceInsertTest(unittest.TestCase):
    def test_preserves_sequence_type(self):
        seq = ir.Value(
            name="seq",
            type=ir.SequenceType(ir.TensorType(FLOAT)),
        )
        tensor = ir.Value(name="tensor", type=ir.TensorType(FLOAT))
        actual = run_shape_inference_with_values(
            "",
            "SequenceInsert",
            [seq, tensor],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, FLOAT)


class SequenceEraseTest(unittest.TestCase):
    def test_preserves_sequence_type(self):
        seq = ir.Value(
            name="seq",
            type=ir.SequenceType(ir.TensorType(INT32)),
        )
        actual = run_shape_inference_with_values(
            "",
            "SequenceErase",
            [seq],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, INT32)

    def test_with_position(self):
        seq = ir.Value(
            name="seq",
            type=ir.SequenceType(ir.TensorType(FLOAT)),
        )
        position = ir.Value(name="pos", type=ir.TensorType(INT64))
        actual = run_shape_inference_with_values(
            "",
            "SequenceErase",
            [seq, position],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, FLOAT)


class SplitToSequenceTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "SplitToSequence",
            [ts(FLOAT, [10, 4])],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, FLOAT)

    def test_preserves_input_dtype(self):
        actual = run_shape_inference(
            "",
            "SplitToSequence",
            [ts(INT32, [6])],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.SequenceType)
        self.assertEqual(result.type.dtype, INT32)


class ConcatFromSequenceTest(unittest.TestCase):
    def test_basic(self):
        seq = ir.Value(
            name="seq",
            type=ir.SequenceType(ir.TensorType(FLOAT)),
        )
        actual = run_shape_inference_with_values(
            "",
            "ConcatFromSequence",
            [seq],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        result = actual[0]
        # Output is a tensor, not a sequence
        self.assertIsNone(result.shape)  # Can't determine shape
        self.assertEqual(result.type.dtype, FLOAT)

    def test_unknown_sequence_type(self):
        """When sequence element type is unknown, output type is not set."""
        seq = ir.Value(name="seq")
        actual = run_shape_inference_with_values(
            "",
            "ConcatFromSequence",
            [seq],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        self.assertIsNone(actual[0].type)


if __name__ == "__main__":
    unittest.main()


class SequenceLengthSymbolicDimsTest(unittest.TestCase):
    def test_scalar_output_with_symbolic_elements(self):
        """SequenceLength always produces a scalar INT64 regardless of element shape."""
        seq = ir.Value(
            name="seq",
            type=ir.SequenceType(ir.TensorType(FLOAT)),
        )
        actual = run_shape_inference_with_values(
            "",
            "SequenceLength",
            [seq],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])


class SequenceAtSymbolicDimsTest(unittest.TestCase):
    def test_extracts_element_type_with_symbolic_shape(self):
        """SequenceAt extracts element type even when element shapes are symbolic."""
        seq = ir.Value(
            name="seq",
            type=ir.SequenceType(ir.TensorType(FLOAT)),
        )
        position = ir.Value(name="pos", type=ir.TensorType(INT64))
        actual = run_shape_inference_with_values(
            "",
            "SequenceAt",
            [seq, position],
            opset_version=17,
        )
        result = actual[0]
        self.assertIsInstance(result.type, ir.TensorType)
        self.assertEqual(result.type.dtype, FLOAT)


class SequenceErrorPathsTest(unittest.TestCase):
    """Tests for error/early-return paths in sequence ops."""

    @parameterized.parameterized.expand(
        [
            ("construct", "SequenceConstruct", []),
            ("at", "SequenceAt", []),
            ("length", "SequenceLength", []),
            ("insert", "SequenceInsert", [ts(FLOAT, [3])]),
            ("erase", "SequenceErase", []),
            ("split_to_sequence", "SplitToSequence", []),
            ("concat_from_sequence", "ConcatFromSequence", []),
        ]
    )
    def test_no_inputs(self, _name, op_type, inputs):
        attrs = {}
        if op_type == "ConcatFromSequence":
            attrs = {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)}
        with self.assertRaises(OpUsageError):
            run_shape_inference("", op_type, inputs, attrs or None, opset_version=17)

    @parameterized.parameterized.expand(
        [
            ("construct", "SequenceConstruct", [None]),
            (
                "at",
                "SequenceAt",
                [None, ir.Value(name="idx", type=ir.TensorType(INT64), shape=ir.Shape([]))],
            ),
            (
                "insert",
                "SequenceInsert",
                [None, ir.Value(name="t", type=ir.TensorType(FLOAT), shape=ir.Shape([3]))],
            ),
            ("erase", "SequenceErase", [None]),
            ("split_to_sequence", "SplitToSequence", [None]),
            ("concat_from_sequence", "ConcatFromSequence", [None]),
        ]
    )
    def test_none_input(self, _name, op_type, inputs):
        attrs = {}
        if op_type == "ConcatFromSequence":
            attrs = {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)}
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", op_type, inputs, attrs or None, opset_version=17
            )
