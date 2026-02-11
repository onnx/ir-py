# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Sequence operator shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import InvalidOpUsageError
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


class SequenceErrorPathsTest(unittest.TestCase):
    """Tests for error/early-return paths in sequence ops."""

    def test_construct_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference(
                "",
                "SequenceConstruct",
                [],
                opset_version=17,
            )

    def test_construct_none_input(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "SequenceConstruct",
                [None],
                opset_version=17,
            )

    def test_at_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference(
                "",
                "SequenceAt",
                [],
                opset_version=17,
            )

    def test_at_none_seq(self):
        with self.assertRaises(InvalidOpUsageError):
            idx = ir.Value(name="idx", type=ir.TensorType(INT64), shape=ir.Shape([]))
            run_shape_inference_with_values(
                "",
                "SequenceAt",
                [None, idx],
                opset_version=17,
            )

    def test_length_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference(
                "",
                "SequenceLength",
                [],
                opset_version=17,
            )

    def test_insert_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference(
                "",
                "SequenceInsert",
                [ts(FLOAT, [3])],
                opset_version=17,
            )

    def test_insert_none_seq(self):
        with self.assertRaises(InvalidOpUsageError):
            tensor = ir.Value(name="t", type=ir.TensorType(FLOAT), shape=ir.Shape([3]))
            run_shape_inference_with_values(
                "",
                "SequenceInsert",
                [None, tensor],
                opset_version=17,
            )

    def test_erase_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference(
                "",
                "SequenceErase",
                [],
                opset_version=17,
            )

    def test_erase_none_seq(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "SequenceErase",
                [None],
                opset_version=17,
            )

    def test_split_to_sequence_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference(
                "",
                "SplitToSequence",
                [],
                opset_version=17,
            )

    def test_split_to_sequence_none_data(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "SplitToSequence",
                [None],
                opset_version=17,
            )

    def test_concat_from_sequence_no_inputs(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference(
                "",
                "ConcatFromSequence",
                [],
                {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
                opset_version=17,
            )

    def test_concat_from_sequence_none_seq(self):
        with self.assertRaises(InvalidOpUsageError):
            run_shape_inference_with_values(
                "",
                "ConcatFromSequence",
                [None],
                {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
                opset_version=17,
            )
