# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Shape, Size, and Flatten shape inference."""

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


class ShapeOpTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("rank_3", [3, 4, 5], [3]),
            ("rank_2", [3, 4], [2]),
            ("symbolic", ["batch", "seq", 256], [3]),
            ("scalar", [], [0]),
        ]
    )
    def test_shape(self, _name, input_shape, expected_shape):
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, input_shape)],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, expected_shape)])

    def test_shape_with_start_end(self):
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 3, 4, 5])],
            {
                "start": ir.Attr("start", ir.AttributeType.INT, 1),
                "end": ir.Attr("end", ir.AttributeType.INT, 3),
            },
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [2])])

    def test_shape_start_only(self):
        """From ONNX test_shape_start_1."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"start": ir.Attr("start", ir.AttributeType.INT, 1)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [2])])

    def test_shape_end_only(self):
        """From ONNX test_shape_end_1."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"end": ir.Attr("end", ir.AttributeType.INT, 1)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [1])])

    def test_shape_negative_start(self):
        """From ONNX test_shape_negative_start."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"start": ir.Attr("start", ir.AttributeType.INT, -1)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [1])])

    def test_shape_clip_start(self):
        """From ONNX test_shape_clip1: start=-5 clipped to 0."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"start": ir.Attr("start", ir.AttributeType.INT, -5)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [3])])

    def test_shape_clip_end(self):
        """From ONNX test_shape_clip2: end=10 clipped to rank."""
        actual = run_shape_inference(
            "",
            "Shape",
            [ts(FLOAT, [2, 4, 3])],
            {"end": ir.Attr("end", ir.AttributeType.INT, 10)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [3])])


class SizeTest(unittest.TestCase):
    def test_size(self):
        actual = run_shape_inference(
            "",
            "Size",
            [ts(FLOAT, [3, 4, 5])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])

    def test_size_symbolic(self):
        actual = run_shape_inference(
            "",
            "Size",
            [ts(FLOAT, ["batch", 128])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [])])


class FlattenTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            # From ONNX test_flatten: axis=2
            ("axis_2", [2, 3, 4, 5], 2, [6, 20]),
            # From ONNX test_flatten_default_axis: axis=1 default
            ("default_axis", [2, 3, 4], None, [2, 12]),
            # From ONNX test_flatten_zero_axis: axis=0
            ("axis_0", [2, 3, 4], 0, [1, 24]),
            ("axis_end", [2, 3, 4], 3, [24, 1]),
        ]
    )
    def test_flatten(self, _name, input_shape, axis, expected_shape):
        attrs = {}
        if axis is not None:
            attrs["axis"] = ir.Attr("axis", ir.AttributeType.INT, axis)
        actual = run_shape_inference(
            "",
            "Flatten",
            [ts(FLOAT, input_shape)],
            attrs or None,
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_flatten_unknown_dim(self):
        """From ONNX test_flatten_unknown_dim: symbolic dims → unknown dims."""
        actual = run_shape_inference(
            "",
            "Flatten",
            [ts(FLOAT, [2, "N", 4, 5])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 2)},
            opset_version=17,
        )
        # Both output dims should be unknown (symbolic input)
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)

    def test_shape_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Shape", [], opset_version=17)

    def test_shape_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Shape",
                [None],
                opset_version=17,
            )

    def test_size_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Size", [], opset_version=17)

    def test_size_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Size",
                [None],
                opset_version=17,
            )

    def test_flatten_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Flatten", [], opset_version=17)

    def test_flatten_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Flatten",
                [None],
                opset_version=17,
            )

    def test_flatten_missing_shape(self):
        """Flatten with unknown input shape → dtype only."""
        actual = run_shape_inference("", "Flatten", [ts(FLOAT)], opset_version=17)
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)


class UniqueTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "", "Unique", [ts(FLOAT, [5])], opset_version=17, num_outputs=4
        )
        # Y: 1-D with symbolic length
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 1)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].type.dtype, FLOAT)
        # indices: 1-D INT64, same symbolic length as Y
        self.assertEqual(actual[1].shape.rank(), 1)
        self.assertEqual(actual[1].type.dtype, INT64)
        # inverse_indices: 1-D INT64
        self.assertEqual(actual[2].shape.rank(), 1)
        self.assertEqual(actual[2].type.dtype, INT64)
        # counts: 1-D INT64, same symbolic length as Y
        self.assertEqual(actual[3].shape.rank(), 1)
        self.assertEqual(actual[3].type.dtype, INT64)

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "Unique", [None], opset_version=17, num_outputs=4
            )


class EinsumTest(unittest.TestCase):
    """Tests for Einsum shape inference."""

    def _einsum(self, equation: str, inputs: list, **kwargs):
        attrs = {"equation": ir.Attr("equation", ir.AttributeType.STRING, equation)}
        return run_shape_inference("", "Einsum", inputs, attrs, opset_version=17, **kwargs)

    def test_matmul(self):
        """ij,jk->ik: matrix multiply."""
        actual = self._einsum("ij,jk->ik", [ts(FLOAT, [3, 4]), ts(FLOAT, [4, 5])])
        self.assertEqual(list(actual[0].shape), [3, 5])

    def test_batch_matmul(self):
        """bij,bjk->bik: batch matmul."""
        actual = self._einsum("bij,bjk->bik", [ts(FLOAT, [2, 3, 4]), ts(FLOAT, [2, 4, 5])])
        self.assertEqual(list(actual[0].shape), [2, 3, 5])

    def test_transpose(self):
        """ij->ji: transpose."""
        actual = self._einsum("ij->ji", [ts(FLOAT, [3, 4])])
        self.assertEqual(list(actual[0].shape), [4, 3])

    def test_trace(self):
        """ii->: trace (scalar output)."""
        actual = self._einsum("ii->", [ts(FLOAT, [3, 3])])
        self.assertEqual(list(actual[0].shape), [])

    def test_sum_all(self):
        """ij->: sum to scalar."""
        actual = self._einsum("ij->", [ts(FLOAT, [3, 4])])
        self.assertEqual(list(actual[0].shape), [])

    def test_diagonal(self):
        """ii->i: diagonal."""
        actual = self._einsum("ii->i", [ts(FLOAT, [3, 3])])
        self.assertEqual(list(actual[0].shape), [3])

    def test_implicit_output(self):
        """ij,jk (no ->): implicit output is sorted non-repeated labels ik."""
        actual = self._einsum("ij,jk", [ts(FLOAT, [3, 4]), ts(FLOAT, [4, 5])])
        self.assertEqual(list(actual[0].shape), [3, 5])

    def test_implicit_single_input(self):
        """Ij (no ->): implicit output is 'ij' (sorted non-repeated)."""
        actual = self._einsum("ij", [ts(FLOAT, [3, 4])])
        self.assertEqual(list(actual[0].shape), [3, 4])

    def test_outer_product(self):
        """i,j->ij: outer product."""
        actual = self._einsum("i,j->ij", [ts(FLOAT, [3]), ts(FLOAT, [5])])
        self.assertEqual(list(actual[0].shape), [3, 5])

    def test_dot_product(self):
        """i,i->: dot product."""
        actual = self._einsum("i,i->", [ts(FLOAT, [4]), ts(FLOAT, [4])])
        self.assertEqual(list(actual[0].shape), [])

    def test_symbolic_dims(self):
        """ij,jk->ik with symbolic dims."""
        actual = self._einsum("ij,jk->ik", [ts(FLOAT, ["M", "K"]), ts(FLOAT, ["K", "N"])])
        self.assertEqual(actual[0].shape.rank(), 2)

    def test_ellipsis_batch_matmul(self):
        """...ij,...jk->...ik: ellipsis batch matmul."""
        actual = self._einsum(
            "...ij,...jk->...ik", [ts(FLOAT, [2, 3, 4]), ts(FLOAT, [2, 4, 5])]
        )
        self.assertEqual(list(actual[0].shape), [2, 3, 5])

    def test_ellipsis_multi_batch(self):
        """...ij,...jk->...ik with 2 batch dims."""
        actual = self._einsum(
            "...ij,...jk->...ik",
            [ts(FLOAT, [2, 3, 4, 5]), ts(FLOAT, [2, 3, 5, 6])],
        )
        self.assertEqual(list(actual[0].shape), [2, 3, 4, 6])

    def test_ellipsis_broadcast_1(self):
        """Ellipsis broadcast: [1, 3, 4] and [2, 4, 5] → batch dim 2."""
        actual = self._einsum(
            "...ij,...jk->...ik", [ts(FLOAT, [1, 3, 4]), ts(FLOAT, [2, 4, 5])]
        )
        self.assertEqual(list(actual[0].shape), [2, 3, 5])

    def test_missing_equation_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Einsum",
                [ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4]))],
                opset_version=17,
            )

    def test_wrong_num_inputs_raises(self):
        """Equation has 2 operands but only 1 input."""
        with self.assertRaises(OpUsageError):
            self._einsum("ij,jk->ik", [ts(FLOAT, [3, 4])])

    def test_rank_mismatch_raises(self):
        """Rank of input doesn't match equation."""
        with self.assertRaises(OpUsageError):
            self._einsum("ijk->ijk", [ts(FLOAT, [3, 4])])

    def test_no_inputs_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Einsum",
                [None],
                {"equation": ir.Attr("equation", ir.AttributeType.STRING, "i->i")},
                opset_version=17,
            )

    def test_unknown_input_shape_graceful(self):
        """When an input has no shape, output shape is None."""
        attrs = {"equation": ir.Attr("equation", ir.AttributeType.STRING, "ij->ij")}
        actual = run_shape_inference("", "Einsum", [ts(FLOAT)], attrs, opset_version=17)
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)


if __name__ == "__main__":
    unittest.main()
