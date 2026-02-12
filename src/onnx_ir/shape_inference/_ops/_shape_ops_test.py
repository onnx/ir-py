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
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)

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


class NonZeroTest(unittest.TestCase):
    def test_symbolic_input(self):
        """NonZero on ["N", 3] → rank-2 output with symbolic dims."""
        actual = run_shape_inference("", "NonZero", [ts(FLOAT, ["N", 3])], opset_version=17)
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)
        self.assertEqual(result.shape[0], 2)
        self.assertIsInstance(result.shape[1], ir.SymbolicDim)
        self.assertEqual(result.type.dtype, INT64)


class DetTest(unittest.TestCase):
    def test_det_basic(self):
        actual = run_shape_inference("", "Det", [ts(FLOAT, [3, 3])], opset_version=17)
        self.assertEqual(actual, [ts(FLOAT, [])])

    def test_det_batched(self):
        actual = run_shape_inference("", "Det", [ts(FLOAT, [2, 5, 3, 3])], opset_version=17)
        self.assertEqual(actual, [ts(FLOAT, [2, 5])])


class NonZeroConcreteTest(unittest.TestCase):
    def test_non_zero_concrete_rank(self):
        """NonZero on [3, 4] → output shape [2, symbolic]."""
        actual = run_shape_inference("", "NonZero", [ts(FLOAT, [3, 4])], opset_version=17)
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape[0], 2)
        self.assertIsInstance(result.shape[1], ir.SymbolicDim)

    def test_non_zero_no_shape(self):
        """NonZero without input shape → both dims are symbolic."""
        actual = run_shape_inference("", "NonZero", [ts(FLOAT)], opset_version=17)
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 2)
        self.assertIsInstance(result.shape[0], ir.SymbolicDim)
        self.assertIsInstance(result.shape[1], ir.SymbolicDim)


class StringSplitTest(unittest.TestCase):
    def test_string_split_basic(self):
        STRING = ir.DataType.STRING
        actual = run_shape_inference(
            "", "StringSplit", [ts(STRING, [3])], opset_version=22, num_outputs=2
        )
        # Y: rank 2 (X.rank + 1)
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertEqual(actual[0].shape[0], 3)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].type.dtype, STRING)
        # Z: same shape as X
        self.assertEqual(actual[1].shape, ir.Shape([3]))
        self.assertEqual(actual[1].type.dtype, INT64)

    def test_string_split_no_shape(self):
        STRING = ir.DataType.STRING
        actual = run_shape_inference(
            "", "StringSplit", [ts(STRING)], opset_version=22, num_outputs=2
        )
        self.assertIsNone(actual[0].shape)


class StringNormalizerTest(unittest.TestCase):
    def test_string_normalizer_basic(self):
        STRING = ir.DataType.STRING
        actual = run_shape_inference(
            "", "StringNormalizer", [ts(STRING, [5])], opset_version=10
        )
        self.assertEqual(actual, [ts(STRING, [5])])


class ScanTest(unittest.TestCase):
    def test_scan_basic(self):
        """Scan with a body graph: 1 state var + 1 scan output."""
        # Body graph: 1 input (state), 2 outputs (state_out, scan_out)
        body_state_in = ir.Value(
            name="state_in", type=ir.TensorType(FLOAT), shape=ir.Shape([2])
        )
        body_state_out = ir.Value(
            name="state_out", type=ir.TensorType(FLOAT), shape=ir.Shape([2])
        )
        body_scan_out = ir.Value(
            name="scan_out", type=ir.TensorType(FLOAT), shape=ir.Shape([3])
        )
        body_graph = ir.Graph(
            inputs=[body_state_in],
            outputs=[body_state_out, body_scan_out],
            nodes=[],
            name="scan_body",
        )
        attrs = {
            "body": ir.Attr("body", ir.AttributeType.GRAPH, body_graph),
            "num_scan_inputs": ir.Attr("num_scan_inputs", ir.AttributeType.INT, 0),
        }
        # 1 input (the state), 0 scan inputs → num_state = 1
        state_val = ir.Value(name="state", type=ir.TensorType(FLOAT), shape=ir.Shape([2]))
        actual = run_shape_inference_with_values(
            "", "Scan", [state_val], attrs, opset_version=11, num_outputs=2
        )
        # State output: same shape as body state output
        self.assertEqual(actual[0].shape, ir.Shape([2]))
        # Scan output: symbolic_dim prepended to body scan output shape
        self.assertIsNotNone(actual[1].shape)
        self.assertEqual(actual[1].shape.rank(), 2)
        self.assertIsInstance(actual[1].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[1].shape[1], 3)

    def test_scan_no_body(self):
        """Scan without body graph → all outputs are None."""
        state_val = ir.Value(name="state", type=ir.TensorType(FLOAT), shape=ir.Shape([2]))
        actual = run_shape_inference_with_values(
            "", "Scan", [state_val], opset_version=11, num_outputs=2
        )
        self.assertIsNone(actual[0].shape)
        self.assertIsNone(actual[1].shape)


class ImageDecoderTest(unittest.TestCase):
    def test_image_decoder_basic(self):
        UINT8 = ir.DataType.UINT8
        actual = run_shape_inference("", "ImageDecoder", [ts(UINT8, [100])], opset_version=20)
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[2], 3)
        self.assertEqual(actual[0].type.dtype, UINT8)

    def test_image_decoder_grayscale(self):
        UINT8 = ir.DataType.UINT8
        attrs = {
            "pixel_format": ir.Attr("pixel_format", ir.AttributeType.STRING, "Grayscale"),
        }
        actual = run_shape_inference(
            "", "ImageDecoder", [ts(UINT8, [100])], attrs, opset_version=20
        )
        self.assertEqual(actual[0].shape[2], 1)


class MelWeightMatrixTest(unittest.TestCase):
    def test_mel_weight_matrix_basic(self):
        actual = run_shape_inference(
            "",
            "MelWeightMatrix",
            [ts(INT64, []), ts(INT64, []), ts(FLOAT, []), ts(FLOAT, []), ts(INT64, [])],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)
        self.assertEqual(actual[0].type.dtype, FLOAT)


class TfIdfVectorizerTest(unittest.TestCase):
    def test_tfidf_vectorizer_1d(self):
        actual = run_shape_inference("", "TfIdfVectorizer", [ts(INT64, [10])], opset_version=9)
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 1)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_tfidf_vectorizer_2d(self):
        actual = run_shape_inference(
            "", "TfIdfVectorizer", [ts(INT64, [4, 10])], opset_version=9
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertEqual(actual[0].shape[0], 4)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)


class OptionalTest(unittest.TestCase):
    def test_optional_basic(self):
        actual = run_shape_inference("", "Optional", [ts(FLOAT, [3, 4])], opset_version=15)
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_optional_get_element(self):
        actual = run_shape_inference(
            "", "OptionalGetElement", [ts(FLOAT, [3, 4])], opset_version=18
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_optional_has_element(self):
        BOOL = ir.DataType.BOOL
        actual = run_shape_inference(
            "", "OptionalHasElement", [ts(FLOAT, [3, 4])], opset_version=18
        )
        self.assertEqual(actual, [ts(BOOL, [])])


class TensorScatterTest(unittest.TestCase):
    def test_tensor_scatter_basic(self):
        actual = run_shape_inference(
            "",
            "TensorScatter",
            [ts(FLOAT, [5, 3]), ts(INT64, [2, 1]), ts(FLOAT, [2, 3])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, [5, 3])])


class CompressTest(unittest.TestCase):
    def test_symbolic_input(self):
        """Compress on ["N", 3] → 1D output with symbolic dim."""
        actual = run_shape_inference("", "Compress", [ts(FLOAT, ["N", 3])], opset_version=17)
        result = actual[0]
        self.assertIsNotNone(result.shape)
        self.assertEqual(result.shape.rank(), 1)
        self.assertIsInstance(result.shape[0], ir.SymbolicDim)


if __name__ == "__main__":
    unittest.main()
