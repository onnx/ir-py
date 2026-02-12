# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Einsum shape inference."""

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


class EinsumTest(unittest.TestCase):
    """Tests for Einsum shape inference."""

    def _einsum(self, equation: str, inputs: list, **kwargs):
        attrs = {"equation": ir.Attr("equation", ir.AttributeType.STRING, equation)}
        return run_shape_inference("", "Einsum", inputs, attrs, opset_version=17, **kwargs)

    @parameterized.parameterized.expand(
        [
            ("matmul", "ij,jk->ik", [[3, 4], [4, 5]], [3, 5]),
            ("batch_matmul", "bij,bjk->bik", [[2, 3, 4], [2, 4, 5]], [2, 3, 5]),
            ("transpose", "ij->ji", [[3, 4]], [4, 3]),
            ("trace", "ii->", [[3, 3]], []),
            ("sum_all", "ij->", [[3, 4]], []),
            ("diagonal", "ii->i", [[3, 3]], [3]),
            ("implicit_matmul", "ij,jk", [[3, 4], [4, 5]], [3, 5]),
            ("implicit_single", "ij", [[3, 4]], [3, 4]),
            ("outer_product", "i,j->ij", [[3], [5]], [3, 5]),
            ("dot_product", "i,i->", [[4], [4]], []),
            ("ellipsis_batch", "...ij,...jk->...ik", [[2, 3, 4], [2, 4, 5]], [2, 3, 5]),
            (
                "ellipsis_multi_batch",
                "...ij,...jk->...ik",
                [[2, 3, 4, 5], [2, 3, 5, 6]],
                [2, 3, 4, 6],
            ),
            (
                "ellipsis_broadcast",
                "...ij,...jk->...ik",
                [[1, 3, 4], [2, 4, 5]],
                [2, 3, 5],
            ),
        ]
    )
    def test_einsum(self, _name, equation, input_shapes, expected_shape):
        inputs = [ts(FLOAT, s) for s in input_shapes]
        actual = self._einsum(equation, inputs)
        self.assertEqual(list(actual[0].shape), expected_shape)

    @parameterized.parameterized.expand(
        [
            ("matmul", "ij,jk->ik", [["M", "K"], ["K", "N"]], ["M", "N"]),
            (
                "batch_matmul",
                "bij,bjk->bik",
                [["B", "M", "K"], ["B", "K", "N"]],
                ["B", "M", "N"],
            ),
            ("transpose", "ij->ji", [["M", "N"]], ["N", "M"]),
            ("outer_product", "i,j->ij", [["M"], ["N"]], ["M", "N"]),
            ("diagonal", "ii->i", [["N", "N"]], ["N"]),
            (
                "ellipsis_batch",
                "...ij,...jk->...ik",
                [["B", "M", "K"], ["B", "K", "N"]],
                ["B", "M", "N"],
            ),
        ]
    )
    def test_symbolic_dims(self, _name, equation, input_shapes, expected_shape):
        inputs = [ts(FLOAT, s) for s in input_shapes]
        actual = self._einsum(equation, inputs)
        self.assertEqual(actual[0].shape, ir.Shape(expected_shape))

    def test_unknown_input_shape_graceful(self):
        attrs = {"equation": ir.Attr("equation", ir.AttributeType.STRING, "ij->ij")}
        actual = run_shape_inference("", "Einsum", [ts(FLOAT)], attrs, opset_version=17)
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_missing_equation_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Einsum",
                [ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4]))],
                opset_version=17,
            )

    def test_wrong_num_inputs_raises(self):
        with self.assertRaises(OpUsageError):
            self._einsum("ij,jk->ik", [ts(FLOAT, [3, 4])])

    def test_rank_mismatch_raises(self):
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


if __name__ == "__main__":
    unittest.main()
