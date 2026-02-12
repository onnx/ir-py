# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for loss operator shape inference."""

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


class NegativeLogLikelihoodLossTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("mean", "mean", []),
            ("sum", "sum", []),
            ("none", "none", [2]),
        ]
    )
    def test_reduction(self, _name, reduction, expected_shape):
        attrs = {
            "reduction": ir.Attr("reduction", ir.AttributeType.STRING, reduction),
        }
        actual = run_shape_inference(
            "",
            "NegativeLogLikelihoodLoss",
            [ts(FLOAT, [2, 3]), ts(INT64, [2])],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_default_reduction_is_mean(self):
        actual = run_shape_inference(
            "",
            "NegativeLogLikelihoodLoss",
            [ts(FLOAT, [2, 3]), ts(INT64, [2])],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [])])

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "NegativeLogLikelihoodLoss", [], opset_version=13)

    def test_none_input(self):
        v = ir.Value(name="input", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 3]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "NegativeLogLikelihoodLoss",
                [v, None],
                opset_version=13,
            )


class SoftmaxCrossEntropyLossTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("mean", "mean", []),
            ("none", "none", [2]),
        ]
    )
    def test_reduction(self, _name, reduction, expected_shape):
        attrs = {
            "reduction": ir.Attr("reduction", ir.AttributeType.STRING, reduction),
        }
        actual = run_shape_inference(
            "",
            "SoftmaxCrossEntropyLoss",
            [ts(FLOAT, [2, 3]), ts(INT64, [2])],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_default_reduction_is_mean(self):
        actual = run_shape_inference(
            "",
            "SoftmaxCrossEntropyLoss",
            [ts(FLOAT, [2, 3]), ts(INT64, [2])],
            opset_version=13,
        )
        self.assertEqual(actual, [ts(FLOAT, [])])

    def test_log_prob_output(self):
        actual = run_shape_inference(
            "",
            "SoftmaxCrossEntropyLoss",
            [ts(FLOAT, [2, 3]), ts(INT64, [2])],
            opset_version=13,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, []))
        self.assertEqual(actual[1], ts(FLOAT, [2, 3]))

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "SoftmaxCrossEntropyLoss", [], opset_version=13)

    def test_none_input(self):
        v = ir.Value(name="scores", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 3]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "SoftmaxCrossEntropyLoss",
                [v, None],
                opset_version=13,
            )


class NLLLossSymbolicDimsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("none", "none", ["N"]),
            ("mean", "mean", []),
        ]
    )
    def test_symbolic_batch(self, _name, reduction, expected_shape):
        attrs = {
            "reduction": ir.Attr("reduction", ir.AttributeType.STRING, reduction),
        }
        actual = run_shape_inference(
            "",
            "NegativeLogLikelihoodLoss",
            [ts(FLOAT, ["N", "C"]), ts(INT64, ["N"])],
            attrs,
            opset_version=13,
        )
        self.assertEqual(actual[0].shape, ir.Shape(expected_shape))


if __name__ == "__main__":
    unittest.main()
