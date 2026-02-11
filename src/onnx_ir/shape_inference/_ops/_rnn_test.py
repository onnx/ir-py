# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RNN/GRU/LSTM shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class RNNTest(unittest.TestCase):
    def test_basic_forward(self):
        attrs = {
            "hidden_size": ir.Attr("hidden_size", ir.AttributeType.INT, 16),
        }
        actual = run_shape_inference(
            "",
            "RNN",
            [ts(FLOAT, [10, 2, 8])],
            attrs,
            opset_version=21,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [10, 1, 2, 16]))
        self.assertEqual(actual[1], ts(FLOAT, [1, 2, 16]))

    def test_bidirectional(self):
        attrs = {
            "hidden_size": ir.Attr("hidden_size", ir.AttributeType.INT, 16),
            "direction": ir.Attr("direction", ir.AttributeType.STRING, "bidirectional"),
        }
        actual = run_shape_inference(
            "",
            "RNN",
            [ts(FLOAT, [10, 2, 8])],
            attrs,
            opset_version=21,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [10, 2, 2, 16]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 2, 16]))

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "RNN", [None], opset_version=21)


class LSTMTest(unittest.TestCase):
    def test_basic_forward(self):
        attrs = {
            "hidden_size": ir.Attr("hidden_size", ir.AttributeType.INT, 16),
        }
        actual = run_shape_inference(
            "",
            "LSTM",
            [ts(FLOAT, [10, 2, 8])],
            attrs,
            opset_version=21,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [10, 1, 2, 16]))
        self.assertEqual(actual[1], ts(FLOAT, [1, 2, 16]))
        self.assertEqual(actual[2], ts(FLOAT, [1, 2, 16]))

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "LSTM", [None], opset_version=21)


class GRUTest(unittest.TestCase):
    def test_basic_forward(self):
        attrs = {
            "hidden_size": ir.Attr("hidden_size", ir.AttributeType.INT, 16),
        }
        actual = run_shape_inference(
            "",
            "GRU",
            [ts(FLOAT, [10, 2, 8])],
            attrs,
            opset_version=21,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [10, 1, 2, 16]))
        self.assertEqual(actual[1], ts(FLOAT, [1, 2, 16]))

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "GRU", [None], opset_version=21)


if __name__ == "__main__":
    unittest.main()
