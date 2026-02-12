# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RNN/GRU/LSTM shape inference."""

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


class RNNBasicTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("rnn", "RNN", 2, [10, 1, 2, 16], [[1, 2, 16]]),
            ("gru", "GRU", 2, [10, 1, 2, 16], [[1, 2, 16]]),
            ("lstm", "LSTM", 3, [10, 1, 2, 16], [[1, 2, 16], [1, 2, 16]]),
        ]
    )
    def test_basic_forward(self, _name, op_type, num_outputs, y_shape, hidden_shapes):
        attrs = {
            "hidden_size": ir.Attr("hidden_size", ir.AttributeType.INT, 16),
        }
        actual = run_shape_inference(
            "",
            op_type,
            [ts(FLOAT, [10, 2, 8])],
            attrs,
            opset_version=21,
            num_outputs=num_outputs,
        )
        self.assertEqual(actual[0], ts(FLOAT, y_shape))
        for i, hs in enumerate(hidden_shapes):
            self.assertEqual(actual[i + 1], ts(FLOAT, hs))

    @parameterized.parameterized.expand(
        [
            ("rnn", "RNN"),
            ("gru", "GRU"),
            ("lstm", "LSTM"),
        ]
    )
    def test_none_input_raises(self, _name, op_type):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", op_type, [None], opset_version=21)


class LSTMLayoutTest(unittest.TestCase):
    def test_lstm_layout_1(self):
        """LSTM with layout=1 (batch_first): X=[batch, seq, input_size]."""
        attrs = {
            "hidden_size": ir.Attr("hidden_size", ir.AttributeType.INT, 16),
            "layout": ir.Attr("layout", ir.AttributeType.INT, 1),
        }
        actual = run_shape_inference(
            "",
            "LSTM",
            [ts(FLOAT, [2, 10, 8])],
            attrs,
            opset_version=21,
            num_outputs=3,
        )
        # layout=1: Y=[batch, seq, num_directions, hidden_size]
        self.assertEqual(actual[0], ts(FLOAT, [2, 10, 1, 16]))
        # Y_h=[batch, num_directions, hidden_size]
        self.assertEqual(actual[1], ts(FLOAT, [2, 1, 16]))
        # Y_c=[batch, num_directions, hidden_size]
        self.assertEqual(actual[2], ts(FLOAT, [2, 1, 16]))


class RNNHiddenSizeFromWTest(unittest.TestCase):
    def test_rnn_hidden_size_from_w(self):
        """Infer hidden_size from W input when attribute is not provided."""
        # W shape: [num_directions, hidden_size * num_gates, input_size]
        # For RNN, num_gates=1, so W[1] = hidden_size
        x_val = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([10, 2, 8]))
        w_val = ir.Value(name="w", type=ir.TensorType(FLOAT), shape=ir.Shape([1, 16, 8]))
        actual = run_shape_inference_with_values(
            "",
            "RNN",
            [x_val, w_val],
            opset_version=21,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [10, 1, 2, 16]))
        self.assertEqual(actual[1], ts(FLOAT, [1, 2, 16]))

    def test_lstm_hidden_size_from_w(self):
        """Infer hidden_size from W for LSTM (num_gates=4)."""
        x_val = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([10, 2, 8]))
        # W[1] = hidden_size * 4 = 64 → hidden_size = 16
        w_val = ir.Value(name="w", type=ir.TensorType(FLOAT), shape=ir.Shape([1, 64, 8]))
        actual = run_shape_inference_with_values(
            "",
            "LSTM",
            [x_val, w_val],
            opset_version=21,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [10, 1, 2, 16]))


class RNNMissingShapeTest(unittest.TestCase):
    def test_rnn_missing_x_shape(self):
        """When x has no shape, outputs get dtype but no shape."""
        attrs = {
            "hidden_size": ir.Attr("hidden_size", ir.AttributeType.INT, 16),
        }
        actual = run_shape_inference(
            "",
            "RNN",
            [ts(FLOAT)],
            attrs,
            opset_version=21,
            num_outputs=2,
        )
        self.assertIsNone(actual[0].shape)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_rnn_no_hidden_size_symbolic(self):
        """When hidden_size is missing and W is not provided, use symbolic dim."""
        actual = run_shape_inference(
            "",
            "RNN",
            [ts(FLOAT, [10, 2, 8])],
            opset_version=21,
            num_outputs=2,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[3], ir.SymbolicDim)


class RNNSymbolicDimsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("lstm", "LSTM", 16, 3),
            ("gru", "GRU", 32, 2),
        ]
    )
    def test_symbolic_sequence_length(self, _name, op_type, hidden_size, num_outputs):
        attrs = {
            "hidden_size": ir.Attr("hidden_size", ir.AttributeType.INT, hidden_size),
        }
        actual = run_shape_inference(
            "",
            op_type,
            [ts(FLOAT, ["S", "B", 10])],
            attrs,
            opset_version=21,
            num_outputs=num_outputs,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 4)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[2], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 1)
        self.assertEqual(actual[0].shape[3], hidden_size)
        self.assertIsNotNone(actual[1].shape)
        self.assertEqual(actual[1].shape.rank(), 3)


if __name__ == "__main__":
    unittest.main()
