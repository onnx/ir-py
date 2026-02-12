# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Reshape shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class ReshapeTest(unittest.TestCase):
    def _run(self, input_ts, shape_data, expected):
        data = ir.Value(name="data", shape=input_ts.shape, type=input_ts.type)
        shape_val = const_value(shape_data, name="shape")
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertEqual(actual, expected)

    @parameterized.parameterized.expand(
        [
            ("simple", [2, 3, 4], [6, 4], [ts(FLOAT, [6, 4])]),
            ("with_neg_one", [2, 3, 4], [2, -1], [ts(FLOAT, [2, 12])]),
            ("with_zero", [2, 3, 4], [0, -1], [ts(FLOAT, [2, 12])]),
            ("flatten_all", [2, 3, 4], [-1], [ts(FLOAT, [24])]),
            ("add_dim", [6], [2, 3], [ts(FLOAT, [2, 3])]),
            # From ONNX test_reshape_static_shape_zero: 0 means copy from input
            ("zero_keeps_dim", [2, 3, 4], [0, 0, 4], [ts(FLOAT, [2, 3, 4])]),
        ]
    )
    def test_reshape(self, _name, input_shape, target_shape, expected):
        self._run(ts(FLOAT, input_shape), target_shape, expected)

    def test_dynamic_shape(self):
        """When shape input is not const, output rank can still be inferred."""
        data = ir.Value(
            name="data",
            shape=ir.Shape([2, 3]),
            type=ir.TensorType(FLOAT),
        )
        shape_val = ir.Value(
            name="shape",
            shape=ir.Shape([3]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)

    def test_missing_data_shape(self):
        """When data shape is unknown, can still infer rank from shape input."""
        data = ir.Value(name="data", type=ir.TensorType(FLOAT))
        shape_val = const_value([3, 4], name="shape")
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        # Can infer target shape from const shape input
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_allowzero(self):
        """allowzero=1: 0 in target means literal zero-size dim."""
        data = ir.Value(name="data", shape=ir.Shape([2, 0, 4]), type=ir.TensorType(FLOAT))
        shape_val = const_value([0, 4])
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            attributes={"allowzero": ir.Attr("allowzero", ir.AttributeType.INT, 1)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [0, 4])])

    def test_neg_one_non_static_input(self):
        """When input shape is symbolic, -1 becomes a new symbolic dim."""
        data = ir.Value(
            name="data",
            shape=ir.Shape([ir.SymbolicDim("batch"), 12]),
            type=ir.TensorType(FLOAT),
        )
        shape_val = const_value([-1, 3, 4])
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 3)
        self.assertEqual(actual[0].shape[1], 3)
        # -1 dim should be symbolic, not literal -1
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)

    def test_dynamic_shape_with_known_rank(self):
        """Dynamic shape input (non-const) but shape input's shape gives output rank."""
        data = ir.Value(
            name="data",
            shape=ir.Shape([2, 3, 4]),
            type=ir.TensorType(FLOAT),
        )
        shape_val = ir.Value(
            name="shape",
            shape=ir.Shape([2]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertIsInstance(actual[0].shape[1], ir.SymbolicDim)

    def test_neg_one_with_symbolic_known_dims(self):
        """Reshape with -1 where known dims don't fully resolve (symbolic in output)."""
        data = ir.Value(
            name="data",
            shape=ir.Shape([ir.SymbolicDim("N"), 3, 4]),
            type=ir.TensorType(FLOAT),
        )
        shape_val = const_value([-1, 12])
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 12)

    def test_zero_dim_no_input_shape(self):
        """0-dim in target with no input shape → symbolic dim."""
        data = ir.Value(name="data", type=ir.TensorType(FLOAT))
        shape_val = const_value([0, 3])
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 3)

    def test_double_neg_one_error(self):
        """Two -1 dims should record an error."""
        from onnx_ir.shape_inference import ShapeInferenceError

        data = ir.Value(name="data", shape=ir.Shape([2, 3, 4]), type=ir.TensorType(FLOAT))
        shape_val = const_value([-1, -1])
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference_with_values(
                "",
                "Reshape",
                [data, shape_val],
                opset_version=17,
            )

    def test_neg_one_with_zero_dim_static(self):
        """Reshape [-1, 0] with static input [2, 3] and allowzero=0 → -1 resolves."""
        data = ir.Value(name="data", shape=ir.Shape([2, 3]), type=ir.TensorType(FLOAT))
        shape_val = const_value([-1, 0])
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3])])

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Reshape",
                [],
                opset_version=17,
            )

    def test_dynamic_shape_no_rank(self):
        """Shape input with unknown rank → dtype only."""
        data = ir.Value(name="data", shape=ir.Shape([2, 3]), type=ir.TensorType(FLOAT))
        shape_val = ir.Value(name="shape", type=ir.TensorType(ir.DataType.INT64))
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT)])

    def test_zero_copies_from_input_missing_dim(self):
        """0-dim in target copies from input; when input has the dim, it's used."""
        data = ir.Value(name="data", shape=ir.Shape([6]), type=ir.TensorType(FLOAT))
        shape_val = const_value([0, 2, 3])
        actual = run_shape_inference_with_values(
            "",
            "Reshape",
            [data, shape_val],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [6, 2, 3])])


if __name__ == "__main__":
    unittest.main()
