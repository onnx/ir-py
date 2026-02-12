# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Slice shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference import OpUsageError
from onnx_ir.shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class SliceTest(unittest.TestCase):
    def _run(self, input_ts, starts, ends, axes=None, steps=None):
        data = ir.Value(name="data", shape=input_ts.shape, type=input_ts.type)
        inputs = [
            data,
            const_value(starts, "starts"),
            const_value(ends, "ends"),
        ]
        if axes is not None:
            inputs.append(const_value(axes, "axes"))
        if steps is not None:
            if axes is None:
                inputs.append(ir.Value(name="axes_empty"))
            inputs.append(const_value(steps, "steps"))
        return run_shape_inference_with_values(
            "",
            "Slice",
            inputs,
            opset_version=17,
        )

    @parameterized.parameterized.expand(
        [
            ("basic", [10, 20], [1], [5], [0], [1], [4, 20]),
            ("negative_end", [10, 20], [0], [-1], [0], [1], [9, 20]),
            ("step_2", [10, 20], [0], [10], [0], [2], [5, 20]),
            ("axis_1", [10, 20], [2], [8], [1], [1], [10, 6]),
            ("multi_axis", [10, 20, 30], [1, 2], [5, 10], [0, 1], [1, 1], [4, 8, 30]),
            # From ONNX test_slice_giant_number: large end value clipped
            ("giant_end", [3, 2], [0, 0], [3, 2147483647], [0, 1], [1, 1], [3, 2]),
            # From ONNX test_slice_giant_step
            ("giant_step", [3, 2], [0, 0], [3, 2], [0, 1], [1, 2147483647], [3, 1]),
            # From ONNX test_slice_negative_start
            ("negative_start", [3, 2], [-2, 0], [3, 2], [0, 1], [1, 1], [2, 2]),
        ]
    )
    def test_slice(self, _name, shape, starts, ends, axes, steps, expected_shape):
        actual = self._run(ts(FLOAT, shape), starts, ends, axes, steps)
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_negative_step(self):
        # From ONNX test_slice_negative_step: backward slicing
        actual = self._run(ts(FLOAT, [3, 4]), [2, 3], [0, 1], [0, 1], [-1, -1])
        self.assertEqual(actual, [ts(FLOAT, [2, 2])])

    def test_no_axes_defaults_to_range(self):
        """When axes is not provided, defaults to [0, 1, ..., len(starts)-1]."""
        actual = self._run(ts(FLOAT, [3, 2]), [1, 0], [2, 2])
        self.assertEqual(actual, [ts(FLOAT, [1, 2])])

    def test_symbolic_dim_becomes_unknown(self):
        """Symbolic dims that are sliced become unknown."""
        actual = self._run(ts(FLOAT, ["a", 2]), [0], [1], [1], [1])
        result = actual[0]
        # Dim 0 is symbolic and untouched
        self.assertNotIsInstance(result.shape[0], int)
        # Dim 1 is concrete sliced
        self.assertEqual(result.shape[1], 1)

    def test_symbolic_input_const_slice(self):
        """Slice on ["N", "C"] with const starts/ends on axis 1 → ["N", concrete]."""
        actual = self._run(ts(FLOAT, ["N", 10]), [2], [8], [1], [1])
        result = actual[0]
        self.assertEqual(result.shape[0], ir.SymbolicDim("N"))
        self.assertEqual(result.shape[1], 6)
        self.assertEqual(result.type.dtype, FLOAT)

    def test_missing_input_shape(self):
        data = ir.Value(name="data", type=ir.TensorType(FLOAT))
        starts = const_value([0], "starts")
        ends = const_value([5], "ends")
        actual = run_shape_inference_with_values(
            "",
            "Slice",
            [data, starts, ends],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_dynamic_starts_ends(self):
        """When starts/ends are not const, output shape has same rank with symbolic dims."""
        data = ir.Value(name="data", shape=ir.Shape([10, 20]), type=ir.TensorType(FLOAT))
        starts = ir.Value(name="starts", type=ir.TensorType(ir.DataType.INT64))
        ends = ir.Value(name="ends", type=ir.TensorType(ir.DataType.INT64))
        actual = run_shape_inference_with_values(
            "",
            "Slice",
            [data, starts, ends],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertEqual(actual[0].type.dtype, FLOAT)

    def test_slice_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Slice", [ts(FLOAT, [5])], opset_version=17)

    def test_slice_none_data(self):
        starts = const_value([0])
        ends = const_value([3])
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Slice",
                [None, starts, ends],
                opset_version=17,
            )

    def test_slice_non_const_starts(self):
        """Non-constant starts/ends → output shape has same rank with symbolic dims."""
        data = ir.Value(name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([10, 20]))
        starts = ir.Value(
            name="starts", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([1])
        )
        ends = ir.Value(
            name="ends", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([1])
        )
        actual = run_shape_inference_with_values(
            "",
            "Slice",
            [data, starts, ends],
            opset_version=17,
        )
        self.assertIsNotNone(actual[0].shape)
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertEqual(actual[0].type.dtype, FLOAT)


if __name__ == "__main__":
    unittest.main()
