# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Slice shape inference."""

from __future__ import annotations

import unittest

import parameterized

import onnx_ir as ir
from onnx_ir.shape_inference._ops import _testing

FLOAT = ir.DataType.FLOAT


class SliceTest(unittest.TestCase):
    def _run(self, input_ts, starts, ends, axes=None, steps=None):
        data = ir.Value(name="data", shape=input_ts.shape, type=input_ts.type)
        inputs = [data, _testing.const_value(starts, "starts"), _testing.const_value(ends, "ends")]
        if axes is not None:
            inputs.append(_testing.const_value(axes, "axes"))
        if steps is not None:
            if axes is None:
                inputs.append(ir.Value(name="axes_empty"))
            inputs.append(_testing.const_value(steps, "steps"))
        return _testing.run_shape_inference_with_values(
            "", "Slice", inputs, opset_version=17,
        )

    @parameterized.parameterized.expand([
        ("basic", [10, 20], [1], [5], [0], [1], [4, 20]),
        ("negative_end", [10, 20], [0], [-1], [0], [1], [9, 20]),
        ("step_2", [10, 20], [0], [10], [0], [2], [5, 20]),
        ("axis_1", [10, 20], [2], [8], [1], [1], [10, 6]),
        ("multi_axis", [10, 20, 30], [1, 2], [5, 10], [0, 1], [1, 1], [4, 8, 30]),
    ])
    def test_slice(self, _name, shape, starts, ends, axes, steps, expected_shape):
        actual = self._run(_testing.ts(FLOAT, shape), starts, ends, axes, steps)
        self.assertEqual(actual, [_testing.ts(FLOAT, expected_shape)])

    def test_missing_input_shape(self):
        data = ir.Value(name="data", type=ir.TensorType(FLOAT))
        starts = _testing.const_value([0], "starts")
        ends = _testing.const_value([5], "ends")
        actual = _testing.run_shape_inference_with_values(
            "", "Slice", [data, starts, ends], opset_version=17,
        )
        self.assertIsNone(actual[0].shape)


if __name__ == "__main__":
    unittest.main()
