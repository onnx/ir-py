# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ShapeMergePolicy in ShapeInferenceContext."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference._context import ShapeInferenceContext


class ShapeMergePolicyTest(unittest.TestCase):
    """Tests for ShapeMergePolicy in context."""

    def _create_model_with_value(self, existing_shape=None, existing_dtype=None):
        """Helper to create a simple model with a value."""
        value = ir.Value(name="test", shape=existing_shape, type=None)
        if existing_dtype:
            value.dtype = existing_dtype
        graph = ir.Graph(inputs=[value], outputs=[value], nodes=[], opset_imports={"": 17})
        return ir.Model(graph, ir_version=8), value

    def test_skip_policy_keeps_existing(self):
        model, value = self._create_model_with_value(ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="skip")

        modified = ctx.set_shape(value, ir.Shape([4, 5, 6]))
        self.assertFalse(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_override_policy_replaces(self):
        model, value = self._create_model_with_value(ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="override")

        modified = ctx.set_shape(value, ir.Shape([4, 5, 6]))
        self.assertTrue(modified)
        self.assertEqual(value.shape, [4, 5, 6])

    def test_refine_policy_updates_unknown_to_known(self):
        model, value = self._create_model_with_value(ir.Shape([None, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="refine")

        modified = ctx.set_shape(value, ir.Shape([1, 2, 3]))
        self.assertTrue(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_refine_policy_keeps_concrete(self):
        model, value = self._create_model_with_value(ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="refine")

        # Try to refine with symbolic - should keep concrete
        modified = ctx.set_shape(value, ir.Shape(["batch", 2, 3]))
        self.assertFalse(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_strict_policy_raises_on_conflict(self):
        model, value = self._create_model_with_value(ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="strict")

        with self.assertRaises(ValueError) as cm:
            ctx.set_shape(value, ir.Shape([4, 2, 3]))
        self.assertIn("conflict", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
