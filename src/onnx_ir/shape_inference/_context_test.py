# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ShapeMergePolicy in ShapeInferenceContext."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference._context import ShapeInferenceContext


class ShapeMergePolicyTest(unittest.TestCase):
    """Tests for ShapeMergePolicy in context."""

    def test_skip_policy_keeps_existing(self):
        value = ir.val("test", shape=ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(policy="skip")

        modified = ctx.set_shape(value, ir.Shape([4, 5, 6]))
        self.assertFalse(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_override_policy_replaces(self):
        value = ir.val("test", shape=ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(policy="override")

        modified = ctx.set_shape(value, ir.Shape([4, 5, 6]))
        self.assertTrue(modified)
        self.assertEqual(value.shape, [4, 5, 6])

    def test_refine_policy_updates_unknown_to_known(self):
        value = ir.val("test", shape=ir.Shape([None, 2, 3]))
        ctx = ShapeInferenceContext(policy="refine")

        modified = ctx.set_shape(value, ir.Shape([1, 2, 3]))
        self.assertTrue(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_refine_policy_keeps_concrete(self):
        value = ir.val("test", shape=ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(policy="refine")

        # Try to refine with symbolic - should keep concrete
        modified = ctx.set_shape(value, ir.Shape(["batch", 2, 3]))
        self.assertFalse(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_strict_policy_raises_on_conflict(self):
        value = ir.val("test", shape=ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(policy="strict")

        with self.assertRaises(ValueError) as cm:
            ctx.set_shape(value, ir.Shape([4, 2, 3]))
        self.assertIn("conflict", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
