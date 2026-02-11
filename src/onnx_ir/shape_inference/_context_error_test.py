# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ShapeInferenceContext error recording."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import _context


class ErrorRecordingTest(unittest.TestCase):
    """Tests for record_error and errors property."""

    def _make_node(self, op_type: str = "TestOp", name: str | None = "node0") -> ir.Node:
        return ir.Node("", op_type, inputs=[], num_outputs=1, name=name)

    def test_record_error_stores_error(self):
        ctx = _context.ShapeInferenceContext(policy="refine")
        node = self._make_node()
        ctx.record_error(node, "something is wrong")
        self.assertEqual(len(ctx.errors), 1)
        self.assertEqual(ctx.errors[0].message, "something is wrong")
        self.assertEqual(ctx.errors[0].op_type, "TestOp")
        self.assertEqual(ctx.errors[0].node_name, "node0")

    def test_record_error_strict_raises(self):
        ctx = _context.ShapeInferenceContext(policy="strict")
        node = self._make_node()
        with self.assertRaises(ValueError):
            ctx.record_error(node, "bad input")

    def test_record_error_non_strict_does_not_raise(self):
        for policy in ("refine", "override", "skip"):
            ctx = _context.ShapeInferenceContext(policy=policy)
            node = self._make_node()
            ctx.record_error(node, "warning")
            self.assertEqual(len(ctx.errors), 1)

    def test_multiple_errors(self):
        ctx = _context.ShapeInferenceContext(policy="refine")
        ctx.record_error(self._make_node(name="a"), "err1")
        ctx.record_error(self._make_node(name="b"), "err2")
        self.assertEqual(len(ctx.errors), 2)

    def test_error_str(self):
        err = _context.ShapeInferenceError(
            node_name="n", op_type="Add", domain="", message="bad"
        )
        self.assertIn("Add", str(err))
        self.assertIn("bad", str(err))


if __name__ == "__main__":
    unittest.main()
