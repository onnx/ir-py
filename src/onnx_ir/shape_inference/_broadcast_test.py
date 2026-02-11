# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for broadcast_shapes utility."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import broadcast_shapes


class BroadcastShapesTest(unittest.TestCase):
    """Tests for broadcast_shapes utility."""

    def test_same_shape(self):
        s1 = ir.Shape([3, 4, 5])
        s2 = ir.Shape([3, 4, 5])
        result = broadcast_shapes(s1, s2)
        self.assertEqual(result, [3, 4, 5])

    def test_broadcast_with_ones(self):
        s1 = ir.Shape([3, 1, 5])
        s2 = ir.Shape([1, 4, 5])
        result = broadcast_shapes(s1, s2)
        self.assertEqual(result, [3, 4, 5])

    def test_broadcast_different_ranks(self):
        s1 = ir.Shape([4, 5])
        s2 = ir.Shape([3, 4, 5])
        result = broadcast_shapes(s1, s2)
        self.assertEqual(result, [3, 4, 5])

    def test_broadcast_symbolic(self):
        s1 = ir.Shape(["batch", 1, 256])
        s2 = ir.Shape([1, "seq_len", 256])
        result = broadcast_shapes(s1, s2)
        self.assertEqual(str(result), "[batch,seq_len,256]")

    def test_broadcast_none_shape(self):
        s1 = ir.Shape([3, 4])
        result = broadcast_shapes(s1, None)
        self.assertIsNone(result)

    def test_broadcast_incompatible(self):
        s1 = ir.Shape([3, 4])
        s2 = ir.Shape([5, 4])
        result = broadcast_shapes(s1, s2)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
