# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for symbolic shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common.symbolic_shape_inference import SymbolicShapeInferencePass
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


class OpShapeInferenceRegistryTest(unittest.TestCase):
    """Tests for OpShapeInferenceRegistry."""

    def setUp(self):
        # Use a fresh registry for each test
        from onnx_ir.shape_inference._registry import OpShapeInferenceRegistry

        self.registry = OpShapeInferenceRegistry()

    def test_register_with_since_version(self):
        @self.registry.register("", "TestOp", since_version=7)
        def infer_test(ctx, node):
            pass

        # Should work for version 7 and above
        self.assertIsNotNone(self.registry.get("", "TestOp", version=7))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=10))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=20))
        # Should not work below version 7
        self.assertIsNone(self.registry.get("", "TestOp", version=6))

    def test_register_default_since_version(self):
        @self.registry.register("", "TestOp")
        def infer_test(ctx, node):
            pass

        # Default since_version=1, so should work for version 1 and above
        self.assertIsNotNone(self.registry.get("", "TestOp", version=1))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=100))
        # Should not work below version 1
        self.assertIsNone(self.registry.get("", "TestOp", version=0))

    def test_has(self):
        @self.registry.register("", "TestOp", since_version=1)
        def infer_test(ctx, node):
            pass

        self.assertTrue(self.registry.has("", "TestOp"))
        self.assertFalse(self.registry.has("", "NonExistent"))

    def test_multiple_version_registrations(self):
        @self.registry.register("", "TestOp", since_version=7)
        def infer_v7(ctx, node):
            return "v7"

        @self.registry.register("", "TestOp", since_version=14)
        def infer_v14(ctx, node):
            return "v14"

        # Version 6 should return None (below all registrations)
        self.assertIsNone(self.registry.get("", "TestOp", version=6))

        # Version 7-13 should get v7 handler
        func7 = self.registry.get("", "TestOp", version=7)
        self.assertEqual(func7(None, None), "v7")

        func10 = self.registry.get("", "TestOp", version=10)
        self.assertEqual(func10(None, None), "v7")

        func13 = self.registry.get("", "TestOp", version=13)
        self.assertEqual(func13(None, None), "v7")

        # Version 14 and above should get v14 handler
        func14 = self.registry.get("", "TestOp", version=14)
        self.assertEqual(func14(None, None), "v14")

        func20 = self.registry.get("", "TestOp", version=20)
        self.assertEqual(func20(None, None), "v14")

    def test_lookup_is_o1_after_first_access(self):
        """Test that lookup uses cached dict for O(1) access."""

        @self.registry.register("", "TestOp", since_version=7)
        def infer_v7(ctx, node):
            return "v7"

        @self.registry.register("", "TestOp", since_version=14)
        def infer_v14(ctx, node):
            return "v14"

        # First access builds the cache
        self.registry.get("", "TestOp", version=10)

        # Verify cache was built
        key = ("", "TestOp")
        self.assertIn(key, self.registry._cache)
        self.assertIn(key, self.registry._max_version)

        # Cache should have versions 7-13 mapped to v7
        cache = self.registry._cache[key]
        for v in range(7, 14):
            self.assertIn(v, cache)

        # Max version should be (14, infer_v14)
        max_since, max_func = self.registry._max_version[key]
        self.assertEqual(max_since, 14)
        self.assertEqual(max_func(None, None), "v14")

    def test_cache_invalidation_on_new_registration(self):
        """Test that cache is invalidated when new registration is added."""

        @self.registry.register("", "TestOp", since_version=7)
        def infer_v7(ctx, node):
            return "v7"

        # Build cache
        self.registry.get("", "TestOp", version=10)

        key = ("", "TestOp")
        self.assertIn(key, self.registry._cache)

        # Add new registration
        @self.registry.register("", "TestOp", since_version=14)
        def infer_v14(ctx, node):
            return "v14"

        # Cache should be invalidated
        self.assertNotIn(key, self.registry._cache)

        # New lookup should work correctly
        func20 = self.registry.get("", "TestOp", version=20)
        self.assertEqual(func20(None, None), "v14")


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
        from onnx_ir.shape_inference._context import ShapeInferenceContext

        model, value = self._create_model_with_value(ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="skip")

        modified = ctx.set_shape(value, ir.Shape([4, 5, 6]))
        self.assertFalse(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_override_policy_replaces(self):
        from onnx_ir.shape_inference._context import ShapeInferenceContext

        model, value = self._create_model_with_value(ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="override")

        modified = ctx.set_shape(value, ir.Shape([4, 5, 6]))
        self.assertTrue(modified)
        self.assertEqual(value.shape, [4, 5, 6])

    def test_refine_policy_updates_unknown_to_known(self):
        from onnx_ir.shape_inference._context import ShapeInferenceContext

        model, value = self._create_model_with_value(ir.Shape([None, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="refine")

        modified = ctx.set_shape(value, ir.Shape([1, 2, 3]))
        self.assertTrue(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_refine_policy_keeps_concrete(self):
        from onnx_ir.shape_inference._context import ShapeInferenceContext

        model, value = self._create_model_with_value(ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="refine")

        # Try to refine with symbolic - should keep concrete
        modified = ctx.set_shape(value, ir.Shape(["batch", 2, 3]))
        self.assertFalse(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_strict_policy_raises_on_conflict(self):
        from onnx_ir.shape_inference._context import ShapeInferenceContext

        model, value = self._create_model_with_value(ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(model, policy="strict")

        with self.assertRaises(ValueError) as cm:
            ctx.set_shape(value, ir.Shape([4, 2, 3]))
        self.assertIn("conflict", str(cm.exception).lower())


class SymbolicShapeInferencePassTest(unittest.TestCase):
    """Tests for SymbolicShapeInferencePass."""

    def test_add_shape_inference(self):
        x = ir.Value(
            name="x", shape=ir.Shape(["batch", 128]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        y = ir.Value(name="y", shape=ir.Shape([1, 128]), type=ir.TensorType(ir.DataType.FLOAT))
        add_out = ir.Value(name="add_out")
        add_node = ir.Node("", "Add", inputs=[x, y], outputs=[add_out])

        graph = ir.Graph(
            inputs=[x, y], outputs=[add_out], nodes=[add_node], opset_imports={"": 17}
        )
        model = ir.Model(graph, ir_version=8)

        result = SymbolicShapeInferencePass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(str(add_out.shape), "[batch,128]")
        self.assertEqual(add_out.dtype, ir.DataType.FLOAT)

    def test_transpose_shape_inference(self):
        x = ir.Value(
            name="x",
            shape=ir.Shape(["batch", "seq", 256]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        trans_out = ir.Value(name="trans_out")
        trans_node = ir.Node(
            "",
            "Transpose",
            inputs=[x],
            outputs=[trans_out],
            attributes={"perm": ir.Attr("perm", ir.AttributeType.INTS, [2, 0, 1])},
        )

        graph = ir.Graph(
            inputs=[x], outputs=[trans_out], nodes=[trans_node], opset_imports={"": 17}
        )
        model = ir.Model(graph, ir_version=8)

        result = SymbolicShapeInferencePass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(str(trans_out.shape), "[256,batch,seq]")
        self.assertEqual(trans_out.dtype, ir.DataType.FLOAT)

    def test_transpose_default_perm(self):
        x = ir.Value(
            name="x", shape=ir.Shape([2, 3, 4]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        trans_out = ir.Value(name="trans_out")
        trans_node = ir.Node("", "Transpose", inputs=[x], outputs=[trans_out])

        graph = ir.Graph(
            inputs=[x], outputs=[trans_out], nodes=[trans_node], opset_imports={"": 17}
        )
        model = ir.Model(graph, ir_version=8)

        result = SymbolicShapeInferencePass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(trans_out.shape, [4, 3, 2])

    def test_chained_ops(self):
        x = ir.Value(
            name="x", shape=ir.Shape(["batch", 128]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        y = ir.Value(name="y", shape=ir.Shape([1, 128]), type=ir.TensorType(ir.DataType.FLOAT))
        add_out = ir.Value(name="add_out")
        add_node = ir.Node("", "Add", inputs=[x, y], outputs=[add_out])

        trans_out = ir.Value(name="trans_out")
        trans_node = ir.Node(
            "",
            "Transpose",
            inputs=[add_out],
            outputs=[trans_out],
            attributes={"perm": ir.Attr("perm", ir.AttributeType.INTS, [1, 0])},
        )

        graph = ir.Graph(
            inputs=[x, y],
            outputs=[trans_out],
            nodes=[add_node, trans_node],
            opset_imports={"": 17},
        )
        model = ir.Model(graph, ir_version=8)

        result = SymbolicShapeInferencePass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(str(add_out.shape), "[batch,128]")
        self.assertEqual(str(trans_out.shape), "[128,batch]")


if __name__ == "__main__":
    unittest.main()
