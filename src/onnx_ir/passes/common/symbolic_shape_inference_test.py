# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for symbolic shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common.symbolic_shape_inference import SymbolicShapeInferencePass
from onnx_ir.shape_inference import broadcast_shapes


class SymbolicDimTest(unittest.TestCase):
    """Tests for SymbolicDim class."""

    def test_string_value(self):
        dim = ir.SymbolicDim("batch")
        self.assertEqual(dim.value, "batch")
        self.assertIsNone(dim._expr)  # Lazy - not created yet

    def test_none_value(self):
        dim = ir.SymbolicDim(None)
        self.assertIsNone(dim.value)
        self.assertIsNone(dim.expr)

    def test_sympy_expr_value(self):
        import sympy

        expr = sympy.Symbol("N") + 1
        dim = ir.SymbolicDim(expr)
        self.assertEqual(dim.value, "N + 1")
        self.assertEqual(dim.expr, expr)

    def test_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            ir.SymbolicDim(42)

    def test_equality_with_string(self):
        dim = ir.SymbolicDim("N")
        self.assertEqual(dim, "N")
        self.assertNotEqual(dim, "M")

    def test_equality_with_none(self):
        dim_none = ir.SymbolicDim(None)
        dim_named = ir.SymbolicDim("N")
        self.assertEqual(dim_none, None)
        self.assertNotEqual(dim_named, None)

    def test_equality_with_symbolic_dim(self):
        dim1 = ir.SymbolicDim("N")
        dim2 = ir.SymbolicDim("N")
        dim3 = ir.SymbolicDim("M")
        self.assertEqual(dim1, dim2)
        self.assertNotEqual(dim1, dim3)

    def test_hash_consistency(self):
        dim1 = ir.SymbolicDim("batch")
        dim2 = ir.SymbolicDim("batch")
        self.assertEqual(hash(dim1), hash(dim2))

    def test_add_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim + 1
        self.assertEqual(result.value, "N + 1")

    def test_add_with_symbolic_dim(self):
        dim1 = ir.SymbolicDim("N")
        dim2 = ir.SymbolicDim("M")
        result = dim1 + dim2
        self.assertEqual(result.value, "M + N")

    def test_add_with_none_returns_none(self):
        dim = ir.SymbolicDim(None)
        result = dim + 1
        self.assertIsNone(result.value)

    def test_sub_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim - 1
        self.assertEqual(result.value, "N - 1")

    def test_mul_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim * 2
        self.assertEqual(result.value, "2*N")

    def test_floordiv_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim // 2
        self.assertEqual(result.value, "floor(N/2)")

    def test_mod_with_int(self):
        dim = ir.SymbolicDim("N")
        result = dim % 2
        self.assertEqual(result.value, "Mod(N, 2)")

    def test_radd(self):
        dim = ir.SymbolicDim("N")
        result = 1 + dim
        self.assertEqual(result.value, "N + 1")

    def test_rsub(self):
        dim = ir.SymbolicDim("N")
        result = 10 - dim
        self.assertEqual(result.value, "10 - N")

    def test_rmul(self):
        dim = ir.SymbolicDim("N")
        result = 2 * dim
        self.assertEqual(result.value, "2*N")

    def test_unsupported_operand_raises_type_error(self):
        dim = ir.SymbolicDim("N")
        with self.assertRaises(TypeError) as ctx:
            _ = dim + "string"
        self.assertIn("unsupported operand type", str(ctx.exception))

    def test_simplify(self):
        dim = ir.SymbolicDim("N") + 0
        simplified = dim.simplify()
        self.assertEqual(simplified.value, "N")

    def test_evaluate(self):
        dim = ir.SymbolicDim("N") * 2 + 1
        result = dim.evaluate({"N": 5})
        self.assertEqual(result, 11)

    def test_evaluate_none_returns_none(self):
        dim = ir.SymbolicDim(None)
        result = dim.evaluate({"N": 5})
        self.assertIsNone(result)

    def test_evaluate_incomplete_bindings_returns_none(self):
        dim = ir.SymbolicDim("N") + ir.SymbolicDim("M")
        result = dim.evaluate({"N": 5})  # M not provided
        self.assertIsNone(result)

    def test_free_symbols(self):
        dim = ir.SymbolicDim("N") + ir.SymbolicDim("M")
        symbols = dim.free_symbols()
        self.assertEqual(symbols, frozenset({"N", "M"}))

    def test_free_symbols_none(self):
        dim = ir.SymbolicDim(None)
        symbols = dim.free_symbols()
        self.assertEqual(symbols, frozenset())


class ShapeEvaluateTest(unittest.TestCase):
    """Tests for Shape.evaluate() and Shape.simplify()."""

    def test_evaluate_static_shape(self):
        shape = ir.Shape([1, 2, 3])
        result = shape.evaluate({})
        self.assertEqual(result, (1, 2, 3))

    def test_evaluate_symbolic_shape(self):
        shape = ir.Shape(["batch", 256, ir.SymbolicDim("seq") + 1])
        result = shape.evaluate({"batch": 32, "seq": 128})
        self.assertEqual(result, (32, 256, 129))

    def test_evaluate_incomplete_returns_none(self):
        shape = ir.Shape(["batch", "seq"])
        result = shape.evaluate({"batch": 32})  # seq not provided
        self.assertIsNone(result)

    def test_simplify(self):
        shape = ir.Shape([ir.SymbolicDim("N") + 0, ir.SymbolicDim("M") * 1])
        simplified = shape.simplify()
        self.assertEqual(simplified[0].value, "N")
        self.assertEqual(simplified[1].value, "M")

    def test_free_symbols(self):
        shape = ir.Shape(["batch", 256, "seq_len"])
        symbols = shape.free_symbols()
        self.assertEqual(symbols, frozenset({"batch", "seq_len"}))


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

    def test_register_with_int_version(self):
        @self.registry.register("", "TestOp", versions=7)
        def infer_test(ctx, node):
            pass

        # Should work for version 7 and above
        self.assertIsNotNone(self.registry.get("", "TestOp", version=7))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=10))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=20))
        # Should not work below version 7
        self.assertIsNone(self.registry.get("", "TestOp", version=6))

    def test_register_with_range(self):
        @self.registry.register("", "TestOp", versions=range(7, 14))
        def infer_test(ctx, node):
            pass

        self.assertIsNotNone(self.registry.get("", "TestOp", version=7))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=13))
        self.assertIsNone(self.registry.get("", "TestOp", version=6))
        self.assertIsNone(self.registry.get("", "TestOp", version=14))

    def test_register_with_none_versions(self):
        @self.registry.register("", "TestOp", versions=None)
        def infer_test(ctx, node):
            pass

        # Should work for any version
        self.assertIsNotNone(self.registry.get("", "TestOp", version=1))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=100))

    def test_has(self):
        @self.registry.register("", "TestOp", versions=1)
        def infer_test(ctx, node):
            pass

        self.assertTrue(self.registry.has("", "TestOp"))
        self.assertFalse(self.registry.has("", "NonExistent"))

    def test_multiple_version_registrations(self):
        @self.registry.register("", "TestOp", versions=range(7, 14))
        def infer_v7(ctx, node):
            return "v7"

        @self.registry.register("", "TestOp", versions=14)
        def infer_v14(ctx, node):
            return "v14"

        # Version 10 should get v7 handler
        func10 = self.registry.get("", "TestOp", version=10)
        self.assertEqual(func10(None, None), "v7")

        # Version 14 and above should get v14 handler
        func14 = self.registry.get("", "TestOp", version=14)
        self.assertEqual(func14(None, None), "v14")

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
