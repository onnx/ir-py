# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for partial data propagation through shape tensors."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry
from onnx_ir.shape_inference._ops._testing import const_value, ts


def _make_value(
    name: str,
    spec: ir.TypeAndShape,
) -> ir.Value:
    """Create a Value from a TypeAndShape spec."""
    return ir.Value(name=name, shape=spec.shape, type=spec.type)


def _run_node(
    ctx: _context.ShapeInferenceContext,
    domain: str,
    op_type: str,
    inputs: list[ir.Value | None],
    attributes: dict[str, ir.Attr] | None = None,
    num_outputs: int = 1,
) -> list[ir.Value]:
    """Run shape inference on a single node and return the output values."""
    outputs = [ir.Value(name=f"out_{i}") for i in range(num_outputs)]
    node = ir.Node(
        domain, op_type, inputs=inputs, outputs=outputs, attributes=attributes or {}
    )
    for inp in inputs:
        if inp is not None:
            ctx.name_anonymous_dims(inp)
    func = _registry.registry.get(domain, op_type, version=ctx.opset)
    assert func is not None, f"No inference for {domain}::{op_type}"
    func(ctx, node)
    return outputs


FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class ShapeOpPropagationTest(unittest.TestCase):
    """Test that Shape op stores symbolic_value."""

    def test_shape_stores_symbolic_value(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(FLOAT, ["N", 3, "H", "W"]))
        [out] = _run_node(ctx, "", "Shape", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 4)
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(sym_val[1], 3)
        self.assertEqual(str(sym_val[2]), "H")
        self.assertEqual(str(sym_val[3]), "W")

    def test_shape_with_start_end(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(FLOAT, ["N", 3, "H", "W"]))
        [out] = _run_node(
            ctx,
            "",
            "Shape",
            [x],
            attributes={"start": ir.Attr("start", ir.AttributeType.INT, 2)},
        )
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 2)
        self.assertEqual(str(sym_val[0]), "H")
        self.assertEqual(str(sym_val[1]), "W")


class SizeOpPropagationTest(unittest.TestCase):
    """Test that Size op stores symbolic_value (total number of elements)."""

    def test_size_concrete_shape(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(FLOAT, [3, 4, 5]))
        [out] = _run_node(ctx, "", "Size", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(sym_val, [60])

    def test_size_symbolic_shape(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(FLOAT, ["N", 3, "H"]))
        [out] = _run_node(ctx, "", "Size", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        # Product of symbolic dims is a symbolic expression
        self.assertEqual(len(sym_val), 1)

    def test_size_scalar(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(FLOAT, []))
        [out] = _run_node(ctx, "", "Size", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(sym_val, [1])

    def test_size_rank1(self):
        """Size of a 1-D tensor is the single dim value."""
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(FLOAT, [7]))
        [out] = _run_node(ctx, "", "Size", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(sym_val, [7])

    def test_size_no_shape(self):
        """Size with unknown shape → no symbolic_value."""
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(FLOAT))
        [out] = _run_node(ctx, "", "Size", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNone(sym_val)


class IdentityPropagationTest(unittest.TestCase):
    """Test that Identity forwards symbolic_value."""

    def test_identity_propagates_symbolic_value(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(INT64, [4]))
        ctx.set_symbolic_value(x, [ir.SymbolicDim("N"), 3, ir.SymbolicDim("H"), ir.SymbolicDim("W")])
        [out] = _run_node(ctx, "", "Identity", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 4)
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(sym_val[1], 3)

    def test_identity_no_symbolic_value(self):
        """Identity with no symbolic_value on input → no propagation."""
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(FLOAT, [3, 4]))
        [out] = _run_node(ctx, "", "Identity", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNone(sym_val)

    def test_identity_scalar_symbolic_value(self):
        """Identity forwards scalar symbolic value."""
        ctx = _context.ShapeInferenceContext({"": 17})
        x = _make_value("x", ts(INT64, []))
        ctx.set_symbolic_value(x, [42])
        [out] = _run_node(ctx, "", "Identity", [x])
        sym_val = ctx.get_symbolic_value(out)
        self.assertEqual(sym_val, [42])


class SlicePropagationTest(unittest.TestCase):
    """Test that Slice propagates symbolic_value."""

    def test_slice_propagates_symbolic_value(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        # Simulate Shape output with symbolic_value [N, 3, H, W]
        data = ir.Value(name="shape_out", type=ir.TensorType(INT64))
        data.shape = ir.Shape([4])
        ctx.set_symbolic_value(
            data, [ir.SymbolicDim("N"), 3, ir.SymbolicDim("H"), ir.SymbolicDim("W")]
        )

        starts = const_value([0], "starts")
        ends = const_value([2], "ends")
        [out] = _run_node(ctx, "", "Slice", [data, starts, ends])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 2)
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(sym_val[1], 3)

    def test_slice_end_with_sentinel(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="shape_out", type=ir.TensorType(INT64))
        data.shape = ir.Shape([4])
        ctx.set_symbolic_value(
            data, [ir.SymbolicDim("N"), 3, ir.SymbolicDim("H"), ir.SymbolicDim("W")]
        )

        starts = const_value([2], "starts")
        ends = const_value([2**63 - 1], "ends")  # sentinel for end
        [out] = _run_node(ctx, "", "Slice", [data, starts, ends])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 2)
        self.assertEqual(str(sym_val[0]), "H")
        self.assertEqual(str(sym_val[1]), "W")


class GatherPropagationTest(unittest.TestCase):
    """Test that Gather propagates symbolic_value."""

    def test_gather_scalar_index(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="shape_out", type=ir.TensorType(INT64))
        data.shape = ir.Shape([4])
        ctx.set_symbolic_value(
            data, [ir.SymbolicDim("N"), 3, ir.SymbolicDim("H"), ir.SymbolicDim("W")]
        )

        indices = const_value(np.array(0, dtype=np.int64), "idx", dtype=np.int64)
        indices.shape = ir.Shape([])
        [out] = _run_node(ctx, "", "Gather", [data, indices])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 1)
        self.assertEqual(str(sym_val[0]), "N")

    def test_gather_1d_indices(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="shape_out", type=ir.TensorType(INT64))
        data.shape = ir.Shape([4])
        ctx.set_symbolic_value(
            data, [ir.SymbolicDim("N"), 3, ir.SymbolicDim("H"), ir.SymbolicDim("W")]
        )

        indices = const_value([0, 2], "idx")
        [out] = _run_node(ctx, "", "Gather", [data, indices])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 2)
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(str(sym_val[1]), "H")


class ConcatPropagationTest(unittest.TestCase):
    """Test that Concat propagates symbolic_value."""

    def test_concat_symbolic_values(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        a = ir.Value(name="a", type=ir.TensorType(INT64))
        a.shape = ir.Shape([2])
        ctx.set_symbolic_value(a, [ir.SymbolicDim("N"), 3])

        b = const_value([768], "b")
        [out] = _run_node(
            ctx,
            "",
            "Concat",
            [a, b],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
        )
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 3)
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(sym_val[1], 3)
        self.assertEqual(sym_val[2], 768)


class ArithmeticPropagationTest(unittest.TestCase):
    """Test that Add/Sub/Mul/Div propagate symbolic_value."""

    def test_mul_propagates(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        a = ir.Value(name="a", type=ir.TensorType(INT64))
        a.shape = ir.Shape([2])
        ctx.set_symbolic_value(a, [ir.SymbolicDim("N"), 3])

        b = const_value([2, 2], "b")
        [out] = _run_node(ctx, "", "Mul", [a, b])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 2)
        self.assertEqual(str(sym_val[0]), "2*N")
        self.assertEqual(sym_val[1], 6)

    def test_add_propagates(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        a = ir.Value(name="a", type=ir.TensorType(INT64))
        a.shape = ir.Shape([2])
        ctx.set_symbolic_value(a, [ir.SymbolicDim("H"), ir.SymbolicDim("W")])

        b = const_value([10, 20], "b")
        [out] = _run_node(ctx, "", "Add", [a, b])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(str(sym_val[0]), "H + 10")
        self.assertEqual(str(sym_val[1]), "W + 20")


class UnaryPropagationTest(unittest.TestCase):
    """Test that Floor, Ceil, Neg propagate symbolic_value."""

    def test_neg_propagates(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="data", type=ir.TensorType(INT64))
        data.shape = ir.Shape([2])
        ctx.set_symbolic_value(data, [ir.SymbolicDim("N"), 3])
        [out] = _run_node(ctx, "", "Neg", [data])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(str(sym_val[0]), "-N")
        self.assertEqual(sym_val[1], -3)

    def test_floor_propagates(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="data", type=ir.TensorType(INT64))
        data.shape = ir.Shape([2])
        ctx.set_symbolic_value(data, [ir.SymbolicDim("N"), 3])
        [out] = _run_node(ctx, "", "Floor", [data])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        # Floor of an integer symbol is itself
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(sym_val[1], 3)

    def test_ceil_propagates(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="data", type=ir.TensorType(INT64))
        data.shape = ir.Shape([2])
        ctx.set_symbolic_value(data, [ir.SymbolicDim("N"), 3])
        [out] = _run_node(ctx, "", "Ceil", [data])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(sym_val[1], 3)


class CastPropagationTest(unittest.TestCase):
    """Test that Cast propagates symbolic_value."""

    def test_cast_propagates(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="data", type=ir.TensorType(INT64))
        data.shape = ir.Shape([3])
        ctx.set_symbolic_value(data, [ir.SymbolicDim("N"), 3, ir.SymbolicDim("H")])

        [out] = _run_node(
            ctx,
            "",
            "Cast",
            [data],
            attributes={"to": ir.Attr("to", ir.AttributeType.INT, ir.DataType.INT32.value)},
        )
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 3)


class SqueezePropagationTest(unittest.TestCase):
    """Test that Squeeze/Unsqueeze propagate symbolic_value."""

    def test_squeeze_propagates(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="data", type=ir.TensorType(INT64))
        data.shape = ir.Shape([1, 3])
        ctx.set_symbolic_value(data, [ir.SymbolicDim("N"), 3, ir.SymbolicDim("H")])

        axes = const_value([0], "axes")
        [out] = _run_node(ctx, "", "Squeeze", [data, axes])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 3)

    def test_unsqueeze_propagates(self):
        ctx = _context.ShapeInferenceContext({"": 17})
        data = ir.Value(name="data", type=ir.TensorType(INT64))
        data.shape = ir.Shape([3])
        ctx.set_symbolic_value(data, [ir.SymbolicDim("N"), 3, ir.SymbolicDim("H")])

        axes = const_value([0], "axes")
        [out] = _run_node(ctx, "", "Unsqueeze", [data, axes])
        sym_val = ctx.get_symbolic_value(out)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 3)


class ReshapePropagationTest(unittest.TestCase):
    """Test that Reshape reads symbolic_value from shape input."""

    def test_reshape_uses_symbolic_value(self):
        """Test the key pattern: Shape → Reshape uses symbolic dims."""
        ctx = _context.ShapeInferenceContext({"": 17})
        data = _make_value("data", ts(FLOAT, [10, 20, 30]))

        # Shape input is not a constant, but has symbolic_value
        shape_val = ir.Value(name="target_shape", type=ir.TensorType(INT64))
        shape_val.shape = ir.Shape([2])
        ctx.set_symbolic_value(shape_val, [ir.SymbolicDim("N"), 768])

        [out] = _run_node(ctx, "", "Reshape", [data, shape_val])
        self.assertEqual(ir.TypeAndShape(out.type, out.shape), ts(FLOAT, ["N", 768]))

    def test_reshape_symbolic_with_minus_one(self):
        """Test Reshape with -1 and symbolic_value."""
        ctx = _context.ShapeInferenceContext({"": 17})
        data = _make_value("data", ts(FLOAT, [4, 3, 8, 8]))

        shape_val = ir.Value(name="target_shape", type=ir.TensorType(INT64))
        shape_val.shape = ir.Shape([2])
        ctx.set_symbolic_value(shape_val, [ir.SymbolicDim("N"), -1])

        [out] = _run_node(ctx, "", "Reshape", [data, shape_val])
        # N is symbolic, -1 cannot be inferred when N is symbolic
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape.rank(), 2)
        self.assertEqual(str(out.shape[0]), "N")

    def test_reshape_symbolic_with_zero(self):
        """Test Reshape with 0 (copy from input) and symbolic_value."""
        ctx = _context.ShapeInferenceContext({"": 17})
        data = _make_value("data", ts(FLOAT, ["N", 3, 8, 8]))

        shape_val = ir.Value(name="target_shape", type=ir.TensorType(INT64))
        shape_val.shape = ir.Shape([3])
        ctx.set_symbolic_value(shape_val, [0, 3, -1])

        [out] = _run_node(ctx, "", "Reshape", [data, shape_val])
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape.rank(), 3)
        # dim 0: copied from input → N
        self.assertEqual(str(out.shape[0]), "N")
        self.assertEqual(out.shape[1], 3)


class EndToEndPropagationTest(unittest.TestCase):
    """Test the full Shape → Slice → Concat → Reshape pipeline."""

    def test_shape_slice_concat_reshape(self):
        """Shape(x)[:2] ++ [768] → Reshape target."""
        ctx = _context.ShapeInferenceContext({"": 17})

        # Input: x with shape [N, C, H, W]
        x = _make_value("x", ts(FLOAT, ["N", "C", "H", "W"]))

        # Shape(x) → [N, C, H, W]
        [shape_out] = _run_node(ctx, "", "Shape", [x])

        # Slice(shape_out, [0], [2]) → [N, C]
        starts = const_value([0], "starts")
        ends = const_value([2], "ends")
        [sliced] = _run_node(ctx, "", "Slice", [shape_out, starts, ends])

        # Concat(sliced, [768]) → [N, C, 768]
        const_768 = const_value([768], "const_768")
        [target] = _run_node(
            ctx,
            "",
            "Concat",
            [sliced, const_768],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
        )

        # Verify the symbolic_value propagated through
        sym_val = ctx.get_symbolic_value(target)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 3)
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(str(sym_val[1]), "C")
        self.assertEqual(sym_val[2], 768)

        # Reshape(data, target) → [N, C, 768]
        data = _make_value("data", ts(FLOAT, ["N", "C", "H", "W"]))
        [reshaped] = _run_node(ctx, "", "Reshape", [data, target])
        self.assertEqual(
            ir.TypeAndShape(reshaped.type, reshaped.shape),
            ts(FLOAT, ["N", "C", 768]),
        )

    def test_shape_gather_concat_reshape(self):
        """Shape(x)[0] → Unsqueeze → Concat([_, 768]) → Reshape target."""
        ctx = _context.ShapeInferenceContext({"": 17})

        # x with shape [N, 3, H, W]
        x = _make_value("x", ts(FLOAT, ["N", 3, "H", "W"]))

        # Shape(x) → [N, 3, H, W]
        [shape_out] = _run_node(ctx, "", "Shape", [x])

        # Gather(shape_out, [0]) → [N]
        idx = const_value([0], "idx")
        [gathered] = _run_node(ctx, "", "Gather", [shape_out, idx])

        # Concat([gathered, [768]]) → [N, 768]
        const_768 = const_value([768], "const_768")
        [target] = _run_node(
            ctx,
            "",
            "Concat",
            [gathered, const_768],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
        )

        sym_val = ctx.get_symbolic_value(target)
        self.assertIsNotNone(sym_val)
        self.assertEqual(len(sym_val), 2)
        self.assertEqual(str(sym_val[0]), "N")
        self.assertEqual(sym_val[1], 768)

        # Reshape(data, target) → [N, 768]
        data = _make_value("data", ts(FLOAT, [10, 20, 30]))
        [reshaped] = _run_node(ctx, "", "Reshape", [data, target])
        self.assertEqual(
            ir.TypeAndShape(reshaped.type, reshaped.shape),
            ts(FLOAT, ["N", 768]),
        )

    def test_get_symbolic_value_falls_back_to_const(self):
        """get_symbolic_value reads const_value when no symbolic_value is set."""
        ctx = _context.ShapeInferenceContext({"": 17})
        v = const_value([3, 4, 5], "test_const")
        result = ctx.get_symbolic_value(v)
        self.assertIsNotNone(result)
        self.assertEqual(result, [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
