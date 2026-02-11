# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for shape inference using Hypothesis.

These tests verify algebraic invariants (commutativity, rank preservation,
dtype propagation) rather than exact output values, complementing the
unit tests in sibling modules.
"""

from __future__ import annotations

import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

import onnx_ir as ir
from onnx_ir.shape_inference import _context
from onnx_ir.shape_inference._broadcast import broadcast_shapes
from onnx_ir.shape_inference._ops._control_flow import _merge_shapes
from onnx_ir.shape_inference._ops._testing import run_shape_inference, ts

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Concrete dimension: small positive ints (plus 1 for broadcast testing)
_concrete_dim = st.integers(min_value=1, max_value=16)

# Symbolic dimension with a name
_SYMBOLIC_NAMES = ["N", "M", "batch", "seq", "hidden"]
_symbolic_dim = st.sampled_from(_SYMBOLIC_NAMES).map(ir.SymbolicDim)

# Any dimension: concrete or symbolic
_any_dim = st.one_of(_concrete_dim, _symbolic_dim)

# Shapes with rank 0–5
_shape = st.lists(_any_dim, min_size=0, max_size=5).map(ir.Shape)

# Concrete-only shapes (for broadcast, which has clearer rules on concrete dims)
_concrete_shape = st.lists(_concrete_dim, min_size=0, max_size=5).map(ir.Shape)

# Optional shape (may be None)
_optional_shape = st.one_of(st.none(), _shape)

# Broadcastable concrete dimension (either matching or one is 1)
_FLOAT_DTYPES = [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.DOUBLE]
_ALL_DTYPES = [ir.DataType.FLOAT, ir.DataType.INT32, ir.DataType.INT64, ir.DataType.BOOL]

# Unary ops that preserve dtype
_UNARY_OPS = [
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Ceil",
    "Cos",
    "Cosh",
    "Elu",
    "Erf",
    "Exp",
    "Floor",
    "HardSigmoid",
    "Identity",
    "LeakyRelu",
    "Log",
    "Neg",
    "Reciprocal",
    "Relu",
    "Round",
    "Selu",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
]

# Binary elementwise ops that preserve dtype
_BINARY_ARITHMETIC_OPS = ["Add", "Sub", "Mul", "Div"]

# Binary comparison ops (output BOOL)
_BINARY_COMPARISON_OPS = ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"]


# ---------------------------------------------------------------------------
# broadcast_shapes
# ---------------------------------------------------------------------------
class BroadcastShapesPropertyTest(unittest.TestCase):
    @given(s=_shape)
    @settings(max_examples=100)
    def test_broadcast_with_self_returns_self(self, s: ir.Shape):
        """broadcast(S, S) == S for any shape S."""
        result = broadcast_shapes(s, s)
        self.assertIsNotNone(result)
        self.assertEqual(result.rank(), s.rank())

    @given(s1=_concrete_shape, s2=_concrete_shape)
    @settings(max_examples=200)
    def test_broadcast_is_commutative(self, s1: ir.Shape, s2: ir.Shape):
        """broadcast(S1, S2) == broadcast(S2, S1) for concrete shapes."""
        r1 = broadcast_shapes(s1, s2)
        r2 = broadcast_shapes(s2, s1)
        # Both succeed or both fail
        if r1 is None:
            self.assertIsNone(r2)
        else:
            self.assertIsNotNone(r2)
            self.assertEqual(r1, r2)

    @given(s1=_concrete_shape, s2=_concrete_shape)
    @settings(max_examples=200)
    def test_broadcast_rank_is_max(self, s1: ir.Shape, s2: ir.Shape):
        """Broadcast output rank == max(rank1, rank2) when compatible."""
        result = broadcast_shapes(s1, s2)
        if result is not None:
            self.assertEqual(result.rank(), max(s1.rank(), s2.rank()))

    @given(s=_concrete_shape)
    @settings(max_examples=100)
    def test_broadcast_with_scalar(self, s: ir.Shape):
        """broadcast(S, scalar) == S."""
        scalar = ir.Shape([])
        result = broadcast_shapes(s, scalar)
        self.assertIsNotNone(result)
        self.assertEqual(result.rank(), s.rank())

    @given(s=_shape)
    @settings(max_examples=100)
    def test_broadcast_with_none_returns_none(self, s: ir.Shape):
        """broadcast(S, None) == None."""
        self.assertIsNone(broadcast_shapes(s, None))
        self.assertIsNone(broadcast_shapes(None, s))


# ---------------------------------------------------------------------------
# _merge_shapes
# ---------------------------------------------------------------------------
class MergeShapesPropertyTest(unittest.TestCase):
    def _ctx(self) -> _context.ShapeInferenceContext:
        return _context.ShapeInferenceContext(policy="override")

    @given(s=_shape)
    @settings(max_examples=100)
    def test_merge_with_self_preserves_rank(self, s: ir.Shape):
        """merge(S, S) has the same rank as S."""
        result = _merge_shapes(self._ctx(), s, s)
        self.assertIsNotNone(result)
        self.assertEqual(result.rank(), s.rank())

    @given(s1=_concrete_shape, s2=_concrete_shape)
    @settings(max_examples=200)
    def test_merge_is_commutative_on_rank(self, s1: ir.Shape, s2: ir.Shape):
        """Merge returns None symmetrically for different ranks."""
        ctx = self._ctx()
        r1 = _merge_shapes(ctx, s1, s2)
        r2 = _merge_shapes(ctx, s2, s1)
        if r1 is None:
            self.assertIsNone(r2)
        else:
            self.assertIsNotNone(r2)
            self.assertEqual(r1.rank(), r2.rank())

    @given(s=_shape)
    @settings(max_examples=100)
    def test_merge_with_none_returns_none(self, s: ir.Shape):
        """merge(S, None) == None."""
        ctx = self._ctx()
        self.assertIsNone(_merge_shapes(ctx, s, None))
        self.assertIsNone(_merge_shapes(ctx, None, s))

    @given(s1=_concrete_shape, s2=_concrete_shape)
    @settings(max_examples=200)
    def test_merge_preserves_matching_concrete_dims(self, s1: ir.Shape, s2: ir.Shape):
        """Where both shapes have the same concrete dim, it is preserved."""
        ctx = self._ctx()
        result = _merge_shapes(ctx, s1, s2)
        if result is None:
            return
        for d1, d2, dr in zip(s1.dims, s2.dims, result.dims):
            if isinstance(d1, int) and isinstance(d2, int) and d1 == d2:
                self.assertEqual(dr, d1)


# ---------------------------------------------------------------------------
# Unary ops: shape and dtype preservation
# ---------------------------------------------------------------------------
class UnaryOpsPropertyTest(unittest.TestCase):
    @given(
        op=st.sampled_from(_UNARY_OPS),
        dims=st.lists(_concrete_dim, min_size=0, max_size=5),
        dtype=st.sampled_from(_FLOAT_DTYPES),
    )
    @settings(max_examples=300)
    def test_unary_preserves_shape_and_dtype(
        self, op: str, dims: list[int], dtype: ir.DataType
    ):
        """Unary ops produce output with the same shape and dtype as input."""
        input_shapes = [ts(dtype, dims)]
        results = run_shape_inference("", op, input_shapes, opset_version=21)
        self.assertEqual(len(results), 1)
        out = results[0]
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape, ir.Shape(dims))
        self.assertEqual(out.type.dtype, dtype)

    @given(
        op=st.sampled_from(_UNARY_OPS),
        sym_names=st.lists(st.sampled_from(_SYMBOLIC_NAMES), min_size=1, max_size=4),
        dtype=st.sampled_from(_FLOAT_DTYPES),
    )
    @settings(max_examples=200)
    def test_unary_preserves_symbolic_rank(
        self, op: str, sym_names: list[str], dtype: ir.DataType
    ):
        """Unary ops preserve rank even with symbolic dimensions."""
        input_shapes = [ts(dtype, sym_names)]
        results = run_shape_inference("", op, input_shapes, opset_version=21)
        out = results[0]
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape.rank(), len(sym_names))


# ---------------------------------------------------------------------------
# Binary elementwise ops: broadcast properties
# ---------------------------------------------------------------------------
class BinaryOpsPropertyTest(unittest.TestCase):
    @given(
        op=st.sampled_from(_BINARY_ARITHMETIC_OPS),
        dims=st.lists(_concrete_dim, min_size=0, max_size=5),
        dtype=st.sampled_from(_FLOAT_DTYPES),
    )
    @settings(max_examples=200)
    def test_binary_same_shape_preserves(self, op: str, dims: list[int], dtype: ir.DataType):
        """Binary op with identical shapes produces that shape."""
        t = ts(dtype, dims)
        results = run_shape_inference("", op, [t, t], opset_version=21)
        out = results[0]
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape, ir.Shape(dims))
        self.assertEqual(out.type.dtype, dtype)

    @given(
        op=st.sampled_from(_BINARY_ARITHMETIC_OPS),
        dims=st.lists(_concrete_dim, min_size=1, max_size=5),
        dtype=st.sampled_from(_FLOAT_DTYPES),
    )
    @settings(max_examples=200)
    def test_binary_with_scalar_preserves_shape(
        self, op: str, dims: list[int], dtype: ir.DataType
    ):
        """Binary op with a scalar broadcasts to the other shape."""
        t = ts(dtype, dims)
        scalar = ts(dtype, [])
        results = run_shape_inference("", op, [t, scalar], opset_version=21)
        out = results[0]
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape, ir.Shape(dims))

    @given(
        op=st.sampled_from(_BINARY_COMPARISON_OPS),
        dims=st.lists(_concrete_dim, min_size=0, max_size=5),
        dtype=st.sampled_from(_FLOAT_DTYPES),
    )
    @settings(max_examples=200)
    def test_comparison_output_is_bool(self, op: str, dims: list[int], dtype: ir.DataType):
        """Comparison ops always output BOOL dtype."""
        t = ts(dtype, dims)
        results = run_shape_inference("", op, [t, t], opset_version=21)
        out = results[0]
        self.assertEqual(out.type.dtype, ir.DataType.BOOL)
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape, ir.Shape(dims))


if __name__ == "__main__":
    unittest.main()
