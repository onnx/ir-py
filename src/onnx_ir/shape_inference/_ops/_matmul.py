# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for MatMul operator."""

from __future__ import annotations

__all__ = [
    "infer_matmul",
    "infer_matmul_integer",
    "infer_qlinear_matmul",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _broadcast, _context, _registry


def _matmul_shape(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    shape_a: ir.Shape | None,
    shape_b: ir.Shape | None,
    output_dtype: ir.DataType | None,
) -> None:
    """Compute matmul output shape and set on the first output."""
    if shape_a is None or shape_b is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    rank_a = shape_a.rank()
    rank_b = shape_b.rank()

    # Handle 1-D cases
    if rank_a == 1 and rank_b == 1:
        # Dot product -> scalar
        output_shape = ir.Shape([])
    elif rank_a == 1:
        # (K,) x (..., K, N) -> (..., N)
        output_dims = [*shape_b.dims[:-2], shape_b.dims[-1]]
        output_shape = ir.Shape(output_dims)
    elif rank_b == 1:
        # (..., M, K) x (K,) -> (..., M)
        output_dims = list(shape_a.dims[:-1])
        output_shape = ir.Shape(output_dims)
    else:
        # (..., M, K) x (..., K, N) -> (..., M, N)
        # Broadcast batch dimensions
        batch_a = ir.Shape(list(shape_a.dims[:-2])) if rank_a > 2 else ir.Shape([])
        batch_b = ir.Shape(list(shape_b.dims[:-2])) if rank_b > 2 else ir.Shape([])
        batch_shape = _broadcast.broadcast_shapes(batch_a, batch_b)

        m_dim = shape_a.dims[-2]
        n_dim = shape_b.dims[-1]

        if batch_shape is not None:
            output_dims = [*batch_shape.dims, m_dim, n_dim]
        else:
            output_dims = [m_dim, n_dim]
        output_shape = ir.Shape(output_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_registry.registry.register("", "MatMul", since_version=1)
def infer_matmul(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for MatMul operator.

    Follows NumPy matmul semantics:
    - 1-D x 1-D: dot product -> scalar
    - 2-D x 2-D: matrix multiply -> (M, N)
    - Broadcast batch dims for higher-rank inputs

    Spec: https://onnx.ai/onnx/operators/onnx__MatMul.html
    """
    (input_a, input_b) = _context.check_inputs(node, "A", "B")
    output_dtype = input_a.dtype or input_b.dtype
    _matmul_shape(ctx, node, input_a.shape, input_b.shape, output_dtype)


@_registry.registry.register("", "MatMulInteger", since_version=10)
def infer_matmul_integer(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for MatMulInteger operator.

    Same shape logic as MatMul but output dtype is INT32.
    """
    (input_a, input_b) = _context.check_inputs(node, "A", "B")
    _matmul_shape(ctx, node, input_a.shape, input_b.shape, ir.DataType.INT32)


@_registry.registry.register("", "QLinearMatMul", since_version=10)
def infer_qlinear_matmul(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for QLinearMatMul operator.

    Inputs: a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point.
    Shape = matmul(a, b), dtype from y_zero_point (input 7).
    """
    (a,) = _context.check_inputs(node, "a")
    # b is input 3
    output_dtype = ir.DataType.UINT8
    if len(node.inputs) > 7 and node.inputs[7] is not None:
        zp_dtype = node.inputs[7].dtype
        if zp_dtype is not None:
            output_dtype = zp_dtype
    shape_b = (
        node.inputs[3].shape if len(node.inputs) > 3 and node.inputs[3] is not None else None
    )
    _matmul_shape(ctx, node, a.shape, shape_b, output_dtype)
