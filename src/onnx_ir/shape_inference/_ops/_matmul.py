# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for MatMul operator."""

from __future__ import annotations

__all__ = [
    "infer_matmul",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _broadcast, _context, _registry


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

    shape_a = input_a.shape
    shape_b = input_b.shape
    output_dtype = input_a.dtype or input_b.dtype

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
