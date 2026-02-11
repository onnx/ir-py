# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Gemm operator."""

from __future__ import annotations

__all__ = [
    "infer_gemm",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Gemm", since_version=7)
def infer_gemm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Gemm operator.

    Computes Y = alpha * A' * B' + beta * C, where A' and B' are optionally
    transposed.  Output is always 2-D: (M, N).

    Spec: https://onnx.ai/onnx/operators/onnx__Gemm.html
    """
    (input_a, input_b) = _context.check_inputs(node, "A", "B")

    shape_a = input_a.shape
    shape_b = input_b.shape
    output_dtype = input_a.dtype or input_b.dtype

    if shape_a is None or shape_b is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    if shape_a.rank() != 2:
        ctx.record_error(node, f"Gemm input A must be rank 2, got {shape_a.rank()}")
        return
    if shape_b.rank() != 2:
        ctx.record_error(node, f"Gemm input B must be rank 2, got {shape_b.rank()}")
        return

    trans_a_attr = node.attributes.get("transA")
    trans_a = trans_a_attr.as_int() if trans_a_attr is not None else 0
    trans_b_attr = node.attributes.get("transB")
    trans_b = trans_b_attr.as_int() if trans_b_attr is not None else 0

    m_dim = shape_a[1] if trans_a else shape_a[0]
    n_dim = shape_b[0] if trans_b else shape_b[1]

    output_shape = ir.Shape([m_dim, n_dim])
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
