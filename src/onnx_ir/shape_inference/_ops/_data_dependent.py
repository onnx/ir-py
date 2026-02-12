# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for data-dependent output shape operators."""

from __future__ import annotations

__all__ = [
    "infer_compress",
    "infer_non_zero",
    "infer_unique",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "NonZero", since_version=13)
def infer_non_zero(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for NonZero operator.

    Output: [rank(X), num_nonzero], dtype=INT64.
    """
    (x,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        if x.shape is not None:
            output_shape = ir.Shape([x.shape.rank(), ctx.new_symbolic_dim()])
        else:
            output_shape = ir.Shape([ctx.new_symbolic_dim(), ctx.new_symbolic_dim()])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT64)


@_registry.registry.register("", "Compress", since_version=11)
def infer_compress(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Compress operator.

    Output: 1-D with dynamic length.
    """
    (x,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        output_shape = ir.Shape([ctx.new_symbolic_dim()])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


@_registry.registry.register("", "Unique", since_version=11)
def infer_unique(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Unique operator.

    All outputs have dynamic shapes.
    """
    (x,) = _context.check_inputs(node, "X")

    unique_len = ctx.new_symbolic_dim()

    # Y: unique values — 1-D with dynamic length
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([unique_len]), x.dtype)
    # indices — 1-D, same length as Y
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], ir.Shape([unique_len]), ir.DataType.INT64)
    # inverse_indices — 1-D, same length as input (or flattened input)
    if len(node.outputs) > 2:
        inv_len = ctx.new_symbolic_dim()
        ctx.set_shape_and_dtype(node.outputs[2], ir.Shape([inv_len]), ir.DataType.INT64)
    # counts — 1-D, same length as Y
    if len(node.outputs) > 3:
        ctx.set_shape_and_dtype(node.outputs[3], ir.Shape([unique_len]), ir.DataType.INT64)
