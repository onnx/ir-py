# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for DepthToSpace and SpaceToDepth operators."""

from __future__ import annotations

__all__ = [
    "infer_depth_to_space",
    "infer_space_to_depth",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "DepthToSpace", since_version=13)
def infer_depth_to_space(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for DepthToSpace operator."""
    (x,) = _context.check_inputs(node, "input")

    blocksize_attr = _context.require_attr(node, "blocksize")
    b = blocksize_attr.as_int()

    output_shape: ir.Shape | None = None
    if x.shape is not None and x.shape.rank() == 4:
        n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        if isinstance(c, int):
            new_c: int | ir.SymbolicDim = c // (b * b)
        else:
            new_c = ctx.new_symbolic_dim()

        if isinstance(h, int):
            new_h: int | ir.SymbolicDim = h * b
        else:
            new_h = ctx.new_symbolic_dim()

        if isinstance(w, int):
            new_w: int | ir.SymbolicDim = w * b
        else:
            new_w = ctx.new_symbolic_dim()

        output_shape = ir.Shape([n, new_c, new_h, new_w])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


@_registry.registry.register("", "SpaceToDepth", since_version=13)
def infer_space_to_depth(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for SpaceToDepth operator."""
    (x,) = _context.check_inputs(node, "input")

    blocksize_attr = _context.require_attr(node, "blocksize")
    b = blocksize_attr.as_int()

    output_shape: ir.Shape | None = None
    if x.shape is not None and x.shape.rank() == 4:
        n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        if isinstance(c, int):
            new_c: int | ir.SymbolicDim = c * b * b
        else:
            new_c = ctx.new_symbolic_dim()

        if isinstance(h, int):
            new_h: int | ir.SymbolicDim = h // b
        else:
            new_h = ctx.new_symbolic_dim()

        if isinstance(w, int):
            new_w: int | ir.SymbolicDim = w // b
        else:
            new_w = ctx.new_symbolic_dim()

        output_shape = ir.Shape([n, new_c, new_h, new_w])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)
