# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Concat operator."""

from __future__ import annotations

__all__ = [
    "infer_concat",
]

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


@_registry.registry.register("", "Concat", since_version=4)
def infer_concat(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Concat operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Concat.html
    """
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected at least 1 input, got {len(node.inputs)}")
        return

    axis_attr = node.attributes.get("axis")
    if axis_attr is None:
        ctx.record_error(node, "Missing required attribute 'axis'")
        return
    axis = axis_attr.as_int()

    # Collect shapes and dtype
    shapes: list[ir.Shape] = []
    output_dtype: ir.DataType | None = None

    for inp in node.inputs:
        if inp is None:
            return
        if output_dtype is None:
            output_dtype = inp.dtype
        if inp.shape is None:
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
            return
        shapes.append(inp.shape)

    if not shapes:
        return

    rank = shapes[0].rank()
    # Normalize negative axis
    if axis < 0:
        axis += rank

    if not 0 <= axis < rank:
        ctx.record_error(node, f"axis={axis} is out of range for rank {rank}")
        return

    for i, s in enumerate(shapes):
        if s.rank() != rank:
            ctx.record_error(
                node,
                f"Input {i} has rank {s.rank()}, expected {rank}",
            )
            return

    # Build output shape
    output_dims: list[int | ir.SymbolicDim] = []
    for dim_idx in range(rank):
        if dim_idx == axis:
            # Sum along concat axis
            total: int | ir.SymbolicDim | None = 0
            for s in shapes:
                d = s[dim_idx]
                if isinstance(total, int) and isinstance(d, int):
                    total += d
                else:
                    total = ir.SymbolicDim(None)
                    break
            output_dims.append(total)  # type: ignore[arg-type]
        else:
            output_dims.append(shapes[0][dim_idx])

    output_shape = ir.Shape(output_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
