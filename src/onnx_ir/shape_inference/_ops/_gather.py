# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Gather operator."""

from __future__ import annotations

__all__ = [
    "infer_gather",
    "infer_gather_elements",
    "infer_gather_nd",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Gather", since_version=1)
def infer_gather(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Gather operator.

    output_shape = data_shape[:axis] + indices_shape + data_shape[axis+1:]

    Spec: https://onnx.ai/onnx/operators/onnx__Gather.html
    """
    (data, indices) = _context.check_inputs(node, "data", "indices")

    data_shape = data.shape
    indices_shape = indices.shape
    output_dtype = data.dtype

    output_shape: ir.Shape | None = None
    if data_shape is not None and indices_shape is not None:
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr is not None else 0

        rank = data_shape.rank()
        if axis < 0:
            axis += rank

        if not 0 <= axis < rank:
            ctx.record_error(node, f"axis={axis} is out of range for rank {rank}")
            return

        output_dims: list[int | ir.SymbolicDim] = []
        output_dims.extend(data_shape[:axis])
        output_dims.extend(indices_shape.dims)
        output_dims.extend(data_shape[axis + 1 :])
        output_shape = ir.Shape(output_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_registry.registry.register("", "GatherElements", since_version=13)
def infer_gather_elements(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for GatherElements operator.

    Output shape = indices shape, output dtype = data dtype.
    """
    (data, indices) = _context.check_inputs(node, "data", "indices")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], indices.shape, data.dtype)


@_registry.registry.register("", "GatherND", since_version=12)
def infer_gather_nd(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for GatherND operator.

    Output shape is complex; set dtype only for now (graceful degradation).
    """
    (data, _indices) = _context.check_inputs(node, "data", "indices")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], None, data.dtype)
