# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ArgMax and ArgMin operators."""

from __future__ import annotations

__all__ = [
    "infer_argmax_argmin",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


def _normalize_axis(axis: int, rank: int) -> int:
    """Normalize a potentially negative axis to a non-negative value."""
    if axis < 0:
        axis += rank
    return axis


@_reg("", "ArgMax", since_version=13)
@_reg("", "ArgMin", since_version=13)
def infer_argmax_argmin(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for ArgMax/ArgMin operator."""
    (data,) = _context.check_inputs(node, "data")

    axis_attr = node.attributes.get("axis")
    axis = axis_attr.as_int() if axis_attr is not None else 0

    keepdims_attr = node.attributes.get("keepdims")
    keepdims = keepdims_attr.as_int() if keepdims_attr is not None else 1

    output_shape: ir.Shape | None = None
    if data.shape is not None:
        rank = data.shape.rank()
        axis = _normalize_axis(axis, rank)

        new_dims: list[int | ir.SymbolicDim] = []
        for i in range(rank):
            if i == axis:
                if keepdims:
                    new_dims.append(1)
            else:
                new_dims.append(data.shape[i])
        output_shape = ir.Shape(new_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT64)
