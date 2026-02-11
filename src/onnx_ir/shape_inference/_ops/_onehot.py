# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for OneHot operator."""

from __future__ import annotations

__all__ = [
    "infer_onehot",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "OneHot", since_version=11)
def infer_onehot(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for OneHot operator."""
    (indices, _depth, values) = _context.check_inputs(node, "indices", "depth", "values")

    axis_attr = node.attributes.get("axis")
    axis = axis_attr.as_int() if axis_attr is not None else -1

    output_shape: ir.Shape | None = None
    if indices.shape is not None:
        indices_rank = indices.shape.rank()
        output_rank = indices_rank + 1

        if axis < 0:
            axis += output_rank

        new_dims: list[int | ir.SymbolicDim] = []
        for i in range(output_rank):
            if i == axis:
                new_dims.append(ctx.new_symbolic_dim())
            elif i < axis:
                new_dims.append(indices.shape[i])
            else:
                new_dims.append(indices.shape[i - 1])
        output_shape = ir.Shape(new_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, values.dtype)
