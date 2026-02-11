# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Tile operator."""

from __future__ import annotations

__all__ = [
    "infer_tile",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Tile", since_version=13)
def infer_tile(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Tile operator."""
    (input_val, repeats) = _context.check_inputs(node, "input", "repeats")

    output_shape: ir.Shape | None = None
    if input_val.shape is not None and repeats.const_value is not None:
        repeats_vals = [int(x) for x in repeats.const_value.numpy().flatten()]
        new_dims: list[int | ir.SymbolicDim] = []
        for i, dim in enumerate(input_val.shape.dims):
            if i < len(repeats_vals):
                if isinstance(dim, int):
                    new_dims.append(dim * repeats_vals[i])
                else:
                    new_dims.append(ctx.new_symbolic_dim())
            else:
                new_dims.append(dim)
        output_shape = ir.Shape(new_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_val.dtype)
