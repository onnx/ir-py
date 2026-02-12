# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Pad operator."""

from __future__ import annotations

__all__ = [
    "infer_pad",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Pad", since_version=13)
def infer_pad(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Pad operator."""
    (data,) = _context.check_inputs(node, "data")

    output_shape: ir.Shape | None = None
    if data.shape is not None:
        rank = data.shape.rank()
        pads_value = node.inputs[1] if len(node.inputs) > 1 else None
        pads_const = (
            ir.convenience.get_const_tensor(pads_value) if pads_value is not None else None
        )

        if pads_const is not None:
            pads = [int(x) for x in pads_const.numpy().flatten()]
            new_dims: list[int | ir.SymbolicDim] = []
            for i in range(rank):
                dim = data.shape[i]
                new_dims.append(dim + pads[i] + pads[i + rank])
            output_shape = ir.Shape(new_dims)
        else:
            output_shape = ir.Shape([ctx.new_symbolic_dim() for _ in range(rank)])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, data.dtype)
