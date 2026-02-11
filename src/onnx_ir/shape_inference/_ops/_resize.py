# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Resize operator."""

from __future__ import annotations

__all__ = [
    "infer_resize",
]

import math

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Resize", since_version=19)
def infer_resize(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Resize operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Resize.html
    """
    (x,) = _context.check_inputs(node, "X")

    output_dtype = x.dtype
    x_shape = x.shape

    # Check if sizes input (index 3) has const_value
    if x_shape is not None and len(node.inputs) > 3 and node.inputs[3] is not None:
        sizes_input = node.inputs[3]
        sizes_const = ir.convenience.get_const_tensor(sizes_input)
        if sizes_const is not None:
            output_dims = [int(s) for s in sizes_const.numpy().flatten()]
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)
            return

    # Check if scales input (index 2) has const_value and X has all concrete dims
    if x_shape is not None and len(node.inputs) > 2 and node.inputs[2] is not None:
        scales_input = node.inputs[2]
        scales_const = ir.convenience.get_const_tensor(scales_input)
        if scales_const is not None:
            scales = [float(s) for s in scales_const.numpy().flatten()]
            if len(scales) == x_shape.rank() and x_shape.is_static():
                output_dims_list: list[int | ir.SymbolicDim] = []
                for i in range(x_shape.rank()):
                    d = x_shape[i]
                    if isinstance(d, int):
                        output_dims_list.append(math.floor(d * scales[i]))
                    else:
                        output_dims_list.append(ctx.new_symbolic_dim())
                if len(node.outputs) > 0:
                    ctx.set_shape_and_dtype(
                        node.outputs[0], ir.Shape(output_dims_list), output_dtype
                    )
                return

    # Fallback: same rank with symbolic dims
    if x_shape is not None:
        output_dims_sym: list[int | ir.SymbolicDim] = [
            ctx.new_symbolic_dim() for _ in range(x_shape.rank())
        ]
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims_sym), output_dtype)
    else:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
