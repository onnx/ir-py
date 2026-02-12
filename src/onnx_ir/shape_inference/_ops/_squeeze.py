# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Squeeze and Unsqueeze operators."""

from __future__ import annotations

__all__ = [
    "infer_squeeze",
    "infer_unsqueeze",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


def _read_axes_from_input_or_attr(node: ir.Node) -> list[int] | None:
    """Read axes from second input (opset >= 13) or attribute (opset < 13)."""
    if len(node.inputs) >= 2 and node.inputs[1] is not None:
        const = ir.convenience.get_const_tensor(node.inputs[1])
        if const is not None:
            return [int(x) for x in const.numpy().flatten()]
        return None
    axes_attr = node.attributes.get("axes")
    if axes_attr is not None:
        return list(axes_attr.as_ints())
    return None


@_registry.registry.register("", "Squeeze", since_version=1)
def infer_squeeze(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Squeeze operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Squeeze.html
    """
    (data,) = _context.check_inputs(node, "data")

    input_shape = data.shape
    input_dtype = data.dtype

    output_shape: ir.Shape | None = None
    if input_shape is not None:
        rank = input_shape.rank()
        axes = _read_axes_from_input_or_attr(node)

        if axes is not None:
            normalized = {(a + rank if a < 0 else a) for a in axes}
            new_dims = [input_shape[i] for i in range(rank) if i not in normalized]
            output_shape = ir.Shape(new_dims)
        else:
            # No axes specified: remove all dims that are statically 1
            new_dims = [d for d in input_shape.dims if d != 1]
            output_shape = ir.Shape(new_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)

        # Propagate symbolic_value (data values are unchanged by squeeze)
        sym_val = ctx.get_symbolic_value(data)
        if sym_val is not None:
            ctx.set_symbolic_value(node.outputs[0], sym_val)


@_registry.registry.register("", "Unsqueeze", since_version=1)
def infer_unsqueeze(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Unsqueeze operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    """
    (data,) = _context.check_inputs(node, "data")

    input_shape = data.shape
    input_dtype = data.dtype

    output_shape: ir.Shape | None = None
    if input_shape is not None:
        axes = _read_axes_from_input_or_attr(node)
        if axes is not None:
            rank = input_shape.rank()
            output_rank = rank + len(axes)
            # Normalize axes to output rank
            normalized = sorted((a + output_rank if a < 0 else a) for a in axes)

            new_dims: list[int | ir.SymbolicDim] = list(input_shape.dims)
            for a in normalized:
                new_dims.insert(a, 1)
            output_shape = ir.Shape(new_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)

        # Propagate symbolic_value (data values are unchanged by unsqueeze)
        sym_val = ctx.get_symbolic_value(data)
        if sym_val is not None:
            ctx.set_symbolic_value(node.outputs[0], sym_val)
