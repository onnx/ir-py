# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Squeeze and Unsqueeze operators."""

from __future__ import annotations

__all__ = [
    "infer_squeeze",
    "infer_unsqueeze",
]

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


def _read_axes_from_input_or_attr(node: ir.Node) -> list[int] | None:
    """Read axes from second input (opset >= 13) or attribute (opset < 13)."""
    if len(node.inputs) >= 2 and node.inputs[1] is not None:
        const = node.inputs[1].const_value
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
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected at least 1 input, got {len(node.inputs)}")
        return

    data = node.inputs[0]
    if data is None:
        return

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


@_registry.registry.register("", "Unsqueeze", since_version=1)
def infer_unsqueeze(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Unsqueeze operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    """
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected at least 1 input, got {len(node.inputs)}")
        return

    data = node.inputs[0]
    if data is None:
        return

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
