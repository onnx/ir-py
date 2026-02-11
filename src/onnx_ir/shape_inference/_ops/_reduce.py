# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic shape inference for Reduce* operators."""

from __future__ import annotations

__all__ = [
    "infer_reduce",
]


import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


def _normalize_axis(axis: int, rank: int) -> int:
    """Normalize a potentially negative axis to a non-negative value."""
    if axis < 0:
        axis += rank
    return axis


def _read_axes(node: ir.Node) -> list[int] | None:
    """Read the axes parameter from either an attribute or the second input.

    Since opset 18, most Reduce ops take ``axes`` as an optional second input.
    Before that, ``axes`` was an attribute.  This helper handles both cases.

    Returns:
        A list of axis integers, or ``None`` if axes are unknown.
    """
    if len(node.inputs) >= 2 and node.inputs[1] is not None:
        axes_value = node.inputs[1]
        const = ir.convenience.get_const_tensor(axes_value)
        if const is not None:
            return [int(x) for x in const.numpy().flatten()]
        return None

    axes_attr = node.attributes.get("axes")
    if axes_attr is not None:
        return list(axes_attr.as_ints())

    return None


@_reg("", "ReduceSum", since_version=1)
@_reg("", "ReduceMean", since_version=1)
@_reg("", "ReduceMax", since_version=1)
@_reg("", "ReduceMin", since_version=1)
@_reg("", "ReduceProd", since_version=1)
@_reg("", "ReduceL1", since_version=1)
@_reg("", "ReduceL2", since_version=1)
@_reg("", "ReduceLogSum", since_version=1)
@_reg("", "ReduceLogSumExp", since_version=1)
@_reg("", "ReduceSumSquare", since_version=1)
def infer_reduce(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for a Reduce operator.

    Handles ``axes`` (from attribute or input), ``keepdims``, and
    ``noop_with_empty_axes``.
    """
    (data,) = _context.check_inputs(node, "data")

    input_shape = data.shape
    input_dtype = data.dtype

    keepdims_attr = node.attributes.get("keepdims")
    keepdims = keepdims_attr.as_int() if keepdims_attr is not None else 1

    noop_attr = node.attributes.get("noop_with_empty_axes")
    noop_with_empty_axes = noop_attr.as_int() if noop_attr is not None else 0

    output_shape: ir.Shape | None = None

    if input_shape is not None:
        rank = input_shape.rank()
        axes = _read_axes(node)

        if axes is not None:
            if len(axes) == 0 and noop_with_empty_axes:
                output_shape = input_shape
            else:
                if len(axes) == 0:
                    normalized_axes = set(range(rank))
                else:
                    normalized_axes = {_normalize_axis(a, rank) for a in axes}

                new_dims: list[int | ir.SymbolicDim] = []
                for i in range(rank):
                    if i in normalized_axes:
                        if keepdims:
                            new_dims.append(1)
                    else:
                        new_dims.append(input_shape[i])

                output_shape = ir.Shape(new_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)
