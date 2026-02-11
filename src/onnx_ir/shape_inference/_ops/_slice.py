# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Slice operator."""

from __future__ import annotations

__all__ = [
    "infer_slice",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


def _read_const_ints(value: ir.Value | None) -> list[int] | None:
    """Read a 1-D constant integer tensor, or return None."""
    if value is None:
        return None
    const = value.const_value
    if const is None:
        return None
    return [int(x) for x in const.numpy().flatten()]


@_registry.registry.register("", "Slice", since_version=10)
def infer_slice(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Slice operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Slice.html
    """
    (data, _, _) = _context.check_inputs(node, "data", "starts", "ends")

    input_shape = data.shape
    input_dtype = data.dtype

    if input_shape is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, input_dtype)
        return

    rank = input_shape.rank()
    starts = _read_const_ints(node.inputs[1])
    ends = _read_const_ints(node.inputs[2])

    if starts is None or ends is None:
        # Dynamic starts/ends — same rank, sliced dims are symbolic
        if len(node.outputs) > 0:
            symbolic_dims: list[int | ir.SymbolicDim] = [
                ctx.new_symbolic_dim() for _ in range(rank)
            ]
            ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(symbolic_dims), input_dtype)
        return

    axes: list[int] | None = None
    if len(node.inputs) >= 4:
        axes = _read_const_ints(node.inputs[3])

    steps: list[int] | None = None
    if len(node.inputs) >= 5:
        steps = _read_const_ints(node.inputs[4])

    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)

    output_dims: list[int | ir.SymbolicDim] = list(input_shape.dims)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            continue

        dim = input_shape[axis]
        if step == 0:
            ctx.record_error(node, f"Step cannot be 0 for axis {axis}")
            return

        if isinstance(dim, int):
            # Clamp start/end to [0, dim] for positive step, [-1, dim-1] for negative
            if step > 0:
                clamped_start = max(0, min(start if start >= 0 else start + dim, dim))
                clamped_end = max(0, min(end if end >= 0 else end + dim, dim))
            else:
                clamped_start = max(-1, min(start if start >= 0 else start + dim, dim - 1))
                clamped_end = max(-1, min(end if end >= 0 else end + dim, dim - 1))

            slice_len = max(
                0, (clamped_end - clamped_start + (step - (1 if step > 0 else -1))) // step
            )
            output_dims[axis] = slice_len
        else:
            # Symbolic dim: can't determine exact size
            output_dims[axis] = ctx.new_symbolic_dim()

    output_shape = ir.Shape(output_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)
