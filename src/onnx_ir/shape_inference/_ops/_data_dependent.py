# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for data-dependent output shape operators."""

from __future__ import annotations

__all__ = [
    "infer_compress",
    "infer_non_zero",
    "infer_unique",
]

import numpy as np

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "NonZero", since_version=13)
def infer_non_zero(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for NonZero operator.

    Output: [rank(X), num_nonzero], dtype=INT64.
    """
    (x,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        x_const = ir.convenience.get_const_tensor(node.inputs[0])  # type: ignore[arg-type]
        if x_const is not None:
            num_nonzero: int | ir.SymbolicDim = int(np.count_nonzero(x_const.numpy()))
            if x.shape is not None:
                output_shape = ir.Shape([x.shape.rank(), num_nonzero])
            else:
                output_shape = ir.Shape([ctx.new_symbolic_dim(), num_nonzero])
        elif x.shape is not None:
            output_shape = ir.Shape([x.shape.rank(), ctx.new_symbolic_dim()])
        else:
            output_shape = ir.Shape([ctx.new_symbolic_dim(), ctx.new_symbolic_dim()])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT64)


@_registry.registry.register("", "Compress", since_version=11)
def infer_compress(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Compress operator.

    Output: 1-D with dynamic length (no axis), or same rank with compressed axis.
    """
    (x,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        # Try to read the condition (input[1]) as a const tensor
        true_count: int | None = None
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            cond_const = ir.convenience.get_const_tensor(node.inputs[1])
            if cond_const is not None:
                true_count = int(np.sum(cond_const.numpy().astype(bool)))

        axis_attr = node.attributes.get("axis")
        if axis_attr is not None and x.shape is not None:
            axis = axis_attr.as_int()
            rank = x.shape.rank()
            if axis < 0:
                axis += rank
            dims: list[int | ir.SymbolicDim] = list(x.shape.dims)
            dims[axis] = true_count if true_count is not None else ctx.new_symbolic_dim()
            output_shape = ir.Shape(dims)
        else:
            compressed_len: int | ir.SymbolicDim = (
                true_count if true_count is not None else ctx.new_symbolic_dim()
            )
            output_shape = ir.Shape([compressed_len])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


@_registry.registry.register("", "Unique", since_version=11)
def infer_unique(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Unique operator.

    All outputs have dynamic shapes.
    """
    (x,) = _context.check_inputs(node, "X")

    axis_attr = node.attributes.get("axis")
    x_const = ir.convenience.get_const_tensor(node.inputs[0])  # type: ignore[arg-type]

    unique_len: int | ir.SymbolicDim = ctx.new_symbolic_dim()
    inv_len: int | ir.SymbolicDim = ctx.new_symbolic_dim()
    y_shape: ir.Shape

    if x_const is not None:
        arr = x_const.numpy()
        if axis_attr is not None:
            axis = axis_attr.as_int()
            unique_vals = np.unique(arr, axis=axis)
            unique_len = unique_vals.shape[axis]
            inv_len = arr.shape[axis]
            # Y has same shape as input except along axis
            y_dims: list[int | ir.SymbolicDim] = list(arr.shape)
            y_dims[axis] = unique_len
            y_shape = ir.Shape(y_dims)
        else:
            unique_vals = np.unique(arr.flatten())
            unique_len = len(unique_vals)
            inv_len = int(arr.size)
            y_shape = ir.Shape([unique_len])
    else:
        y_shape = ir.Shape([unique_len])

    # Y: unique values
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], y_shape, x.dtype)
    # indices — 1-D, same length as Y
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], ir.Shape([unique_len]), ir.DataType.INT64)
    # inverse_indices — 1-D, same length as input (or flattened input)
    if len(node.outputs) > 2:
        ctx.set_shape_and_dtype(node.outputs[2], ir.Shape([inv_len]), ir.DataType.INT64)
    # counts — 1-D, same length as Y
    if len(node.outputs) > 3:
        ctx.set_shape_and_dtype(node.outputs[3], ir.Shape([unique_len]), ir.DataType.INT64)
