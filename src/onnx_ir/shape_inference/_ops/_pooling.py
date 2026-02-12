# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for pooling operators."""

from __future__ import annotations

__all__ = [
    "infer_average_pool",
    "infer_global_pool",
    "infer_lp_pool",
    "infer_max_pool",
]

import math

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


def _compute_pool_output_shape(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    x_shape: ir.Shape,
    kernel_shape: list[int],
    strides: list[int],
    pads: list[int],
    dilations: list[int],
    ceil_mode: int,
    auto_pad: str = "NOTSET",
) -> list[int | ir.SymbolicDim]:
    """Compute spatial output dimensions for pooling ops."""
    n_spatial = len(kernel_shape)
    spatial_dims: list[int | ir.SymbolicDim] = []

    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(n_spatial):
            in_dim = x_shape[i + 2]
            s = strides[i]
            if isinstance(in_dim, int):
                spatial_dims.append(math.ceil(in_dim / s))
            else:
                spatial_dims.append(ctx.new_symbolic_dim())
        return spatial_dims

    for i in range(n_spatial):
        in_dim = x_shape[i + 2]
        k = kernel_shape[i]
        s = strides[i]
        d = dilations[i]
        pad_begin = pads[i]
        pad_end = pads[i + n_spatial]
        effective_kernel = d * (k - 1) + 1
        numerator: int | ir.SymbolicDim = in_dim + pad_begin + pad_end - effective_kernel
        if ceil_mode:
            out_dim: int | ir.SymbolicDim = math.ceil(numerator / s) + 1  # type: ignore[operator]
            # If last pooling window starts in the padding, reduce output by 1
            if isinstance(out_dim, int) and isinstance(in_dim, int):
                if (out_dim - 1) * s - pad_begin >= in_dim:
                    out_dim -= 1
        else:
            out_dim = numerator // s + 1
        spatial_dims.append(out_dim)
    return spatial_dims


@_reg("", "AveragePool", since_version=11)
def infer_average_pool(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for AveragePool operator."""
    (x,) = _context.check_inputs(node, "X")

    output_dtype = x.dtype
    x_shape = x.shape

    if x_shape is None:
        ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    n_spatial = x_shape.rank() - 2

    kernel_shape = list(_context.require_attr(node, "kernel_shape").as_ints())

    auto_pad_attr = node.attributes.get("auto_pad")
    auto_pad = auto_pad_attr.as_string() if auto_pad_attr is not None else "NOTSET"

    strides_attr = node.attributes.get("strides")
    strides = list(strides_attr.as_ints()) if strides_attr is not None else [1] * n_spatial

    pads_attr = node.attributes.get("pads")
    pads = list(pads_attr.as_ints()) if pads_attr is not None else [0] * (2 * n_spatial)

    ceil_mode_attr = node.attributes.get("ceil_mode")
    ceil_mode = int(ceil_mode_attr.as_int()) if ceil_mode_attr is not None else 0

    dilations_attr = node.attributes.get("dilations")
    dilations = (
        list(dilations_attr.as_ints()) if dilations_attr is not None else [1] * n_spatial
    )

    spatial_dims = _compute_pool_output_shape(
        ctx, node, x_shape, kernel_shape, strides, pads, dilations, ceil_mode, auto_pad
    )

    output_dims: list[int | ir.SymbolicDim] = [x_shape[0], x_shape[1], *spatial_dims]
    ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)


@_reg("", "MaxPool", since_version=12)
def infer_max_pool(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for MaxPool operator."""
    (x,) = _context.check_inputs(node, "X")

    output_dtype = x.dtype
    x_shape = x.shape

    if x_shape is None:
        ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        if len(node.outputs) > 1:
            ctx.set_shape_and_dtype(node.outputs[1], None, ir.DataType.INT64)
        return

    n_spatial = x_shape.rank() - 2

    kernel_shape = list(_context.require_attr(node, "kernel_shape").as_ints())

    auto_pad_attr = node.attributes.get("auto_pad")
    auto_pad = auto_pad_attr.as_string() if auto_pad_attr is not None else "NOTSET"

    strides_attr = node.attributes.get("strides")
    strides = list(strides_attr.as_ints()) if strides_attr is not None else [1] * n_spatial

    pads_attr = node.attributes.get("pads")
    pads = list(pads_attr.as_ints()) if pads_attr is not None else [0] * (2 * n_spatial)

    ceil_mode_attr = node.attributes.get("ceil_mode")
    ceil_mode = int(ceil_mode_attr.as_int()) if ceil_mode_attr is not None else 0

    dilations_attr = node.attributes.get("dilations")
    dilations = (
        list(dilations_attr.as_ints()) if dilations_attr is not None else [1] * n_spatial
    )

    spatial_dims = _compute_pool_output_shape(
        ctx, node, x_shape, kernel_shape, strides, pads, dilations, ceil_mode, auto_pad
    )

    output_dims: list[int | ir.SymbolicDim] = [x_shape[0], x_shape[1], *spatial_dims]
    output_shape = ir.Shape(output_dims)
    ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], output_shape, ir.DataType.INT64)


@_reg("", "LpPool", since_version=18)
def infer_lp_pool(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for LpPool operator."""
    (x,) = _context.check_inputs(node, "X")

    output_dtype = x.dtype
    x_shape = x.shape

    if x_shape is None:
        ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    n_spatial = x_shape.rank() - 2

    kernel_shape_attr = node.attributes.get("kernel_shape")
    if kernel_shape_attr is None:
        raise _context.OpUsageError(node, "kernel_shape attribute is required")
    kernel_shape = list(kernel_shape_attr.as_ints())

    auto_pad_attr = node.attributes.get("auto_pad")
    auto_pad = auto_pad_attr.as_string() if auto_pad_attr is not None else "NOTSET"

    strides_attr = node.attributes.get("strides")
    strides = list(strides_attr.as_ints()) if strides_attr is not None else [1] * n_spatial

    pads_attr = node.attributes.get("pads")
    pads = list(pads_attr.as_ints()) if pads_attr is not None else [0] * (2 * n_spatial)

    ceil_mode_attr = node.attributes.get("ceil_mode")
    ceil_mode = int(ceil_mode_attr.as_int()) if ceil_mode_attr is not None else 0

    dilations_attr = node.attributes.get("dilations")
    dilations = (
        list(dilations_attr.as_ints()) if dilations_attr is not None else [1] * n_spatial
    )

    spatial_dims = _compute_pool_output_shape(
        ctx, node, x_shape, kernel_shape, strides, pads, dilations, ceil_mode, auto_pad
    )

    output_dims: list[int | ir.SymbolicDim] = [x_shape[0], x_shape[1], *spatial_dims]
    ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)


@_reg("", "GlobalLpPool", since_version=2)
@_reg("", "GlobalMaxPool", since_version=1)
@_reg("", "GlobalAveragePool", since_version=1)
def infer_global_pool(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Global*Pool operators."""
    (x,) = _context.check_inputs(node, "X")

    output_dtype = x.dtype
    x_shape = x.shape

    if x_shape is None:
        ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    n_spatial = x_shape.rank() - 2
    output_dims: list[int | ir.SymbolicDim] = [x_shape[0], x_shape[1]] + [1] * n_spatial
    ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)
