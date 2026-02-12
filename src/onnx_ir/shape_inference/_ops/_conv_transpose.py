# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ConvTranspose operator."""

from __future__ import annotations

__all__ = [
    "infer_conv_transpose",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "ConvTranspose", since_version=11)
def infer_conv_transpose(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for ConvTranspose operator.

    Spec: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
    """
    (x, w) = _context.check_inputs(node, "X", "W")

    output_dtype = x.dtype
    x_shape = x.shape
    w_shape = w.shape

    auto_pad_attr = node.attributes.get("auto_pad")
    auto_pad = auto_pad_attr.as_string() if auto_pad_attr is not None else "NOTSET"

    if x_shape is None or w_shape is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    x_rank = x_shape.rank()
    if x_rank < 3:
        ctx.record_error(node, f"ConvTranspose input must be at least rank 3, got {x_rank}")
        return

    n_spatial = x_rank - 2

    group_attr = node.attributes.get("group")
    group = group_attr.as_int() if group_attr is not None else 1

    batch_dim = x_shape[0]
    # W shape is [C_in, C_out/group, *kernel_shape]
    out_channels_per_group = w_shape[1]
    out_channels: int | ir.SymbolicDim = out_channels_per_group * group

    # Check for output_shape attribute
    output_shape_attr = node.attributes.get("output_shape")
    if output_shape_attr is not None:
        spatial_dims: list[int | ir.SymbolicDim] = list(output_shape_attr.as_ints())
        output_dims: list[int | ir.SymbolicDim] = [batch_dim, out_channels, *spatial_dims]
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)
        return

    strides_attr = node.attributes.get("strides")
    strides = list(strides_attr.as_ints()) if strides_attr is not None else [1] * n_spatial

    # For SAME_UPPER/SAME_LOWER, output = input * stride
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        spatial_out: list[int | ir.SymbolicDim] = []
        for i in range(n_spatial):
            in_dim = x_shape[i + 2]
            s = strides[i]
            spatial_out.append(in_dim * s)
        output_dims = [batch_dim, out_channels, *spatial_out]
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)
        return

    # Read kernel_shape from attribute or weight tensor
    kernel_shape_attr = node.attributes.get("kernel_shape")
    if kernel_shape_attr is not None:
        kernel_shape: list[int | None] = list(kernel_shape_attr.as_ints())
    elif w_shape.rank() > 2:
        kernel_shape = [
            w_shape[i + 2] if isinstance(w_shape[i + 2], int) else None  # type: ignore[misc]
            for i in range(n_spatial)
        ]
    else:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    strides_attr = node.attributes.get("strides")
    strides = list(strides_attr.as_ints()) if strides_attr is not None else [1] * n_spatial

    dilations_attr = node.attributes.get("dilations")
    dilations = (
        list(dilations_attr.as_ints()) if dilations_attr is not None else [1] * n_spatial
    )

    pads_attr = node.attributes.get("pads")
    pads = list(pads_attr.as_ints()) if pads_attr is not None else [0] * (2 * n_spatial)

    output_padding_attr = node.attributes.get("output_padding")
    output_padding = (
        list(output_padding_attr.as_ints())
        if output_padding_attr is not None
        else [0] * n_spatial
    )

    spatial_out: list[int | ir.SymbolicDim] = []
    for i in range(n_spatial):
        in_dim = x_shape[i + 2]
        k = kernel_shape[i]
        s = strides[i]
        d = dilations[i]

        if k is None:
            spatial_out.append(ctx.new_symbolic_dim())
            continue

        out_dim = (
            s * (in_dim - 1)
            + output_padding[i]
            + ((k - 1) * d + 1)
            - pads[i]
            - pads[i + n_spatial]
        )
        spatial_out.append(out_dim)

    output_dims = [batch_dim, out_channels, *spatial_out]
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)
