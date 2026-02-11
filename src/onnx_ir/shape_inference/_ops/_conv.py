# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Conv operator."""

from __future__ import annotations

__all__ = [
    "infer_conv",
]

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context


@_registry.registry.register("", "Conv", since_version=1)
def infer_conv(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Conv operator.

    Spatial output dim = floor((input + 2*pad - dilation*(kernel-1) - 1) / stride + 1)

    Spec: https://onnx.ai/onnx/operators/onnx__Conv.html
    """
    if len(node.inputs) < 2:
        ctx.record_error(node, f"Expected at least 2 inputs, got {len(node.inputs)}")
        return

    x = node.inputs[0]
    w = node.inputs[1]
    if x is None or w is None:
        return

    x_shape = x.shape
    w_shape = w.shape
    output_dtype = x.dtype

    if x_shape is None or w_shape is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    x_rank = x_shape.rank()
    if x_rank < 3:
        ctx.record_error(node, f"Conv input must be at least rank 3, got {x_rank}")
        return

    n_spatial = x_rank - 2

    # Read attributes
    kernel_shape_attr = node.attributes.get("kernel_shape")
    if kernel_shape_attr is not None:
        kernel_shape = list(kernel_shape_attr.as_ints())
    elif w_shape.rank() >= 2:
        kernel_shape = [
            w_shape[i + 2] if isinstance(w_shape[i + 2], int) else None
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
    if pads_attr is not None:
        pads = list(pads_attr.as_ints())
    else:
        pads = [0] * (2 * n_spatial)

    group_attr = node.attributes.get("group")
    _group = group_attr.as_int() if group_attr is not None else 1

    # Batch dim and output channels
    batch_dim = x_shape[0]
    out_channels = w_shape[0]

    # Compute spatial output dims
    spatial_dims: list[int | ir.SymbolicDim] = []
    for i in range(n_spatial):
        in_dim = x_shape[i + 2]
        k = kernel_shape[i]
        s = strides[i]
        d = dilations[i]
        pad_begin = pads[i]
        pad_end = pads[i + n_spatial]

        if isinstance(in_dim, int) and k is not None:
            out_dim = (in_dim + pad_begin + pad_end - d * (k - 1) - 1) // s + 1
            spatial_dims.append(out_dim)
        else:
            spatial_dims.append(ir.SymbolicDim(None))

    output_dims: list[int | ir.SymbolicDim] = [batch_dim, out_channels] + spatial_dims
    output_shape = ir.Shape(output_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
