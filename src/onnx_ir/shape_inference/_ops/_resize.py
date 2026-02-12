# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Resize operator."""

from __future__ import annotations

__all__ = [
    "infer_resize",
    "infer_upsample",
]

import math

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Resize", since_version=10)
def infer_resize(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Resize operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Resize.html
    """
    (x,) = _context.check_inputs(node, "X")

    output_dtype = x.dtype
    x_shape = x.shape

    # Read axes attribute (opset 18+)
    axes_attr = node.attributes.get("axes")
    axes: list[int] | None = None
    if axes_attr is not None:
        axes = list(axes_attr.as_ints())

    # Read keep_aspect_ratio_policy
    policy_attr = node.attributes.get("keep_aspect_ratio_policy")
    policy = policy_attr.as_string() if policy_attr is not None else "stretch"

    # Check if sizes input (index 3) has const_value
    if x_shape is not None and len(node.inputs) > 3 and node.inputs[3] is not None:
        sizes_input = node.inputs[3]
        sizes_const = ir.convenience.get_const_tensor(sizes_input)
        if sizes_const is not None:
            sizes_vals = [int(s) for s in sizes_const.numpy().flatten()]
            output_dims: list[int | ir.SymbolicDim] = list(x_shape.dims)
            target_axes = axes if axes is not None else list(range(x_shape.rank()))
            for i, ax in enumerate(target_axes):
                if ax < 0:
                    ax += x_shape.rank()
                if policy in ("not_larger", "not_smaller") and isinstance(x_shape[ax], int):
                    # Compute uniform scale from all target axes
                    scales_per_axis = [
                        sizes_vals[j] / x_shape[target_axes[j]]
                        for j in range(len(target_axes))
                        if isinstance(x_shape[target_axes[j]], int)
                    ]
                    if scales_per_axis:
                        if policy == "not_larger":
                            uniform_scale = min(scales_per_axis)  # type: ignore[type-var]
                        else:
                            uniform_scale = max(scales_per_axis)  # type: ignore[type-var]
                        output_dims[ax] = math.floor(x_shape[ax] * uniform_scale)  # type: ignore[operator]
                    else:
                        output_dims[ax] = sizes_vals[i]
                else:
                    output_dims[ax] = sizes_vals[i]
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)
            return

    # Check if scales input (index 2) has const_value
    if x_shape is not None and len(node.inputs) > 2 and node.inputs[2] is not None:
        scales_input = node.inputs[2]
        scales_const = ir.convenience.get_const_tensor(scales_input)
        if scales_const is not None:
            scales = [float(s) for s in scales_const.numpy().flatten()]
            target_axes = axes if axes is not None else list(range(x_shape.rank()))
            if len(scales) == len(target_axes):
                output_dims_list: list[int | ir.SymbolicDim] = list(x_shape.dims)
                for i, ax in enumerate(target_axes):
                    if ax < 0:
                        ax += x_shape.rank()
                    d = x_shape[ax]
                    if isinstance(d, int):
                        output_dims_list[ax] = math.floor(d * scales[i])
                    else:
                        output_dims_list[ax] = ctx.new_symbolic_dim()
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


@_registry.registry.register("", "Upsample", since_version=1)
def infer_upsample(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for deprecated Upsample operator.

    Upsample(X, scales) -> Y where Y[i] = floor(X[i] * scales[i]).
    """
    (x,) = _context.check_inputs(node, "X")

    output_dtype = x.dtype
    x_shape = x.shape

    if x_shape is not None and len(node.inputs) > 1 and node.inputs[1] is not None:
        scales_const = ir.convenience.get_const_tensor(node.inputs[1])
        if scales_const is not None:
            scales = [float(s) for s in scales_const.numpy().flatten()]
            if len(scales) == x_shape.rank():
                output_dims: list[int | ir.SymbolicDim] = []
                for i in range(x_shape.rank()):
                    d = x_shape[i]
                    if isinstance(d, int):
                        output_dims.append(math.floor(d * scales[i]))
                    else:
                        output_dims.append(ctx.new_symbolic_dim())
                if len(node.outputs) > 0:
                    ctx.set_shape_and_dtype(
                        node.outputs[0], ir.Shape(output_dims), output_dtype
                    )
                return

    if x_shape is not None:
        output_dims_sym: list[int | ir.SymbolicDim] = [
            ctx.new_symbolic_dim() for _ in range(x_shape.rank())
        ]
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims_sym), output_dtype)
    else:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
