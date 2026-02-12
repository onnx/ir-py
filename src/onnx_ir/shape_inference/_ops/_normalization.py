# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for normalization operators."""

from __future__ import annotations

__all__ = [
    "infer_batch_normalization",
    "infer_dequantize_linear",
    "infer_dynamic_quantize_linear",
    "infer_group_normalization",
    "infer_instance_normalization",
    "infer_layer_normalization",
    "infer_lrn",
    "infer_quantize_linear",
    "infer_rms_normalization",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "BatchNormalization", since_version=1)
def infer_batch_normalization(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for BatchNormalization operator."""
    (x, scale, _b, _input_mean, _input_var) = _context.check_inputs(
        node, "X", "scale", "B", "input_mean", "input_var"
    )

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)

    training_mode_attr = node.attributes.get("training_mode")
    training_mode = training_mode_attr.as_int() if training_mode_attr is not None else 0

    if training_mode and len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], scale.shape, x.dtype)
    if training_mode and len(node.outputs) > 2:
        ctx.set_shape_and_dtype(node.outputs[2], scale.shape, x.dtype)


@_reg("", "LayerNormalization", since_version=17)
def infer_layer_normalization(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for LayerNormalization operator."""
    (x, _scale) = _context.check_inputs(node, "X", "Scale")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)

    # Mean and InvStdDev outputs: shape is input[:axis] with trailing 1s
    if len(node.outputs) > 1 or len(node.outputs) > 2:
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr is not None else -1

        stash_type_attr = node.attributes.get("stash_type")
        stash_type = stash_type_attr.as_int() if stash_type_attr is not None else 1
        stash_dtype = ir.DataType(stash_type)

        reduced_shape: ir.Shape | None = None
        if x.shape is not None:
            rank = x.shape.rank()
            if axis < 0:
                axis += rank
            # Shape is input[:axis] with trailing 1s for the reduced dimensions
            reduced_dims: list[int | ir.SymbolicDim] = list(x.shape.dims[:axis]) + [1] * (
                rank - axis
            )
            reduced_shape = ir.Shape(reduced_dims)

        if len(node.outputs) > 1:
            ctx.set_shape_and_dtype(node.outputs[1], reduced_shape, stash_dtype)
        if len(node.outputs) > 2:
            ctx.set_shape_and_dtype(node.outputs[2], reduced_shape, stash_dtype)


@_reg("", "GroupNormalization", since_version=21)
def infer_group_normalization(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for GroupNormalization operator."""
    (x,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)


@_reg("", "RMSNormalization", since_version=23)
def infer_rms_normalization(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RMSNormalization operator."""
    (x,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)

    if len(node.outputs) > 1:
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr is not None else -1

        reduced_shape: ir.Shape | None = None
        if x.shape is not None:
            rank = x.shape.rank()
            if axis < 0:
                axis += rank
            reduced_dims: list[int | ir.SymbolicDim] = list(x.shape.dims[:axis]) + [1] * (
                rank - axis
            )
            reduced_shape = ir.Shape(reduced_dims)

        ctx.set_shape_and_dtype(node.outputs[1], reduced_shape, x.dtype)


@_reg("", "InstanceNormalization", since_version=1)
def infer_instance_normalization(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for InstanceNormalization operator."""
    (x,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)


@_reg("", "LRN", since_version=1)
def infer_lrn(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for LRN operator."""
    (x,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)


@_reg("", "DequantizeLinear", since_version=10)
def infer_dequantize_linear(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for DequantizeLinear operator."""
    (x,) = _context.check_inputs(node, "x")
    # Output dtype matches the scale's dtype (input[1])
    output_dtype = ir.DataType.FLOAT
    if len(node.inputs) > 1 and node.inputs[1] is not None:
        scale_dtype = node.inputs[1].dtype
        if scale_dtype is not None:
            output_dtype = scale_dtype
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, output_dtype)


@_reg("", "QuantizeLinear", since_version=10)
def infer_quantize_linear(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for QuantizeLinear operator."""
    (x,) = _context.check_inputs(node, "x")
    output_dtype = ir.DataType.UINT8
    # Check output_dtype attribute first (opset 21+)
    output_dtype_attr = node.attributes.get("output_dtype")
    if output_dtype_attr is not None:
        output_dtype = ir.DataType(output_dtype_attr.as_int())
    elif len(node.inputs) > 2 and node.inputs[2] is not None:
        zp_dtype = node.inputs[2].dtype
        if zp_dtype is not None:
            output_dtype = zp_dtype
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, output_dtype)


@_reg("", "DynamicQuantizeLinear", since_version=11)
def infer_dynamic_quantize_linear(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for DynamicQuantizeLinear operator."""
    (x,) = _context.check_inputs(node, "x")
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, ir.DataType.UINT8)
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], ir.Shape([]), ir.DataType.FLOAT)
    if len(node.outputs) > 2:
        ctx.set_shape_and_dtype(node.outputs[2], ir.Shape([]), ir.DataType.UINT8)
