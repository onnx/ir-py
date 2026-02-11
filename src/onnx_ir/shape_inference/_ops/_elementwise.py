# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic shape inference for binary element-wise operators."""

from __future__ import annotations

__all__ = [
    "infer_binary_elementwise",
    "_infer_variadic_elementwise",
]


import onnx_ir as ir
from onnx_ir.shape_inference import _broadcast, _context, _registry

_reg = _registry.registry.register


def infer_binary_elementwise(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    *,
    output_dtype_override: ir.DataType | None = None,
) -> None:
    """Infer shape and dtype for a binary element-wise operator.

    Output shape is the broadcast of the two input shapes.
    Output dtype is ``output_dtype_override`` when given, otherwise the dtype of
    the first input that has one.
    """
    (input_a, input_b) = _context.check_inputs(node, "A", "B")

    output_shape = _broadcast.broadcast_shapes(input_a.shape, input_b.shape)
    output_dtype = output_dtype_override or input_a.dtype or input_b.dtype

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


# --- Arithmetic binary ops (output dtype = input dtype) ---


@_reg("", "Add", since_version=7)
@_reg("", "Sub", since_version=7)
@_reg("", "Mul", since_version=7)
@_reg("", "Div", since_version=7)
@_reg("", "Mod", since_version=10)
@_reg("", "Pow", since_version=7)
@_reg("", "BitShift", since_version=11)
@_reg("", "BitwiseAnd", since_version=18)
@_reg("", "BitwiseOr", since_version=18)
@_reg("", "BitwiseXor", since_version=18)
def _infer_arithmetic(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node)


# --- Comparison ops (output dtype = BOOL) ---


@_reg("", "Equal", since_version=7)
@_reg("", "Less", since_version=7)
@_reg("", "Greater", since_version=7)
@_reg("", "LessOrEqual", since_version=12)
@_reg("", "GreaterOrEqual", since_version=12)
def _infer_comparison(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node, output_dtype_override=ir.DataType.BOOL)


# --- Logical binary ops (output dtype = BOOL) ---


@_reg("", "And", since_version=7)
@_reg("", "Or", since_version=7)
@_reg("", "Xor", since_version=7)
def _infer_logical(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node, output_dtype_override=ir.DataType.BOOL)


# --- String ops ---


@_reg("", "StringConcat", since_version=20)
def _infer_string_concat(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node, output_dtype_override=ir.DataType.STRING)


# --- Variadic elementwise ops (output dtype = first input dtype) ---


@_reg("", "Max", since_version=8)
@_reg("", "Mean", since_version=8)
@_reg("", "Min", since_version=8)
@_reg("", "Sum", since_version=8)
def _infer_variadic_elementwise(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for variadic elementwise operators.

    Output shape is the broadcast of all inputs. Output dtype is the first input's dtype.
    """
    if len(node.inputs) < 1:
        raise _context.OpUsageError(node, "Expected at least 1 input")

    inputs = [v for v in node.inputs if v is not None]
    if not inputs:
        raise _context.OpUsageError(node, "Expected at least 1 non-None input")

    output_dtype = inputs[0].dtype
    output_shape = inputs[0].shape
    for inp in inputs[1:]:
        output_shape = _broadcast.broadcast_shapes(output_shape, inp.shape)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
