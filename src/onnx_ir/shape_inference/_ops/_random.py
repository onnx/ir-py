# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for random generation and *Like operators."""

from __future__ import annotations

__all__ = [
    "infer_bernoulli",
    "infer_blackman_window",
    "infer_eye_like",
    "infer_hamming_window",
    "infer_hann_window",
    "infer_multinomial",
    "infer_random_normal",
    "infer_random_normal_like",
    "infer_random_uniform",
    "infer_random_uniform_like",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


def _infer_like(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Shared inference for *Like ops (EyeLike, RandomNormalLike, etc.).

    Output has the same shape as the input. dtype comes from the optional
    ``dtype`` attribute; when absent the input dtype is used.
    """
    (input_val,) = _context.check_inputs(node, "input")

    dtype_attr = node.attributes.get("dtype")
    if dtype_attr is not None:
        output_dtype: ir.DataType | None = ir.DataType(dtype_attr.as_int())
    else:
        output_dtype = input_val.dtype

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, output_dtype)


def _infer_random_from_shape(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Shared inference for RandomNormal / RandomUniform.

    Shape and dtype are read entirely from attributes (no tensor inputs).
    """
    shape_attr = _context.require_attr(node, "shape")
    output_dims = [int(d) for d in shape_attr.as_ints()]
    output_shape = ir.Shape(output_dims)

    dtype_attr = node.attributes.get("dtype")
    output_dtype = (
        ir.DataType(dtype_attr.as_int()) if dtype_attr is not None else ir.DataType.FLOAT
    )

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


def _infer_window(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Shared inference for BlackmanWindow, HammingWindow, HannWindow.

    Output is a 1-D tensor whose length equals the scalar ``size`` input.
    """
    (size,) = _context.check_inputs(node, "size")

    dtype_attr = node.attributes.get("output_datatype")
    output_dtype = (
        ir.DataType(dtype_attr.as_int()) if dtype_attr is not None else ir.DataType.FLOAT
    )

    size_const = ir.convenience.get_const_tensor(size)
    if size_const is not None:
        size_val = int(size_const.numpy().item())
        output_shape: ir.Shape = ir.Shape([size_val])
    else:
        output_shape = ir.Shape([ctx.new_symbolic_dim()])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


# -- *Like ops ---------------------------------------------------------------


@_reg("", "EyeLike", since_version=9)
def infer_eye_like(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for EyeLike operator."""
    _infer_like(ctx, node)


@_reg("", "RandomNormalLike", since_version=1)
def infer_random_normal_like(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RandomNormalLike operator."""
    _infer_like(ctx, node)


@_reg("", "RandomUniformLike", since_version=1)
def infer_random_uniform_like(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RandomUniformLike operator."""
    _infer_like(ctx, node)


@_reg("", "Bernoulli", since_version=15)
def infer_bernoulli(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Bernoulli operator."""
    _infer_like(ctx, node)


# -- Random from shape attrs -------------------------------------------------


@_reg("", "RandomNormal", since_version=1)
def infer_random_normal(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RandomNormal operator."""
    _infer_random_from_shape(ctx, node)


@_reg("", "RandomUniform", since_version=1)
def infer_random_uniform(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RandomUniform operator."""
    _infer_random_from_shape(ctx, node)


# -- Multinomial --------------------------------------------------------------


@_reg("", "Multinomial", since_version=7)
def infer_multinomial(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Multinomial operator.

    Input shape: [batch_size, class_size]
    Output shape: [batch_size, sample_size]
    """
    (input_val,) = _context.check_inputs(node, "input")

    sample_size_attr = node.attributes.get("sample_size")
    sample_size: int = sample_size_attr.as_int() if sample_size_attr is not None else 1

    dtype_attr = node.attributes.get("dtype")
    output_dtype = (
        ir.DataType(dtype_attr.as_int()) if dtype_attr is not None else ir.DataType.INT32
    )

    output_shape: ir.Shape | None = None
    if input_val.shape is not None and input_val.shape.rank() == 2:
        batch_size = input_val.shape[0]
        output_shape = ir.Shape([batch_size, sample_size])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


# -- Window ops ---------------------------------------------------------------


@_reg("", "BlackmanWindow", since_version=17)
def infer_blackman_window(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for BlackmanWindow operator."""
    _infer_window(ctx, node)


@_reg("", "HammingWindow", since_version=17)
def infer_hamming_window(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for HammingWindow operator."""
    _infer_window(ctx, node)


@_reg("", "HannWindow", since_version=17)
def infer_hann_window(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for HannWindow operator."""
    _infer_window(ctx, node)
