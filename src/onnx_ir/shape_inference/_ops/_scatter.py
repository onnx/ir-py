# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ScatterElements and ScatterND operators."""

from __future__ import annotations

__all__ = [
    "infer_scatter_elements",
    "infer_scatter_nd",
    "infer_tensor_scatter",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "ScatterElements", since_version=18)
def infer_scatter_elements(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for ScatterElements operator."""
    (data,) = _context.check_inputs(node, "data")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)


@_registry.registry.register("", "ScatterND", since_version=18)
def infer_scatter_nd(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for ScatterND operator."""
    (data,) = _context.check_inputs(node, "data")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)


@_registry.registry.register("", "TensorScatter", since_version=24)
def infer_tensor_scatter(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for TensorScatter operator.

    Output = data (first input) shape/dtype.
    """
    (data,) = _context.check_inputs(node, "data")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)
