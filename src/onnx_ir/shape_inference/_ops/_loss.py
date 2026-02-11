# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for loss operators."""

from __future__ import annotations

__all__ = [
    "infer_negative_log_likelihood_loss",
    "infer_softmax_cross_entropy_loss",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "NegativeLogLikelihoodLoss", since_version=13)
def infer_negative_log_likelihood_loss(
    ctx: _context.ShapeInferenceContext, node: ir.Node
) -> None:
    """Infer shape and dtype for NegativeLogLikelihoodLoss operator.

    Spec: https://onnx.ai/onnx/operators/onnx__NegativeLogLikelihoodLoss.html
    """
    (input_val, target) = _context.check_inputs(node, "input", "target")

    output_dtype = input_val.dtype

    reduction_attr = node.attributes.get("reduction")
    reduction = reduction_attr.as_string() if reduction_attr is not None else "mean"

    if reduction == "none":
        output_shape = target.shape
    else:
        output_shape = ir.Shape([])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_reg("", "SoftmaxCrossEntropyLoss", since_version=13)
def infer_softmax_cross_entropy_loss(
    ctx: _context.ShapeInferenceContext, node: ir.Node
) -> None:
    """Infer shape and dtype for SoftmaxCrossEntropyLoss operator.

    Spec: https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html
    """
    (scores, labels) = _context.check_inputs(node, "scores", "labels")

    output_dtype = scores.dtype

    reduction_attr = node.attributes.get("reduction")
    reduction = reduction_attr.as_string() if reduction_attr is not None else "mean"

    if reduction == "none":
        output_shape = labels.shape
    else:
        output_shape = ir.Shape([])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # Output[1]: log_prob (optional)
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], scores.shape, output_dtype)
