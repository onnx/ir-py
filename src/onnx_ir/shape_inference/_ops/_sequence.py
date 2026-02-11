# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Sequence operators."""

from __future__ import annotations

__all__ = [
    "infer_concat_from_sequence",
    "infer_sequence_at",
    "infer_sequence_construct",
    "infer_sequence_empty",
    "infer_sequence_erase",
    "infer_sequence_insert",
    "infer_sequence_length",
    "infer_split_to_sequence",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


def _get_sequence_elem_type(value: ir.Value) -> ir.TypeProtocol | None:
    """Extract the element type from a sequence-typed value."""
    if value.type is None:
        return None
    if isinstance(value.type, ir.SequenceType):
        return value.type.elem_type
    return None


@_reg("", "SequenceConstruct", since_version=11)
def infer_sequence_construct(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for SequenceConstruct operator.

    Output is a sequence whose element type matches the input tensors' type.

    Spec: https://onnx.ai/onnx/operators/onnx__SequenceConstruct.html
    """
    (first,) = _context.check_inputs(node, "inputs[0]")

    elem_type = first.type if first.type is not None else ir.TensorType(first.dtype)  # type: ignore[arg-type]
    if elem_type is None:
        return

    if len(node.outputs) > 0:
        ctx.set_type(node.outputs[0], ir.SequenceType(elem_type))


@_reg("", "SequenceEmpty", since_version=11)
def infer_sequence_empty(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for SequenceEmpty operator.

    Output is an empty sequence. Element dtype comes from the ``dtype`` attribute.

    Spec: https://onnx.ai/onnx/operators/onnx__SequenceEmpty.html
    """
    dtype_attr = node.attributes.get("dtype")
    # Default dtype is FLOAT per spec
    dtype = ir.DataType(dtype_attr.as_int()) if dtype_attr is not None else ir.DataType.FLOAT

    if len(node.outputs) > 0:
        ctx.set_type(node.outputs[0], ir.SequenceType(ir.TensorType(dtype)))


@_reg("", "SequenceAt", since_version=11)
def infer_sequence_at(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for SequenceAt operator.

    Output is a tensor extracted from the input sequence.

    Spec: https://onnx.ai/onnx/operators/onnx__SequenceAt.html
    """
    (seq, _position) = _context.check_inputs(node, "input_sequence", "position")

    elem_type = _get_sequence_elem_type(seq)
    if elem_type is not None and len(node.outputs) > 0:
        ctx.set_type(node.outputs[0], elem_type)


@_reg("", "SequenceLength", since_version=11)
def infer_sequence_length(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for SequenceLength operator.

    Output is a scalar INT64 tensor.

    Spec: https://onnx.ai/onnx/operators/onnx__SequenceLength.html
    """
    _context.check_inputs(node, "input_sequence")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([]), ir.DataType.INT64)


@_reg("", "SequenceInsert", since_version=11)
def infer_sequence_insert(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for SequenceInsert operator.

    Output sequence has the same element type as the input sequence.

    Spec: https://onnx.ai/onnx/operators/onnx__SequenceInsert.html
    """
    (seq, _tensor) = _context.check_inputs(node, "input_sequence", "tensor")

    if seq.type is not None and len(node.outputs) > 0:
        ctx.set_type(node.outputs[0], seq.type)


@_reg("", "SequenceErase", since_version=11)
def infer_sequence_erase(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for SequenceErase operator.

    Output sequence has the same element type as the input sequence.

    Spec: https://onnx.ai/onnx/operators/onnx__SequenceErase.html
    """
    (seq,) = _context.check_inputs(node, "input_sequence")

    if seq.type is not None and len(node.outputs) > 0:
        ctx.set_type(node.outputs[0], seq.type)


@_reg("", "SplitToSequence", since_version=11)
def infer_split_to_sequence(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for SplitToSequence operator.

    Output is a sequence of tensors with the same dtype as the input tensor.

    Spec: https://onnx.ai/onnx/operators/onnx__SplitToSequence.html
    """
    (data,) = _context.check_inputs(node, "input")

    elem_type = data.type if data.type is not None else ir.TensorType(data.dtype)  # type: ignore[arg-type]
    if elem_type is None:
        return

    if len(node.outputs) > 0:
        ctx.set_type(node.outputs[0], ir.SequenceType(elem_type))


@_reg("", "ConcatFromSequence", since_version=11)
def infer_concat_from_sequence(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer type for ConcatFromSequence operator.

    Output is a tensor with the same dtype as the sequence elements.

    Spec: https://onnx.ai/onnx/operators/onnx__ConcatFromSequence.html
    """
    (seq,) = _context.check_inputs(node, "input_sequence")

    elem_type = _get_sequence_elem_type(seq)
    if elem_type is not None and len(node.outputs) > 0:
        # Output is a tensor, not a sequence — use element dtype
        dtype = elem_type.dtype if hasattr(elem_type, "dtype") else None
        ctx.set_shape_and_dtype(node.outputs[0], None, dtype)
