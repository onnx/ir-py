# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for recurrent operators (RNN, GRU, LSTM)."""

from __future__ import annotations

__all__ = [
    "infer_lstm",
    "infer_rnn_gru",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


def _infer_rnn(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    num_gates: int,
) -> None:
    """Shared shape inference logic for RNN, GRU, and LSTM."""
    (x,) = _context.check_inputs(node, "X")

    direction_attr = node.attributes.get("direction")
    direction = direction_attr.as_string() if direction_attr is not None else "forward"
    num_directions = 2 if direction == "bidirectional" else 1

    layout_attr = node.attributes.get("layout")
    layout = layout_attr.as_int() if layout_attr is not None else 0

    # Try to get hidden_size from attribute
    hidden_size_attr = node.attributes.get("hidden_size")
    hidden_size: int | ir.SymbolicDim | None = None
    if hidden_size_attr is not None:
        hidden_size = hidden_size_attr.as_int()
    else:
        # Try to infer from W input (input 1): [num_directions, hidden_size * num_gates, input_size]
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            w = node.inputs[1]
            if w.shape is not None and w.shape.rank() >= 2:
                w_dim1 = w.shape.dims[1]
                hidden_size = w_dim1 // num_gates

    if hidden_size is None:
        hidden_size = ctx.new_symbolic_dim()

    output_dtype = x.dtype

    if x.shape is None:
        # Can only set dtype on outputs
        for output in node.outputs:
            if output is not None:
                ctx.set_shape_and_dtype(output, None, output_dtype)
        return

    if layout == 0:
        seq_length = x.shape.dims[0]
        batch_size = x.shape.dims[1]
    else:
        batch_size = x.shape.dims[0]
        seq_length = x.shape.dims[1]

    # Y: all hidden states
    if len(node.outputs) > 0 and node.outputs[0] is not None:
        if layout == 0:
            y_shape = ir.Shape([seq_length, num_directions, batch_size, hidden_size])
        else:
            y_shape = ir.Shape([batch_size, seq_length, num_directions, hidden_size])
        ctx.set_shape_and_dtype(node.outputs[0], y_shape, output_dtype)

    # Y_h: last hidden state
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        if layout == 0:
            y_h_shape = ir.Shape([num_directions, batch_size, hidden_size])
        else:
            y_h_shape = ir.Shape([batch_size, num_directions, hidden_size])
        ctx.set_shape_and_dtype(node.outputs[1], y_h_shape, output_dtype)

    # Y_c: last cell state (LSTM only, output index 2)
    if num_gates == 4 and len(node.outputs) > 2 and node.outputs[2] is not None:
        if layout == 0:
            y_c_shape = ir.Shape([num_directions, batch_size, hidden_size])
        else:
            y_c_shape = ir.Shape([batch_size, num_directions, hidden_size])
        ctx.set_shape_and_dtype(node.outputs[2], y_c_shape, output_dtype)


@_reg("", "GRU", since_version=7)
@_reg("", "RNN", since_version=7)
def infer_rnn_gru(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RNN/GRU operators."""
    num_gates = 3 if node.op_type == "GRU" else 1
    _infer_rnn(ctx, node, num_gates)


@_reg("", "LSTM", since_version=7)
def infer_lstm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for LSTM operator."""
    _infer_rnn(ctx, node, num_gates=4)
