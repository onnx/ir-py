# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for attention operators."""

from __future__ import annotations

__all__ = [
    "infer_attention",
    "infer_rotary_embedding",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "Attention", since_version=23)
def infer_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Attention operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Attention.html

    3D: Q=[B,Sq,Hq], K=[B,Sk,Hk], V=[B,Sk,Hv] -> Y=[B,Sq,Hq]
        (Hq = q_num_heads * head_size, output preserves Q's hidden dim)
    4D: Q=[B,Nh,Sq,Hd], K=[B,Nkv,Sk,Hd], V=[B,Nkv,Sk,Hdv] -> Y=[B,Nh,Sq,Hdv]
    present_key/value: if past is provided, present is past shape with seq extended
    """
    (q, k, v) = _context.check_inputs(node, "Q", "K", "V")

    output_dtype = q.dtype

    # Output[0]: Y shape
    # 3D: Y=[B, Sq, q_num_heads * v_head_size]
    # 4D: Y=[B, q_num_heads, Sq, v_head_size]
    if len(node.outputs) > 0:
        output_shape: ir.Shape | None = None
        if q.shape is not None:
            rank = q.shape.rank()
            if rank == 3 and v.shape is not None:
                # Output hidden = q_num_heads * v_head_size
                q_nh_attr = node.attributes.get("q_num_heads")
                kv_nh_attr = node.attributes.get("kv_num_heads")
                q_nh = q_nh_attr.as_int() if q_nh_attr is not None else None
                kv_nh = kv_nh_attr.as_int() if kv_nh_attr is not None else None
                v_hidden = v.shape[2]
                if q_nh is not None and kv_nh is not None and isinstance(v_hidden, int):
                    v_head_size = v_hidden // kv_nh
                    out_hidden: int | ir.SymbolicDim = q_nh * v_head_size
                else:
                    out_hidden = v_hidden
                output_shape = ir.Shape([q.shape[0], q.shape[1], out_hidden])
            elif rank == 4 and v.shape is not None:
                output_shape = ir.Shape([q.shape[0], q.shape[1], q.shape[2], v.shape[3]])
            else:
                output_shape = q.shape
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # present_key and present_value
    past_key = node.inputs[4] if len(node.inputs) > 4 and node.inputs[4] is not None else None
    past_value = (
        node.inputs[5] if len(node.inputs) > 5 and node.inputs[5] is not None else None
    )

    if len(node.outputs) > 1 and node.outputs[1] is not None:
        pk_shape = _attention_present_shape(k, past_key)
        ctx.set_shape_and_dtype(node.outputs[1], pk_shape, output_dtype)

    if len(node.outputs) > 2 and node.outputs[2] is not None:
        pv_shape = _attention_present_shape(v, past_value)
        ctx.set_shape_and_dtype(node.outputs[2], pv_shape, output_dtype)

    # Output[3]: qk_matmul_output = [B, q_num_heads, Sq, total_seq_length]
    if len(node.outputs) > 3 and node.outputs[3] is not None:
        qk_shape: ir.Shape | None = None
        if q.shape is not None:
            rank = q.shape.rank()
            batch = q.shape[0]

            # Get q_num_heads: from attribute (3D) or Q shape dim 1 (4D)
            qk_nh_attr = node.attributes.get("q_num_heads")
            qk_q_nh: int | ir.SymbolicDim | None = None
            if qk_nh_attr is not None:
                qk_q_nh = qk_nh_attr.as_int()
            elif rank == 4:
                qk_q_nh = q.shape[1]

            sq = q.shape[1] if rank == 3 else q.shape[2]

            # total_seq_length: past_seq + current_seq (from K)
            total_seq: int | ir.SymbolicDim = ctx.new_symbolic_dim()
            pk_shape = _attention_present_shape(k, past_key)
            if pk_shape is not None:
                total_seq = pk_shape[pk_shape.rank() - 2]
            elif k.shape is not None:
                k_seq = k.shape[k.shape.rank() - 2]
                total_seq = k_seq

            if qk_q_nh is not None:
                qk_shape = ir.Shape([batch, qk_q_nh, sq, total_seq])
        ctx.set_shape_and_dtype(node.outputs[3], qk_shape, output_dtype)


def _attention_present_shape(current: ir.Value, past: ir.Value | None) -> ir.Shape | None:
    """Compute present_key/present_value shape.

    When past is provided, present matches past shape with seq dim extended.
    When past is None, present = current shape.
    """
    if past is not None and past.shape is not None:
        past_rank = past.shape.rank()
        seq_idx = past_rank - 2
        past_seq = past.shape[seq_idx]
        if current.shape is not None:
            cur_rank = current.shape.rank()
            cur_seq = current.shape[cur_rank - 2]
        else:
            return None
        if isinstance(past_seq, int) and isinstance(cur_seq, int):
            total_seq: int | ir.SymbolicDim = past_seq + cur_seq
        else:
            total_seq = past_seq + cur_seq  # type: ignore[assignment]
        dims = list(past.shape.dims)
        dims[seq_idx] = total_seq
        return ir.Shape(dims)
    if current.shape is None:
        return None
    return current.shape


@_reg("", "RotaryEmbedding", since_version=23)
def infer_rotary_embedding(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RotaryEmbedding operator.

    Spec: https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html
    """
    (input_val,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)

    # Optional second output (output_position_ids)
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        position_ids_shape: ir.Shape | None = None
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            position_ids_shape = node.inputs[1].shape
        ctx.set_shape_and_dtype(node.outputs[1], position_ids_shape, ir.DataType.INT64)
