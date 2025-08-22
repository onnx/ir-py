"""Standard Inferrers for ONNX IR nodes."""

from __future__ import annotations

from collections.abc import Collection

import sympy

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ElementwiseInferrer(_common.NodeInferrer):
    """Base class for elementwise operation inferrers."""

    def __init__(self, op_type: str, opsets: Collection[int] | None = None) -> None:
        """Initialize the elementwise inferrer with the operation type."""
        if opsets is None:
            opsets = _common.inclusive_range(_common.MAX_SUPPORTED_OPSET)
        super().__init__(op_type, opsets=opsets)

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for elementwise operations."""
        assert node.inputs[0] is not None
        return _common.InferenceResult(
            (ir.Value(shape=node.inputs[0].shape, type=node.inputs[0].type),)
        )


def broadcast_shapes_bidirectional(shape1: ir.Shape, shape2: ir.Shape) -> ir.Shape:
    """Broadcast two shapes bidirectionally.

    Args:
        shape1: The first shape to broadcast.
        shape2: The second shape to broadcast.

    Returns:
        A new shape that is the result of broadcasting both shapes.
    """
    rank1 = len(shape1)
    rank2 = len(shape2)
    new_rank = max(rank1, rank2)
    new_dims = []

    for i in range(new_rank):
        dim1_idx = rank1 - 1 - i
        dim2_idx = rank2 - 1 - i

        # Get expressions for dimensions
        dim1_expr = _common.get_expr(shape1, dim1_idx) if i < rank1 else sympy.Integer(1)
        dim2_expr = _common.get_expr(shape2, dim2_idx) if i < rank2 else sympy.Integer(1)

        # Broadcasting rules
        if dim1_expr == 1:
            new_dim_expr = dim2_expr
        elif dim2_expr == 1:
            new_dim_expr = dim1_expr
        elif dim1_expr == dim2_expr:
            new_dim_expr = dim1_expr
        else:
            # Incompatible dimensions - this should be caught at runtime
            # For symbolic inference, we assume they can be broadcast
            new_dim_expr = sympy.Max(dim1_expr, dim2_expr)

        # Add to the front to maintain right-to-left processing order
        new_dims.insert(0, new_dim_expr)

    # Create new shape directly
    return ir.Shape(new_dims)


class BinaryInferrer(_common.NodeInferrer):
    """Base class for binary operation inferrers."""

    def __init__(self, op_type: str) -> None:
        """Initialize the binary inferrer with the operation type."""
        super().__init__(op_type, opsets=_common.inclusive_range(_common.MAX_SUPPORTED_OPSET))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for binary operations."""
        assert node.inputs[0] is not None
        assert node.inputs[1] is not None
        first_type = node.inputs[0].type
        second_type = node.inputs[1].type
        if first_type is not None and second_type is not None and first_type != second_type:
            return _common.InferenceResult(
                status="invalid_node",
                msg=f"Input types do not match: {first_type} vs {second_type}.",
            )

        # Broadcast the input shapes
        first_shape = node.inputs[0].shape
        second_shape = node.inputs[1].shape
        if first_shape is None or second_shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="Input shapes cannot be None."
            )

        output_shape = broadcast_shapes_bidirectional(first_shape, second_shape)
        output_type = first_type if first_type is not None else second_type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
