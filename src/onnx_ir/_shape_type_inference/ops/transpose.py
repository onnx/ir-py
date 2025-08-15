"""Transpose operation inferrer for ONNX IR nodes."""

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class TransposeInferrer(_common.NodeInferrer):
    """Inferrer for Transpose operations."""

    def __init__(self) -> None:
        """Initialize the Transpose inferrer."""
        super().__init__(
            "Transpose", opsets=_common.inclusive_range(_common.MAX_SUPPORTED_OPSET)
        )

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Transpose operations."""
        assert node.inputs[0] is not None
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="Transpose input shape cannot be None."
            )

        rank = len(input_shape)

        # Get permutation from attributes
        perm = node.attributes.get_ints("perm")

        # Default permutation is reversed order
        if perm is None:
            perm = list(reversed(range(rank)))

        # Validate permutation
        if len(perm) != rank:
            return _common.InferenceResult(
                status="invalid_node",
                msg=f"Permutation length {len(perm)} does not match input rank {rank}.",
            )

        if sorted(perm) != list(range(rank)):
            return _common.InferenceResult(
                status="invalid_node",
                msg=f"Invalid permutation {perm}. Must be a permutation of [0, 1, ..., {rank - 1}].",
            )

        # Apply permutation to create output shape
        output_dims = []
        for axis in perm:
            # Handle negative axis
            if axis < 0:
                axis += rank

            if axis < 0 or axis >= rank:
                return _common.InferenceResult(
                    status="invalid_node",
                    msg=f"Permutation axis {axis} is out of bounds for rank {rank}.",
                )

            # Copy dimension from input to output according to permutation
            output_dims.append(input_shape[axis])

        return _common.InferenceResult(
            values=(ir.Value(shape=ir.Shape(output_dims), type=node.inputs[0].type),)
        )
