"""Constant operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ConstantInferrer(_common.NodeInferrer):
    """Inferrer for Constant operations."""

    def __init__(self) -> None:
        """Initialize the Constant inferrer."""
        super().__init__(
            "Constant", opsets=_common.inclusive_range(_common.MAX_SUPPORTED_OPSET)
        )

    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Constant operations."""
        assert node.inputs[0] is not None
        tensor = ir.convenience.get_const_tensor(node.inputs[0])
        if tensor is None:
            return _common.InferenceResult(
                status="missing_info", msg="Constant tensor cannot be obtained."
            )

        # Create shape from the tensor dimensions
        output_shape = ir.Shape(tensor.shape)

        # Get the data type from the tensor
        output_type = ir.TensorType(tensor.dtype)

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
