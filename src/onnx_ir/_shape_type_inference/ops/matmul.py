"""MatMul operation inferrer for ONNX IR nodes."""

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common
from onnx_ir._shape_type_inference.ops.standard_ops import broadcast_shapes_bidirectional


class MatMulInferrer(_common.NodeInferrer):
    """Inferrer for MatMul operations."""

    def __init__(self) -> None:
        """Initialize the MatMul inferrer."""
        super().__init__("MatMul", opsets=_common.inclusive_range(_common.MAX_SUPPORTED_OPSET))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for MatMul operations."""
        assert node.inputs[0] is not None and node.inputs[1] is not None

        lhs_shape = node.inputs[0].shape
        rhs_shape = node.inputs[1].shape
        if lhs_shape is None or rhs_shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="MatMul input shapes cannot be None."
            )

        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)

        if lhs_rank == 0 or rhs_rank == 0:
            return _common.InferenceResult(
                status="invalid_node", msg="MatMul inputs cannot be scalars."
            )

        # Compute output shape based on matrix multiplication rules
        if lhs_rank == 1 and rhs_rank == 1:
            # Vector dot product: (n,) x (n,) -> scalar
            output_shape = ir.Shape([])
        elif lhs_rank == 1:
            # Matrix-vector: (n,) x (..., n, k) -> (..., k)
            output_dims = [*rhs_shape[:-2], rhs_shape[-1]]
            output_shape = ir.Shape(output_dims)
        elif rhs_rank == 1:
            # Vector-matrix: (..., m, n) x (n,) -> (..., m)
            output_shape = ir.Shape(lhs_shape[:-1])
        else:
            # Matrix-matrix: (..., m, n) x (..., n, k) -> (..., m, k)
            # Broadcast batch dimensions
            lhs_batch = lhs_shape[:-2]
            rhs_batch = rhs_shape[:-2]
            if lhs_batch and rhs_batch:
                # TODO(justinchuby): Ensure this is correct
                batch_shape = broadcast_shapes_bidirectional(
                    ir.Shape(lhs_batch), ir.Shape(rhs_batch)
                )
                output_dims = [*batch_shape, lhs_shape[-2], rhs_shape[-1]]
                output_shape = ir.Shape(output_dims)
            elif lhs_batch:
                output_dims = [*lhs_batch, lhs_shape[-2], rhs_shape[-1]]
                output_shape = ir.Shape(output_dims)
            elif rhs_batch:
                output_dims = [*rhs_batch, lhs_shape[-2], rhs_shape[-1]]
                output_shape = ir.Shape(output_dims)
            else:
                output_dims = [lhs_shape[-2], rhs_shape[-1]]
                output_shape = ir.Shape(output_dims)

        output_type = node.inputs[0].type
        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
