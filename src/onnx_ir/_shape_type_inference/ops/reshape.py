"""Reshape operation inferrer for ONNX IR nodes."""

import sympy

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ReshapeInferrer(_common.NodeInferrer):
    """Inferrer for Reshape operations."""

    def __init__(self) -> None:
        super().__init__(
            "Reshape", opsets=_common.inclusive_range(_common.MAX_SUPPORTED_OPSET)
        )

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Reshape operations."""
        if len(node.inputs) != 2:
            return _common.InferenceResult(
                status="invalid_node",
                msg=f"Reshape operation must have exactly two inputs, got {len(node.inputs)}.",
            )
        if node.inputs[0] is None or node.inputs[1] is None:
            return _common.InferenceResult(
                status="missing_info", msg="Reshape operation inputs cannot be None."
            )
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                status="invalid_node",
                msg=f"Reshape operation must have exactly one output, got {len(node.outputs)}.",
            )

        input_shape = node.inputs[0].shape
        shape_input = node.inputs[1]

        if input_shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="Reshape input shape cannot be None."
            )

        # Try to get the shape values from the second input
        # For symbolic inference, we may not have concrete values
        shape = ir.convenience.get_const_tensor(shape_input)
        if shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="Reshape shape input is not known."
            )

        shape_values = shape.numpy().tolist()

        # Calculate total elements in input
        total_elements = sympy.Integer(1)
        for dim in range(input_shape.rank()):
            total_elements *= _common.get_expr(input_shape, dim)

        # Process shape values
        output_dims = []
        deferred_dim_idx = -1
        non_deferred_size = sympy.Integer(1)

        for i, dim_value in enumerate(shape_values):
            if dim_value == -1:
                if deferred_dim_idx != -1:
                    return _common.InferenceResult(
                        status="invalid_node", msg="Reshape can have at most one -1 dimension."
                    )
                deferred_dim_idx = i
                output_dims.append(None)  # Placeholder
            elif dim_value == 0:
                # Copy from input shape
                if i >= len(input_shape):
                    return _common.InferenceResult(
                        status="invalid_node",
                        msg=f"Cannot copy dimension {i} from input shape of rank {len(input_shape)}.",
                    )
                dim_expr = _common.get_expr(input_shape, i)
                output_dims.append(dim_expr)
                non_deferred_size *= dim_expr
            else:
                output_dims.append(dim_value)
                non_deferred_size *= sympy.Integer(dim_value)

        # Calculate deferred dimension
        if deferred_dim_idx != -1:
            deferred_dim = total_elements // non_deferred_size
            output_dims[deferred_dim_idx] = deferred_dim

        # Create output shape
        return _common.InferenceResult(
            values=(ir.Value(shape=ir.Shape(output_dims), type=node.inputs[0].type),)
        )
