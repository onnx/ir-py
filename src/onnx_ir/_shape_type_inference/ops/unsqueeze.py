"""Unsqueeze operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common

logger = logging.getLogger(__name__)


def _normalize_axes(axes: Sequence[int], output_rank: int) -> set[int]:
    """Normalize axes to be within the valid range for the given output rank."""
    normalized_axes = set()
    for axis in axes:
        if axis < 0:
            axis += output_rank
        if axis < 0 or axis >= output_rank:
            raise ValueError(
                f"Unsqueeze axis {axis} is out of bounds for output rank {output_rank}."
            )
        normalized_axes.add(axis)

    # Check for duplicate axes
    if len(normalized_axes) != len(axes):
        raise ValueError("Unsqueeze axes must be unique.")

    return normalized_axes


def _compute_output_shape(input_shape: ir.Shape, axes: set[int]) -> ir.Shape:
    """Compute output shape by inserting 1s at specified axes."""
    input_rank = len(input_shape)
    output_rank = input_rank + len(axes)

    output_dims = []
    input_axis = 0

    for output_axis in range(output_rank):
        if output_axis in axes:
            # Insert dimension of size 1
            output_dims.append(1)
        else:
            # Copy dimension from input
            output_dims.append(input_shape[input_axis])
            input_axis += 1

    return ir.Shape(output_dims)


class Unsqueeze12Inferrer(_common.NodeInferrer):
    """Inferrer for Unsqueeze-12 and lower.

    In these versions, axes are provided as an attribute.
    We assume that axes doesn't have duplicates.
    """

    def __init__(self) -> None:
        """Initialize the Unsqueeze inferrer."""
        super().__init__("Unsqueeze", opsets=range(13))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Unsqueeze operations."""
        input = node.inputs[0]
        assert input is not None
        input_shape = input.shape
        if input_shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="Unsqueeze input shape is not known."
            )

        input_rank = len(input_shape)

        # Get axes to unsqueeze from attributes
        axes = node.attributes.get_ints("axes")
        if axes is None:
            return _common.InferenceResult(
                status="invalid_node", msg="Unsqueeze operation requires axes attribute."
            )

        output_rank = input_rank + len(axes)

        try:
            normalized_axes = _normalize_axes(axes, output_rank)
        except ValueError as e:
            return _common.InferenceResult(status="invalid_node", msg=str(e))

        output_shape = _compute_output_shape(input_shape, normalized_axes)
        return _common.InferenceResult(values=(ir.Value(shape=output_shape, type=input.type),))


class Unsqueeze13Inferrer(_common.NodeInferrer):
    """Inferrer for Unsqueeze-13 and higher.

    In these versions, axes are provided as a second input tensor.
    We assume that axes doesn't have duplicates.
    """

    def __init__(self) -> None:
        """Initialize the Unsqueeze inferrer."""
        super().__init__("Unsqueeze", opsets=range(13, _common.MAX_SUPPORTED_OPSET))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Unsqueeze operations."""
        assert node.inputs[0] is not None
        assert node.inputs[1] is not None

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="Unsqueeze input shape is not known."
            )

        input_rank = len(input_shape)

        axes_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        if axes_tensor is not None:
            axes = axes_tensor.numpy().tolist()
            if not isinstance(axes, list):
                axes = [axes]

            output_rank = input_rank + len(axes)

            try:
                normalized_axes = _normalize_axes(axes, output_rank)
            except ValueError as e:
                return _common.InferenceResult(status="invalid_node", msg=str(e))

            output_shape = _compute_output_shape(input_shape, normalized_axes)
        else:
            # Handle case where axes tensor is not constant
            axes_shape = node.inputs[1].shape
            if axes_shape is None or axes_shape.is_dynamic():
                return _common.InferenceResult(
                    status="missing_info",
                    msg="Unsqueeze axes input shape is not known or is dynamic",
                )

            # We know the number of axes to insert but not their positions
            added_axes_count = axes_shape[0]
            assert isinstance(added_axes_count, int)
            output_rank = input_rank + added_axes_count
            # Create output shape with unknown dimensions
            output_shape = ir.Shape([None] * output_rank)

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=node.inputs[0].type),)
        )
