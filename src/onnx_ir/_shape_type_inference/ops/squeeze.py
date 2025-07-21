"""Squeeze operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common

logger = logging.getLogger(__name__)


def _compute_output_shape_no_axes(input_shape: ir.Shape) -> ir.Shape:
    """Compute output shape when no axes are specified."""
    output_dims = []
    for dim in input_shape:
        # For symbolic dimensions, we assume they are not 1
        # Only squeeze literal 1s
        if isinstance(dim, int):
            if dim == 1:
                continue  # Skip dimension of size 1
            else:
                output_dims.append(dim)
        else:
            logger.warning(
                "Squeeze operation has symbolic dimension %s, assuming it is not 1.", dim
            )
            output_dims.append(dim)
    return ir.Shape(output_dims)


def _normalize_axes(axes: Sequence[int], rank: int) -> set[int]:
    """Normalize axes to be within the valid range for the given rank."""
    normalized_axes = set()
    for axis in axes:
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(f"Squeeze axis {axis} is out of bounds for rank {rank}.")
        normalized_axes.add(axis)
    return normalized_axes


def _compute_output_shape_with_axes(input_shape: ir.Shape, axes: set[int]) -> ir.Shape:
    """Compute output shape when axes are specified."""
    output_dims = [dim for i, dim in enumerate(input_shape) if i not in axes]
    return ir.Shape(output_dims)


class Squeeze12Inferrer(_common.NodeInferrer):
    """Inferrer for Squeeze-12 and lower.

    We assume that axes doesn't have duplicates.
    """

    def __init__(self) -> None:
        """Initialize the Squeeze inferrer."""
        super().__init__("Squeeze", opsets=range(13))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Squeeze operations."""
        input = node.inputs[0]
        assert input is not None
        input_shape = input.shape
        if input_shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="Squeeze input shape is not known."
            )

        rank = len(input_shape)

        # Get axes to squeeze
        axes = node.attributes.get_ints("axes")

        if axes is None:
            output_shape = _compute_output_shape_no_axes(input_shape)
        else:
            try:
                axes = _normalize_axes(axes, rank)
            except ValueError as e:
                return _common.InferenceResult(status="invalid_node", msg=str(e))
            output_shape = _compute_output_shape_with_axes(input_shape, axes)
        return _common.InferenceResult(values=(ir.Value(shape=output_shape, type=input.type),))


class Squeeze13Inferrer(_common.NodeInferrer):
    """Inferrer for Squeeze-13 and higher.

    We assume that axes doesn't have duplicates.
    """

    def __init__(self) -> None:
        """Initialize the Squeeze inferrer."""
        super().__init__("Squeeze", opsets=range(14, _common.MAX_SUPPORTED_OPSET))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Squeeze operations."""
        assert node.inputs[0] is not None
        assert node.inputs[1] is not None

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(
                status="missing_info", msg="Squeeze input shape is not known."
            )

        rank = len(input_shape)

        axes_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        if axes_tensor is not None:
            try:
                axes = _normalize_axes(axes_tensor.numpy().tolist(), rank)
            except ValueError as e:
                return _common.InferenceResult(status="invalid_node", msg=str(e))
            output_shape = _compute_output_shape_with_axes(input_shape, axes)
        else:
            axes_shape = node.inputs[1].shape
            if axes_shape is None or axes_shape.is_dynamic():
                return _common.InferenceResult(
                    status="missing_info",
                    msg="Squeeze axes input shape is not known or is dynamic",
                )
            removed_axes_count = axes_shape[0]
            assert isinstance(removed_axes_count, int)
            output_shape = ir.Shape([None] * (rank - removed_axes_count))
        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=node.inputs[0].type),)
        )
