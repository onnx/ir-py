# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass for converting float32 models to float16.

This module provides a pass to convert ONNX models from float32 to float16,
with options to keep input/output types as float32 and block specific operators.

Based on the float16 converter from onnxconverter-common:
https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py
"""

from __future__ import annotations

__all__ = [
    "ConvertFloatToFloat16Pass",
    "DEFAULT_OP_BLOCK_LIST",
]

import logging
import warnings
from collections.abc import Sequence

import numpy as np

import onnx_ir as ir

logger = logging.getLogger(__name__)

# Default list of operators that should not be converted to float16
# These operators are typically not numerically stable or not supported in float16
DEFAULT_OP_BLOCK_LIST = frozenset(
    [
        "ArrayFeatureExtractor",
        "Binarizer",
        "CastMap",
        "CategoryMapper",
        "DictVectorizer",
        "FeatureVectorizer",
        "Imputer",
        "LabelEncoder",
        "LinearClassifier",
        "LinearRegressor",
        "Normalizer",
        "OneHotEncoder",
        "RandomUniformLike",
        "SVMClassifier",
        "SVMRegressor",
        "Scaler",
        "TreeEnsembleClassifier",
        "TreeEnsembleRegressor",
        "ZipMap",
        "NonMaxSuppression",
        "TopK",
        "RoiAlign",
        "Resize",
        "Range",
        "CumSum",
        "Min",
        "Max",
        "Upsample",
    ]
)


def _between(a: float, b: np.ndarray, c: float) -> np.ndarray:
    return np.logical_and(a < b, b < c)


def _convert_np_to_float16(
    np_array: np.ndarray, min_positive_val: float = 1e-7, max_finite_val: float = 1e4
) -> np.ndarray:
    """Convert float32 numpy array to float16 without changing sign or finiteness.

    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.

    Args:
        np_array: The numpy array to convert.
        min_positive_val: Minimum positive value to clamp to.
        max_finite_val: Maximum finite value to clamp to.

    Returns:
        The converted float16 numpy array.
    """
    # Warn about potential truncation
    pos_values = np_array[np.where(np_array > 0)]
    if pos_values.shape[0] > 0:
        pos_max = pos_values.max()
        pos_min = pos_values.min()

        if pos_max >= max_finite_val:
            warnings.warn(
                f"the float32 number {pos_max} will be truncated to {max_finite_val}",
                stacklevel=2,
            )

        if pos_min <= min_positive_val:
            warnings.warn(
                f"the float32 number {pos_min} will be truncated to {min_positive_val}",
                stacklevel=2,
            )

    neg_values = np_array[np.where(np_array < 0)]
    if neg_values.shape[0] > 0:
        neg_max = neg_values.max()
        neg_min = neg_values.min()

        if neg_min <= -max_finite_val:
            warnings.warn(
                f"the float32 number {neg_min} will be truncated to {-max_finite_val}",
                stacklevel=2,
            )

        if neg_max >= -min_positive_val:
            warnings.warn(
                f"the float32 number {neg_max} will be truncated to {-min_positive_val}",
                stacklevel=2,
            )

    # Clamp values to the specified range
    np_array = np.where(_between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(_between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(
        _between(max_finite_val, np_array, float("inf")), max_finite_val, np_array
    )
    np_array = np.where(
        _between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array
    )
    return np.asarray(np_array, dtype=np.float16)


class ConvertFloatToFloat16Pass(ir.passes.InPlacePass):
    """Convert float32 tensors to float16.

    This pass converts all float32 tensors in the model to float16, with options to:
    - Keep input/output types as float32 (inserting Cast nodes as needed)
    - Block specific operators from being converted
    - Block specific nodes by name from being converted

    The pass handles:
    - Graph inputs and outputs
    - Initializers (constant tensors)
    - Value types
    - Tensor attributes in nodes
    - Subgraphs (e.g., in If, Loop, Scan nodes)

    Example::

        >>> import onnx_ir as ir
        >>> model = ir.load("model.onnx")
        >>> pass_ = ConvertFloatToFloat16Pass()
        >>> result = pass_(model)
        >>> if result.modified:
        ...     ir.save(result.model, "model_fp16.onnx")
    """

    def __init__(
        self,
        *,
        min_positive_val: float = 1e-7,
        max_finite_val: float = 1e4,
        keep_io_types: bool = False,
        op_block_list: Sequence[str] | None = None,
        node_block_list: Sequence[str] | None = None,
    ):
        """Initialize the ConvertFloatToFloat16Pass.

        Args:
            min_positive_val: Minimum positive value to clamp to during conversion.
            max_finite_val: Maximum finite value to clamp to during conversion.
            keep_io_types: If True, keep graph inputs/outputs as float32 and insert Cast nodes.
            op_block_list: List of operator types to not convert. If None, uses DEFAULT_OP_BLOCK_LIST.
            node_block_list: List of node names to not convert.
        """
        self.min_positive_val = min_positive_val
        self.max_finite_val = max_finite_val
        self.keep_io_types = keep_io_types
        self.op_block_list = (
            frozenset(op_block_list) if op_block_list is not None else DEFAULT_OP_BLOCK_LIST
        )
        self.node_block_list = frozenset(node_block_list) if node_block_list else frozenset()

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Convert the model from float32 to float16.

        Args:
            model: The model to convert.

        Returns:
            PassResult with the converted model and whether it was modified.
        """
        modified = False

        # Convert the main graph
        if self._convert_graph_in_place(model.graph, is_top_level=True):
            modified = True

        # Convert functions
        for function in model.functions.values():
            if self._convert_graph_in_place(function, is_top_level=False):
                modified = True

        if modified:
            logger.info("ConvertFloatToFloat16Pass modified the model")

        return ir.passes.PassResult(model, modified=modified)

    def _is_node_blocked(self, node: ir.Node) -> bool:
        """Check if a node is in the block list."""
        return (node.op_type in self.op_block_list) or (node.name in self.node_block_list)

    def _convert_graph_in_place(
        self,
        graph: ir.Graph | ir.Function,
        is_top_level: bool,
    ) -> bool:
        """Convert a graph from float32 to float16 in place.

        Args:
            graph: The graph to convert.
            is_top_level: Whether this is the top-level graph (vs a subgraph).

        Returns:
            True if the graph was modified.
        """
        modified = False

        # Convert initializers (only for Graph, not Function)
        if isinstance(graph, ir.Graph):
            for value in graph.initializers.values():
                if self._convert_tensor_value(value):
                    modified = True

        # Convert node attributes and value types
        for node in graph:
            # Convert attributes and output types only for non-blocked nodes
            if not self._is_node_blocked(node):
                if self._convert_node_attributes(node):
                    modified = True

                # Convert output value types
                for output_val in node.outputs:
                    if output_val is not None and self._convert_value_type(output_val):
                        modified = True

        # Convert input types (unless keeping IO types for top-level)
        if not (is_top_level and self.keep_io_types):
            for input_val in graph.inputs:
                if self._convert_value_type(input_val):
                    modified = True

        # Convert output types (unless keeping IO types for top-level)
        if not (is_top_level and self.keep_io_types):
            for output_val in graph.outputs:
                if self._convert_value_type(output_val):
                    modified = True

        # Process subgraphs
        for node in graph:
            for attr in node.attributes.values():
                if isinstance(attr, ir.Attr):
                    value = attr.value
                    if isinstance(value, ir.Graph):
                        if self._convert_graph_in_place(value, is_top_level=False):
                            modified = True
                    elif isinstance(value, Sequence):
                        for item in value:
                            if isinstance(item, ir.Graph):
                                if self._convert_graph_in_place(item, is_top_level=False):
                                    modified = True

        return modified

    def _convert_tensor_value(self, value: ir.Value) -> bool:
        """Convert a value's tensor data from float32 to float16.

        Args:
            value: The value to convert.

        Returns:
            True if the value was converted.
        """
        if value.const_value is None:
            return False

        tensor = value.const_value
        if not isinstance(tensor, ir.TensorProtocol):
            return False

        if tensor.dtype != ir.DataType.FLOAT:
            return False

        # Convert the tensor data
        try:
            np_array = tensor.numpy()
            if np_array.dtype != np.float32:
                return False

            # Convert to float16 with clamping
            fp16_array = _convert_np_to_float16(
                np_array, self.min_positive_val, self.max_finite_val
            )

            # Create new float16 tensor
            new_tensor = ir.tensor(
                fp16_array,
                dtype=ir.DataType.FLOAT16,
                name=tensor.name,
                doc_string=tensor.doc_string,
            )

            # Update the value's const_value
            value.const_value = new_tensor

            # Also update the type
            self._convert_value_type(value)
        except Exception:
            logger.warning("Failed to convert tensor %s", value.name, exc_info=True)
            return False
        else:
            return True

    def _convert_value_type(self, value: ir.Value) -> bool:
        """Convert a value's type from float32 to float16.

        Args:
            value: The value to convert.

        Returns:
            True if the value type was converted.
        """
        if value.type is None:
            return False

        if not isinstance(value.type, ir.TensorType):
            return False

        if value.type.dtype != ir.DataType.FLOAT:
            return False

        # Update the type to float16
        value.type = ir.TensorType(ir.DataType.FLOAT16, denotation=value.type.denotation)
        return True

    def _convert_node_attributes(self, node: ir.Node) -> bool:
        """Convert tensor attributes in a node from float32 to float16.

        Args:
            node: The node to process.

        Returns:
            True if any attributes were converted.
        """
        modified = False

        for attr_name, attr in list(node.attributes.items()):
            if isinstance(attr, ir.Attr):
                value = attr.value
                # Handle single tensor attribute
                if isinstance(value, ir.TensorProtocol):
                    if value.dtype == ir.DataType.FLOAT:
                        new_tensor = self._convert_tensor_proto(value)
                        if new_tensor is not None:
                            # Replace the attribute with the converted tensor
                            node.attributes[attr_name] = ir.AttrTensor(
                                attr_name, new_tensor, attr.doc_string
                            )
                            modified = True

                # Handle list of tensors attribute
                elif isinstance(value, Sequence) and value:
                    new_tensors = []
                    any_converted = False
                    for item in value:
                        if (
                            isinstance(item, ir.TensorProtocol)
                            and item.dtype == ir.DataType.FLOAT
                        ):
                            new_tensor = self._convert_tensor_proto(item)
                            if new_tensor is not None:
                                new_tensors.append(new_tensor)
                                any_converted = True
                            else:
                                new_tensors.append(item)
                        else:
                            new_tensors.append(item)

                    if any_converted:
                        node.attributes[attr_name] = ir.AttrTensors(
                            attr_name, new_tensors, attr.doc_string
                        )
                        modified = True

        return modified

    def _convert_tensor_proto(self, tensor: ir.TensorProtocol) -> ir.TensorProtocol | None:
        """Convert a TensorProtocol from float32 to float16.

        Args:
            tensor: The tensor to convert.

        Returns:
            The converted tensor, or None if conversion failed.
        """
        try:
            np_array = tensor.numpy()
            if np_array.dtype != np.float32:
                return None

            fp16_array = _convert_np_to_float16(
                np_array, self.min_positive_val, self.max_finite_val
            )
            return ir.tensor(
                fp16_array,
                dtype=ir.DataType.FLOAT16,
                name=tensor.name,
                doc_string=tensor.doc_string,
            )
        except Exception:
            logger.warning("Failed to convert tensor attribute", exc_info=True)
            return None
