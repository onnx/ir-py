# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Add default attributes to nodes that are missing optional attributes."""

from __future__ import annotations

__all__ = [
    "AddDefaultAttributesPass",
]

import logging
from typing import Union

import onnx  # noqa: TID251

import onnx_ir as ir

logger = logging.getLogger(__name__)

# Type alias for attribute values
AttrValue = Union[
    float,
    int,
    str,
    ir.TensorProtocol,
    list[float],
    list[int],
    list[str],
    list[ir.TensorProtocol],
]

# Default ONNX opset version to use when not specified in the model
DEFAULT_OPSET_VERSION = 1


def _decode_string_value(s: bytes | str) -> str:
    """Decode a string value from bytes if necessary."""
    return s.decode("utf-8") if isinstance(s, bytes) else s


def _has_valid_default(attr_def: onnx.defs.OpSchema.Attribute) -> bool:
    """Check if an attribute definition has a valid default value."""
    return bool(
        attr_def.default_value and attr_def.default_value.type != onnx.AttributeProto.UNDEFINED
    )


class AddDefaultAttributesPass(ir.passes.InPlacePass):
    """Add default values for optional attributes that are not present in nodes.

    This pass iterates through all nodes in the model and for each node:
    1. Gets the ONNX schema for the operator
    2. For each optional attribute with a default value in the schema
    3. If the attribute is not present in the node, adds it with the default value

    This is the reverse operation of RemoveDefaultAttributesPass.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Main entry point for the add default attributes pass."""
        modified = False

        # Get the opset version for the main ONNX domain
        onnx_opset_version = model.graph.opset_imports.get("", DEFAULT_OPSET_VERSION)

        # Process all nodes in the model graph and subgraphs
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if self._add_default_attributes_to_node(node, onnx_opset_version):
                modified = True

        # Process nodes in functions
        for function in model.functions.values():
            for node in ir.traversal.RecursiveGraphIterator(function):
                if self._add_default_attributes_to_node(node, onnx_opset_version):
                    modified = True

        if modified:
            logger.info("AddDefaultAttributes pass modified the model")

        return ir.passes.PassResult(model, modified=modified)

    def _add_default_attributes_to_node(self, node: ir.Node, onnx_opset_version: int) -> bool:
        """Add default attributes to a single node. Returns True if modified."""
        # Only process standard ONNX operators
        if node.domain not in {"", "ai.onnx"}:
            return False

        # Get the operator schema
        try:
            op_schema = onnx.defs.get_schema(
                node.op_type, onnx_opset_version, domain=node.domain
            )
        except onnx.defs.SchemaError:
            logger.debug(
                "Schema not found for %s, skipping default attribute addition",
                node,
            )
            return False

        modified = False
        # Iterate through all attributes in the schema
        for attr_name, attr_def in op_schema.attributes.items():
            # Skip if attribute is required or already present in the node
            if attr_def.required or attr_name in node.attributes:
                continue

            # Skip if attribute doesn't have a default value
            if not _has_valid_default(attr_def):
                continue

            # Create an IR Attr from the ONNX AttributeProto default value
            default_attr_proto = attr_def.default_value
            try:
                default_attr = self._proto_attr_to_ir_attr(default_attr_proto)
                node.attributes[attr_name] = default_attr
                modified = True
                logger.debug("Added default attribute '%s' to node %s", attr_name, node)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(
                    "Failed to convert default attribute '%s' for node %s: %s",
                    attr_name,
                    node,
                    e,
                )

        return modified

    def _proto_attr_to_ir_attr(self, attr_proto: onnx.AttributeProto) -> ir.Attr:
        """Convert an ONNX AttributeProto to an IR Attr."""
        attr_type = ir.AttributeType(attr_proto.type)
        name = attr_proto.name

        # Extract the value based on the attribute type
        value: AttrValue
        if attr_type == ir.AttributeType.FLOAT:
            value = attr_proto.f
        elif attr_type == ir.AttributeType.INT:
            value = attr_proto.i
        elif attr_type == ir.AttributeType.STRING:
            value = _decode_string_value(attr_proto.s)
        elif attr_type == ir.AttributeType.TENSOR:
            # Convert TensorProto to IR Tensor
            value = ir.serde.deserialize_tensor(attr_proto.t)
        elif attr_type == ir.AttributeType.GRAPH:
            # This is more complex and may require special handling
            # For now, we skip graph attributes
            raise ValueError(f"Graph attributes are not supported: {name}")
        elif attr_type == ir.AttributeType.FLOATS:
            value = list(attr_proto.floats)
        elif attr_type == ir.AttributeType.INTS:
            value = list(attr_proto.ints)
        elif attr_type == ir.AttributeType.STRINGS:
            value = [_decode_string_value(s) for s in attr_proto.strings]
        elif attr_type == ir.AttributeType.TENSORS:
            value = [ir.serde.deserialize_tensor(t) for t in attr_proto.tensors]
        elif attr_type == ir.AttributeType.GRAPHS:
            # This is more complex and may require special handling
            raise ValueError(f"Graph attributes are not supported: {name}")
        else:
            raise ValueError(f"Unsupported attribute type: {attr_type}")

        return ir.Attr(name, attr_type, value)
