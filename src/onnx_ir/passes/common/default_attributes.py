# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Add default attributes to nodes that are missing optional attributes."""

from __future__ import annotations

__all__ = [
    "AddDefaultAttributesPass",
]

import logging

import onnx  # noqa: TID251

import onnx_ir as ir

logger = logging.getLogger(__name__)


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
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Main entry point for the add default attributes pass."""
        modified = False

        # Process all nodes in the model graph and subgraphs
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if _add_default_attributes_to_node(node, model.graph.opset_imports):
                modified = True

        # Process nodes in functions
        for function in model.functions.values():
            for node in ir.traversal.RecursiveGraphIterator(function):
                if _add_default_attributes_to_node(node, model.graph.opset_imports):
                    modified = True

        if modified:
            logger.info("AddDefaultAttributes pass modified the model")

        return ir.passes.PassResult(model, modified=modified)


def _add_default_attributes_to_node(node: ir.Node, opset_imports: dict[str, int]) -> bool:
    """Add default attributes to a single node. Returns True if modified."""
    # Get the operator schema
    if node.version is not None:
        opset_version = node.version
    elif node.domain in opset_imports:
        opset_version = opset_imports[node.domain]
    else:
        logger.warning(
            "OpSet version for domain '%s' not found. Skipping node %s",
            node.domain,
            node,
        )
        return False

    try:
        op_schema = onnx.defs.get_schema(node.op_type, opset_version, domain=node.domain)
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
        default_attr = ir.serde.deserialize_attribute(default_attr_proto)
        node.attributes[attr_name] = default_attr
        logger.debug("Added default attribute '%s' to node %s", attr_name, node)
        modified = True

    return modified
