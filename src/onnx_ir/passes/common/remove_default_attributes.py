# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Remove default attributes from nodes."""

from __future__ import annotations

__all__ = [
    "RemoveDefaultAttributesPass",
]

import logging

import onnx  # noqa: TID251

import onnx_ir as ir

logger = logging.getLogger(__name__)


class RemoveDefaultAttributesPass(ir.passes.InPlacePass):
    """Remove default attributes from nodes.

    This pass removes attributes that have default values of 0, 1, or -1 as specified
    in the ONNX operator schema. Special handling for Conv op where pads all 0 or
    strides all 1 are also removed.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = self._process_graph_or_function(model.graph)
        for function in model.functions.values():
            modified = self._process_graph_or_function(function) or modified
        return ir.passes.PassResult(model, modified=modified)

    def _process_graph_or_function(self, graph_or_function: ir.Graph | ir.Function) -> bool:
        """Process all nodes in the graph or function."""
        modified = False
        graph_opset_version = graph_or_function.opset_imports.get("", None)

        for node in ir.traversal.RecursiveGraphIterator(graph_or_function):
            modified = self._process_node(node, graph_opset_version) or modified
        return modified

    def _process_node(self, node: ir.Node, graph_opset_version: int | None) -> bool:
        """Process a single node to remove default attributes."""
        if node.domain not in {"", "onnx.ai"}:
            return False

        # Use node.version if defined, otherwise use graph opset version
        onnx_opset_version = node.version if node.version is not None else graph_opset_version

        if onnx_opset_version is None:
            logger.debug(
                "No ONNX opset version found for node %s, skipping default attribute removal",
                node.name,
            )
            return False

        try:
            op_schema = onnx.defs.get_schema(
                node.op_type, onnx_opset_version, domain=node.domain
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.debug(
                "Failed to get schema for %s, skipping default attribute removal",
                node.op_type,
            )
            return False

        modified = False
        attrs_to_remove = []

        # Check each attribute in the node
        for attr_name, attr_value in node.attributes.items():
            if attr_name not in op_schema.attributes:
                continue

            schema_attr = op_schema.attributes[attr_name]

            # Check if we should remove this attribute
            if self._should_remove_attribute(node, attr_name, attr_value, schema_attr):
                attrs_to_remove.append(attr_name)

        # Remove the attributes
        for attr_name in attrs_to_remove:
            node.attributes.pop(attr_name)
            modified = True
            logger.debug("Removed default attribute '%s' from node '%s'", attr_name, node.name)

        return modified

    def _should_remove_attribute(
        self,
        node: ir.Node,
        attr_name: str,
        attr_value: ir.Attr,
        schema_attr: onnx.defs.OpSchema.Attribute,
    ) -> bool:
        """Check if an attribute should be removed."""
        # Special case for Conv node as per issue requirements:
        # The schema doesn't specify defaults for pads/strides, but they have
        # implicit defaults (pads=all 0, strides=all 1) that should be removed
        if node.op_type == "Conv":
            # Remove pads if all zeros
            if attr_name == "pads" and attr_value.type == ir.AttributeType.INTS:
                pads = attr_value.value
                if all(p == 0 for p in pads):
                    return True
            # Remove strides if all ones
            if attr_name == "strides" and attr_value.type == ir.AttributeType.INTS:
                strides = attr_value.value
                if all(s == 1 for s in strides):
                    return True

        # Check if the attribute has a default value in the schema
        has_int_default = schema_attr.default_value.HasField("i")
        has_ints_default = len(list(schema_attr.default_value.ints)) > 0

        if not has_int_default and not has_ints_default:
            return False

        # Check for int attributes with default values of 0, 1, or -1
        if has_int_default:
            default_int = schema_attr.default_value.i
            if default_int not in {0, 1, -1}:
                return False

            if attr_value.type == ir.AttributeType.INT:
                return attr_value.value == default_int

        # Check for ints attributes with default values
        if has_ints_default:
            default_ints = list(schema_attr.default_value.ints)
            # Only remove if all default values are 0, 1, or -1
            if not all(v in {0, 1, -1} for v in default_ints):
                return False

            if attr_value.type == ir.AttributeType.INTS:
                return tuple(attr_value.value) == tuple(default_ints)

        return False
