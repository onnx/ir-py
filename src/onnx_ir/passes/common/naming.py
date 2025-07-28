# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Name fix pass for ensuring unique names for all values and nodes."""

from __future__ import annotations

__all__ = [
    "NameFixPass",
]

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


class NameFixPass(ir.passes.InPlacePass):
    """Pass for fixing names to ensure all values and nodes have unique names.

    This pass ensures that:
    1. Graph inputs and outputs have unique names (take precedence)
    2. All intermediate values have unique names (assign names to unnamed values)
    3. All values in subgraphs have unique names
    4. All nodes have unique names (assign names to unnamed nodes)

    The pass maintains global uniqueness across the entire model.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        # Use sets to track seen names globally
        seen_value_names: set[str] = set()
        seen_node_names: set[str] = set()

        # Dictionary to track which values have been assigned names
        value_to_name: dict[ir.Value, str] = {}

        # Counters for generating unique names (using list to pass by reference)
        value_counter = [0]
        node_counter = [0]

        # Process the main graph
        if _fix_graph_names(
            model.graph,
            seen_value_names,
            seen_node_names,
            value_to_name,
            value_counter,
            node_counter,
        ):
            modified = True

        # Process functions
        for function in model.functions.values():
            # Reset seen names and counters for each function
            seen_value_names: set[str] = set()
            seen_node_names: set[str] = set()
            value_to_name: dict[ir.Value, str] = {}
            value_counter = [0]
            node_counter = [0]
            if _fix_graph_names(
                function,
                seen_value_names,
                seen_node_names,
                value_to_name,
                value_counter,
                node_counter,
            ):
                modified = True

        if modified:
            logger.info("Name fix pass modified the model")

        return ir.passes.PassResult(model, modified=modified)


def _fix_graph_names(
    graph_like: ir.Graph | ir.Function,
    seen_value_names: set[str],
    seen_node_names: set[str],
    value_to_name: dict[ir.Value, str],
    value_counter: list[int],
    node_counter: list[int],
) -> bool:
    """Fix names in a graph and return whether modifications were made."""
    modified = False

    # Step 1: Fix graph input names first (they have precedence)
    for input_value in graph_like.inputs:
        if _process_value(input_value, seen_value_names, value_to_name, value_counter):
            modified = True

    # Step 2: Fix graph output names (they have precedence)
    for output_value in graph_like.outputs:
        if _process_value(output_value, seen_value_names, value_to_name, value_counter):
            modified = True

    # Step 3: Process all nodes and their values. Initializers are processed as node inputs.
    for node in ir.traversal.RecursiveGraphIterator(graph_like):
        # Fix node name
        if not node.name:
            if _assign_node_name(node, seen_node_names, node_counter):
                modified = True
        else:
            if _fix_duplicate_node_name(node, seen_node_names):
                modified = True

        # Fix input value names (only if not already processed)
        for input_value in node.inputs:
            if input_value is not None:
                if _process_value(input_value, seen_value_names, value_to_name, value_counter):
                    modified = True

        # Fix output value names (only if not already processed)
        for output_value in node.outputs:
            if _process_value(output_value, seen_value_names, value_to_name, value_counter):
                modified = True

    return modified


def _process_value(
    value: ir.Value,
    seen_value_names: set[str],
    value_to_name: dict[ir.Value, str],
    value_counter: list[int],
) -> bool:
    """Process a value only if it hasn't been processed before."""
    if value in value_to_name:
        return False

    modified = False
    if not value.name:
        modified = _assign_value_name(value, seen_value_names, value_counter)
    else:
        modified = _fix_duplicate_value_name(value, seen_value_names)

    # Record the final name for this value
    assert value.name is not None
    value_to_name[value] = value.name
    return modified


def _assign_value_name(value: ir.Value, seen_names: set[str], counter: list[int]) -> bool:
    """Assign a name to an unnamed value. Returns True if modified."""
    assert not value.name, (
        "value should not have a name already if function is called correctly"
    )

    new_name = f"val_{counter[0]}"
    while new_name in seen_names:
        counter[0] += 1
        new_name = f"val_{counter[0]}"

    value.name = new_name
    seen_names.add(new_name)
    logger.debug("Assigned name %s to unnamed value", new_name)
    return True


def _assign_node_name(node: ir.Node, seen_names: set[str], counter: list[int]) -> bool:
    """Assign a name to an unnamed node. Returns True if modified."""
    assert not node.name, "node should not have a name already if function is called correctly"

    new_name = f"node_{counter[0]}"

    while new_name in seen_names:
        counter[0] += 1
        new_name = f"node_{counter[0]}"

    node.name = new_name
    seen_names.add(new_name)
    logger.debug("Assigned name %s to unnamed node", new_name)
    return True


def _fix_duplicate_value_name(value: ir.Value, seen_names: set[str]) -> bool:
    """Fix a value's name if it conflicts with existing names. Returns True if modified."""
    original_name = value.name

    assert original_name, "value should have a name already if function is called correctly"

    if original_name not in seen_names:
        # Name is unique, just record it
        seen_names.add(original_name)
        return False

    # If name is already seen, make it unique
    base_name = original_name
    suffix = 1
    new_name = base_name
    while new_name in seen_names:
        new_name = f"{base_name}_{suffix}"
        suffix += 1
    value.name = new_name
    seen_names.add(new_name)
    logger.debug("Renamed value from %s to %s for uniqueness", original_name, new_name)
    return True


def _fix_duplicate_node_name(node: ir.Node, seen_names: set[str]) -> bool:
    """Fix a node's name if it conflicts with existing names. Returns True if modified."""
    original_name = node.name

    assert original_name, "node should have a name already if function is called correctly"

    if original_name not in seen_names:
        # Name is unique, just record it
        seen_names.add(original_name)
        return False

    # If name is already seen, make it unique
    base_name = original_name
    suffix = 1
    new_name = base_name
    while new_name in seen_names:
        new_name = f"{base_name}_{suffix}"
        suffix += 1
    node.name = new_name
    seen_names.add(new_name)
    logger.debug("Renamed node from %s to %s for uniqueness", original_name, new_name)
    return True
