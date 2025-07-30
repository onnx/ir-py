# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Name fix pass for ensuring unique names for all values and nodes."""

from __future__ import annotations

from typing import Callable

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
    3. All values in subgraphs have unique names within their graph and parent graphs
    4. All nodes have unique names within their graph

    The pass maintains global uniqueness across the entire model.

    You can customize the name generation functions for nodes and values by passing
    `generate_node_name` and `generate_value_name` parameters to the constructor.

    For example, you can use a custom naming scheme like this::

        def custom_node_name(node: ir.Node) -> str:
            return f"custom_node_{node.op_type}"
        def custom_value_name(value: ir.Value) -> str:
            return f"custom_value_{value.type}"

        name_fix_pass = NameFixPass(
            generate_node_name=custom_node_name,
            generate_value_name=custom_value_name
        )

    .. versionadded:: 0.1.5
    """

    def __init__(
        self,
        generate_node_name: Callable[[ir.Node], str] = lambda n: n.name or "node",
        generate_value_name: Callable[[ir.Value], str] = lambda v: v.name or "v",
    ) -> None:
        """Initialize the NameFixPass with custom name generation functions.

        Args:
            generate_node_name: Function to generate a preferred name for a node.
            generate_value_name: Function to generate a preferred name for a value.
        """
        super().__init__()
        self._generate_node_name = generate_node_name
        self._generate_value_name = generate_value_name

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        # Process the main graph
        if self._fix_graph_names(model.graph):
            modified = True

        # Process functions
        for function in model.functions.values():
            if self._fix_graph_names(function):
                modified = True

        if modified:
            logger.info("Name fix pass modified the model")

        return ir.passes.PassResult(model, modified=modified)

    def _fix_graph_names(self, graph_like: ir.Graph | ir.Function) -> bool:
        """Fix names in a graph and return whether modifications were made."""
        modified = False

        # Set to track which values have been assigned names
        seen_values: set[ir.Value] = set()

        # The first set is a dummy placeholder so that there is always a [-1] scope for access
        # (even though we don't write to it)
        scoped_seen_value_names: list[set[str]] = [set()]
        scoped_seen_node_names: list[set[str]] = [set()]

        # Counters for generating unique names (using list to pass by reference)
        value_counter = [0]
        node_counter = [0]

        def enter_graph(graph_like) -> None:
            """Callback for entering a subgraph."""
            # Initialize new scopes with all names from the parent scope
            scoped_seen_value_names.append(set(scoped_seen_value_names[-1]))
            scoped_seen_node_names.append(set())

            nonlocal modified

            # Step 1: Fix graph input names first (they have precedence)
            for input_value in graph_like.inputs:
                if self._process_value(
                    input_value, scoped_seen_value_names[-1], seen_values, value_counter
                ):
                    modified = True

            # Step 2: Fix graph output names (they have precedence)
            for output_value in graph_like.outputs:
                if self._process_value(
                    output_value, scoped_seen_value_names[-1], seen_values, value_counter
                ):
                    modified = True

            if isinstance(graph_like, ir.Graph):
                # For graphs, also fix initializers
                for initializer in graph_like.initializers.values():
                    if self._process_value(
                        initializer, scoped_seen_value_names[-1], seen_values, value_counter
                    ):
                        modified = True

        def exit_graph(_) -> None:
            """Callback for exiting a subgraph."""
            # Pop the current scope
            scoped_seen_value_names.pop()
            scoped_seen_node_names.pop()

        # Step 3: Process all nodes and their values
        for node in ir.traversal.RecursiveGraphIterator(
            graph_like, enter_graph=enter_graph, exit_graph=exit_graph
        ):
            # Fix node name
            if not node.name:
                if self._assign_node_name(node, scoped_seen_node_names[-1], node_counter):
                    modified = True
            else:
                if self._fix_duplicate_node_name(node, scoped_seen_node_names[-1]):
                    modified = True

            # Fix input value names (only if not already processed)
            for input_value in node.inputs:
                if input_value is not None:
                    if self._process_value(
                        input_value, scoped_seen_value_names[-1], seen_values, value_counter
                    ):
                        modified = True

            # Fix output value names (only if not already processed)
            for output_value in node.outputs:
                if self._process_value(
                    output_value, scoped_seen_value_names[-1], seen_values, value_counter
                ):
                    modified = True

        return modified

    def _process_value(
        self,
        value: ir.Value,
        seen_value_names: set[str],
        seen_values: set[ir.Value],
        value_counter: list[int],
    ) -> bool:
        """Process a value only if it hasn't been processed before."""
        if value in seen_values:
            return False

        modified = False

        if not value.name:
            modified = self._assign_value_name(value, seen_value_names, value_counter)
        else:
            old_name = value.name
            modified = self._fix_duplicate_value_name(value, seen_value_names)
            if modified:
                assert value.graph is not None
                if value.is_initializer():
                    value.graph.initializers.pop(old_name)
                    # Add the initializer back with the new name
                    value.graph.initializers.add(value)

        # Record the final name for this value
        assert value.name is not None
        seen_values.add(value)
        return modified

    def _assign_value_name(
        self, value: ir.Value, seen_names: set[str], counter: list[int]
    ) -> bool:
        """Assign a name to an unnamed value. Returns True if modified."""
        assert not value.name, (
            "value should not have a name already if function is called correctly"
        )

        preferred_name = self._generate_value_name(value)
        value.name = _find_and_record_next_unique_name(preferred_name, seen_names, counter)
        logger.debug("Assigned name %s to unnamed value", value.name)
        return True

    def _assign_node_name(
        self, node: ir.Node, seen_names: set[str], counter: list[int]
    ) -> bool:
        """Assign a name to an unnamed node. Returns True if modified."""
        assert not node.name, (
            "node should not have a name already if function is called correctly"
        )

        preferred_name = self._generate_node_name(node)
        node.name = _find_and_record_next_unique_name(preferred_name, seen_names, counter)
        logger.debug("Assigned name %s to unnamed node", node.name)
        return True

    def _fix_duplicate_value_name(self, value: ir.Value, seen_names: set[str]) -> bool:
        """Fix a value's name if it conflicts with existing names. Returns True if modified."""
        original_name = value.name

        assert original_name, (
            "value should have a name already if function is called correctly"
        )

        if original_name not in seen_names:
            # Name is unique, just record it
            seen_names.add(original_name)
            return False

        # If name is already seen, make it unique
        base_name = self._generate_value_name(value)
        value.name = _find_and_record_next_unique_name(base_name, seen_names)
        logger.debug("Renamed value from %s to %s for uniqueness", original_name, value.name)
        return True

    def _fix_duplicate_node_name(self, node: ir.Node, seen_names: set[str]) -> bool:
        """Fix a node's name if it conflicts with existing names. Returns True if modified."""
        original_name = node.name

        assert original_name, "node should have a name already if function is called correctly"

        if original_name not in seen_names:
            # Name is unique, just record it
            seen_names.add(original_name)
            return False

        # If name is already seen, make it unique
        base_name = self._generate_node_name(node)
        node.name = _find_and_record_next_unique_name(base_name, seen_names)
        logger.debug("Renamed node from %s to %s for uniqueness", original_name, node.name)
        return True


def _find_and_record_next_unique_name(
    preferred_name: str, seen_names: set[str], counter: list[int] | None = None
) -> str:
    """Generate a unique name based on the preferred name and current counter."""
    new_name = preferred_name
    if counter is None:
        counter = [0]
    while new_name in seen_names:
        counter[0] += 1
        new_name = f"{preferred_name}_{counter[0]}"
    seen_names.add(new_name)
    return new_name
