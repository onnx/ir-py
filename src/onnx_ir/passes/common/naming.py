# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Name fix pass for ensuring unique names for all values and nodes."""

from __future__ import annotations

__all__ = [
    "NameFixPass",
    "NameGenerator",
    "SimpleNameGenerator",
]

import collections
import logging
from typing import Protocol

import onnx_ir as ir

logger = logging.getLogger(__name__)


class NameGenerator(Protocol):
    def generate_node_name(self, node: ir.Node) -> str:
        """Generate a preferred name for a node."""
        ...

    def generate_value_name(self, value: ir.Value) -> str:
        """Generate a preferred name for a value."""
        ...


class SimpleNameGenerator(NameGenerator):
    """Base class for name generation functions."""

    def generate_node_name(self, node: ir.Node) -> str:
        """Generate a preferred name for a node."""
        return node.name or "node"

    def generate_value_name(self, value: ir.Value) -> str:
        """Generate a preferred name for a value."""
        return value.name or "v"


class NameFixPass(ir.passes.InPlacePass):
    """Pass for fixing names to ensure all values and nodes have unique names.

    This pass ensures that:
    1. Graph inputs and outputs have unique names (take precedence)
    2. All intermediate values have unique names (assign names to unnamed values)
    3. All values in subgraphs have unique names within their graph and parent graphs
    4. All nodes have unique names within their graph

    The pass maintains global uniqueness across the entire model.

    You can customize the name generation functions for nodes and values by passing
    a subclass of :class:`NameGenerator`.

    For example, you can use a custom naming scheme like this::

        class CustomNameGenerator:
            def custom_node_name(node: ir.Node) -> str:
                return f"custom_node_{node.op_type}"

            def custom_value_name(value: ir.Value) -> str:
                return f"custom_value_{value.type}"

        name_fix_pass = NameFixPass(name_generator=CustomNameGenerator())

    .. versionadded:: 0.1.6
    """

    def __init__(
        self,
        name_generator: NameGenerator | None = None,
    ) -> None:
        """Initialize the NameFixPass with custom name generation functions.

        Args:
            name_generator (NameGenerator, optional): An instance of a subclass of
                :class:`NameGenerator` to customize name generation for nodes and values.
                If not provided, defaults to a basic implementation that uses
                the node's or value's existing name or a generic name like "node" or "v".
        """
        super().__init__()
        self._name_generator = name_generator or SimpleNameGenerator()

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Process the main graph
        modified = self._fix_graph_names(model.graph)

        # Process functions
        for function in model.functions.values():
            modified = self._fix_graph_names(function) or modified

        return ir.passes.PassResult(model, modified=modified)

    def _fix_graph_names(self, graph_like: ir.Graph | ir.Function) -> bool:
        """Fix names in a graph and return whether modifications were made."""
        modified = False

        # Set to track which values have been assigned names
        seen_values: set[ir.Value] = set()

        # The first set is a dummy placeholder so that there is always a [-1] scope for access
        # (even though we don't write to it)
        scoped_used_value_names: list[set[str]] = [set()]
        scoped_used_node_names: list[set[str]] = [set()]

        # Counters for generating unique names (using list to pass by reference)
        value_counter = collections.Counter()
        node_counter = collections.Counter()

        def enter_graph(graph_like) -> None:
            """Callback for entering a subgraph."""
            # Initialize new scopes with all names from the parent scope
            scoped_used_value_names.append(set(scoped_used_value_names[-1]))
            scoped_used_node_names.append(set())

            nonlocal modified

            # Step 1: Fix graph input names first (they have precedence)
            for input_value in graph_like.inputs:
                if self._process_value(
                    input_value, scoped_used_value_names[-1], seen_values, value_counter
                ):
                    modified = True

            # Step 2: Fix graph output names (they have precedence)
            for output_value in graph_like.outputs:
                if self._process_value(
                    output_value, scoped_used_value_names[-1], seen_values, value_counter
                ):
                    modified = True

            if isinstance(graph_like, ir.Graph):
                # For graphs, also fix initializers
                for initializer in graph_like.initializers.values():
                    if self._process_value(
                        initializer, scoped_used_value_names[-1], seen_values, value_counter
                    ):
                        modified = True

        def exit_graph(_) -> None:
            """Callback for exiting a subgraph."""
            # Pop the current scope
            scoped_used_value_names.pop()
            scoped_used_node_names.pop()

        # Step 3: Process all nodes and their values
        for node in ir.traversal.RecursiveGraphIterator(
            graph_like, enter_graph=enter_graph, exit_graph=exit_graph
        ):
            # Fix node name
            if not node.name:
                if self._assign_node_name(node, scoped_used_node_names[-1], node_counter):
                    modified = True
            else:
                if self._fix_duplicate_node_name(
                    node, scoped_used_node_names[-1], node_counter
                ):
                    modified = True

            # Fix input value names (only if not already processed)
            for input_value in node.inputs:
                if input_value is not None:
                    if self._process_value(
                        input_value, scoped_used_value_names[-1], seen_values, value_counter
                    ):
                        modified = True

            # Fix output value names (only if not already processed)
            for output_value in node.outputs:
                if self._process_value(
                    output_value, scoped_used_value_names[-1], seen_values, value_counter
                ):
                    modified = True

        return modified

    def _process_value(
        self,
        value: ir.Value,
        used_value_names: set[str],
        seen_values: set[ir.Value],
        value_counter: collections.Counter,
    ) -> bool:
        """Process a value only if it hasn't been processed before."""
        if value in seen_values:
            return False

        modified = False

        if not value.name:
            modified = self._assign_value_name(value, used_value_names, value_counter)
        else:
            modified = self._fix_duplicate_value_name(value, used_value_names, value_counter)
            # initializers dictionary is updated automatically when the Value is renamed

        # Record the final name for this value
        assert value.name is not None
        seen_values.add(value)
        return modified

    def _assign_value_name(
        self, value: ir.Value, used_names: set[str], counter: collections.Counter
    ) -> bool:
        """Assign a name to an unnamed value. Returns True if modified."""
        assert not value.name, (
            "value should not have a name already if function is called correctly"
        )

        preferred_name = self._name_generator.generate_value_name(value)
        value.name = _find_and_record_next_unique_name(preferred_name, used_names, counter)
        logger.debug("Assigned name %s to unnamed value", value.name)
        return True

    def _assign_node_name(
        self, node: ir.Node, used_names: set[str], counter: collections.Counter
    ) -> bool:
        """Assign a name to an unnamed node. Returns True if modified."""
        assert not node.name, (
            "node should not have a name already if function is called correctly"
        )

        preferred_name = self._name_generator.generate_node_name(node)
        node.name = _find_and_record_next_unique_name(preferred_name, used_names, counter)
        logger.debug("Assigned name %s to unnamed node", node.name)
        return True

    def _fix_duplicate_value_name(
        self, value: ir.Value, used_names: set[str], counter: collections.Counter
    ) -> bool:
        """Fix a value's name if it conflicts with existing names. Returns True if modified."""
        original_name = value.name

        assert original_name, (
            "value should have a name already if function is called correctly"
        )

        if original_name not in used_names:
            # Name is unique, just record it
            used_names.add(original_name)
            return False

        # If name is already used, make it unique
        base_name = self._name_generator.generate_value_name(value)
        value.name = _find_and_record_next_unique_name(base_name, used_names, counter)
        logger.debug("Renamed value from %s to %s for uniqueness", original_name, value.name)
        return True

    def _fix_duplicate_node_name(
        self, node: ir.Node, used_names: set[str], counter: collections.Counter
    ) -> bool:
        """Fix a node's name if it conflicts with existing names. Returns True if modified."""
        original_name = node.name

        assert original_name, "node should have a name already if function is called correctly"

        if original_name not in used_names:
            # Name is unique, just record it
            used_names.add(original_name)
            return False

        # If name is already used, make it unique
        base_name = self._name_generator.generate_node_name(node)
        node.name = _find_and_record_next_unique_name(base_name, used_names, counter)
        logger.debug("Renamed node from %s to %s for uniqueness", original_name, node.name)
        return True


def _find_and_record_next_unique_name(
    preferred_name: str, used_names: set[str], counter: collections.Counter
) -> str:
    """Generate a unique name based on the preferred name and current counter."""
    new_name = preferred_name
    while new_name in used_names:
        counter[preferred_name] += 1
        new_name = f"{preferred_name}_{counter[preferred_name]}"
    used_names.add(new_name)
    return new_name
