# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Name fix pass for ensuring unique names for all values and nodes."""

from __future__ import annotations

__all__ = [
    "NameFixPass",
]

import logging
from collections.abc import Set as AbstractSet

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
        """Main entry point for the name fix pass."""
        modified = False

        # Use sets to track seen names globally
        seen_value_names: set[str] = set()
        seen_node_names: set[str] = set()
        
        # Counters for generating unique names (using list to pass by reference)
        value_counter = [0]
        node_counter = [0]

        # Process the main graph
        if self._fix_graph_names(
            model.graph, seen_value_names, seen_node_names, value_counter, node_counter
        ):
            modified = True

        # Process functions
        for function in model.functions.values():
            if self._fix_function_names(
                function, seen_value_names, seen_node_names, value_counter, node_counter
            ):
                modified = True

        if modified:
            logger.info("Name fix pass modified the model")

        return ir.passes.PassResult(model, modified=modified)


    def _fix_graph_names(
        self,
        graph: ir.Graph,
        seen_value_names: set[str],
        seen_node_names: set[str],
        value_counter: list[int],
        node_counter: list[int],
    ) -> bool:
        """Fix names in a graph and return whether modifications were made."""
        modified = False
        
        # Keep track of values we've already processed to avoid double-processing
        processed_values: set[ir.Value] = set()

        # Step 1: Fix graph input names first (they have precedence)
        for input_value in graph.inputs:
            if self._process_value(input_value, seen_value_names, value_counter, processed_values):
                modified = True

        # Step 2: Fix graph output names (they have precedence)
        for output_value in graph.outputs:
            if self._process_value(output_value, seen_value_names, value_counter, processed_values):
                modified = True

        # Step 3: Fix initializer names
        for initializer in graph.initializers.values():
            if self._process_value(initializer, seen_value_names, value_counter, processed_values):
                modified = True

        # Step 4: Process all nodes and their values
        for node in ir.traversal.RecursiveGraphIterator(graph):
            # Fix node name
            if node.name is None or node.name == "":
                if self._assign_node_name(node, seen_node_names, node_counter):
                    modified = True
            else:
                if self._fix_duplicate_node_name(node, seen_node_names):
                    modified = True

            # Fix input value names (only if not already processed)
            for input_value in node.inputs:
                if input_value is not None:
                    if self._process_value(input_value, seen_value_names, value_counter, processed_values):
                        modified = True

            # Fix output value names (only if not already processed)
            for output_value in node.outputs:
                if self._process_value(output_value, seen_value_names, value_counter, processed_values):
                    modified = True

        return modified

    def _fix_function_names(
        self,
        function: ir.Function,
        seen_value_names: set[str],
        seen_node_names: set[str],
        value_counter: list[int],
        node_counter: list[int],
    ) -> bool:
        """Fix names in a function and return whether modifications were made."""
        modified = False
        
        # Keep track of values we've already processed to avoid double-processing
        processed_values: set[ir.Value] = set()

        # Process function inputs first (they have precedence)
        for input_value in function.inputs:
            if self._process_value(input_value, seen_value_names, value_counter, processed_values):
                modified = True

        # Process function outputs (they have precedence)
        for output_value in function.outputs:
            if self._process_value(output_value, seen_value_names, value_counter, processed_values):
                modified = True

        # Process all nodes and their values
        for node in ir.traversal.RecursiveGraphIterator(function):
            # Fix node name
            if node.name is None or node.name == "":
                if self._assign_node_name(node, seen_node_names, node_counter):
                    modified = True
            else:
                if self._fix_duplicate_node_name(node, seen_node_names):
                    modified = True

            # Fix input value names (only if not already processed)
            for input_value in node.inputs:
                if input_value is not None:
                    if self._process_value(input_value, seen_value_names, value_counter, processed_values):
                        modified = True

            # Fix output value names (only if not already processed)
            for output_value in node.outputs:
                if self._process_value(output_value, seen_value_names, value_counter, processed_values):
                    modified = True

        return modified

    def _process_value(
        self, 
        value: ir.Value, 
        seen_value_names: set[str], 
        value_counter: list[int], 
        processed_values: set[ir.Value]
    ) -> bool:
        """Process a value only if it hasn't been processed before."""
        if value in processed_values:
            return False
        
        processed_values.add(value)
        
        if value.name is None or value.name == "":
            return self._assign_value_name(value, seen_value_names, value_counter)
        else:
            return self._fix_duplicate_value_name(value, seen_value_names)

    def _assign_value_name(
        self, value: ir.Value, seen_names: set[str], counter: list[int]
    ) -> bool:
        """Assign a name to an unnamed value. Returns True if modified."""
        while True:
            new_name = f"val_{counter[0]}"
            counter[0] += 1
            if new_name not in seen_names:
                value.name = new_name
                seen_names.add(new_name)
                logger.debug("Assigned name %s to unnamed value", new_name)
                return True

    def _assign_node_name(
        self, node: ir.Node, seen_names: set[str], counter: list[int]
    ) -> bool:
        """Assign a name to an unnamed node. Returns True if modified."""
        while True:
            new_name = f"node_{counter[0]}"
            counter[0] += 1
            if new_name not in seen_names:
                node.name = new_name
                seen_names.add(new_name)
                logger.debug("Assigned name %s to unnamed node", new_name)
                return True

    def _fix_duplicate_value_name(
        self, value: ir.Value, seen_names: set[str]
    ) -> bool:
        """Fix a value's name if it conflicts with existing names. Returns True if modified."""
        original_name = value.name
        
        if original_name is None or original_name == "":
            return False  # Should not happen if called correctly
            
        # If name is already seen, make it unique
        if original_name in seen_names:
            base_name = original_name
            suffix = 1
            while True:
                new_name = f"{base_name}_{suffix}"
                if new_name not in seen_names:
                    value.name = new_name
                    seen_names.add(new_name)
                    logger.debug("Renamed value from %s to %s for uniqueness", original_name, new_name)
                    return True
                suffix += 1
        else:
            # Name is unique, just record it
            seen_names.add(original_name)
            return False

    def _fix_duplicate_node_name(
        self, node: ir.Node, seen_names: set[str]
    ) -> bool:
        """Fix a node's name if it conflicts with existing names. Returns True if modified."""
        original_name = node.name
        
        if original_name is None or original_name == "":
            return False  # Should not happen if called correctly
            
        # If name is already seen, make it unique
        if original_name in seen_names:
            base_name = original_name
            suffix = 1
            while True:
                new_name = f"{base_name}_{suffix}"
                if new_name not in seen_names:
                    node.name = new_name
                    seen_names.add(new_name)
                    logger.debug("Renamed node from %s to %s for uniqueness", original_name, new_name)
                    return True
                suffix += 1
        else:
            # Name is unique, just record it
            seen_names.add(original_name)
            return False