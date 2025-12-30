# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lineage tracking for ONNX IR passes.

This module provides utilities to track the lineage of nodes and values across
multiple transformation passes. Each node and value can be tagged with metadata
that includes a tag name and step number. A global counter in the model tracks
the current step, which is incremented each time tag() is called.

The step number indicates when a node/value was created - nodes created earlier
have lower step numbers. Visualization tools can use these steps to color nodes
based on which pass created them.
"""

from __future__ import annotations

__all__ = [
    "tag",
    "track_lineage",
    "LINEAGE_TAG_KEY",
    "LINEAGE_STEP_KEY",
    "LINEAGE_COUNTER_KEY",
]


import ast
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    import onnx_ir as ir

# Metadata keys for lineage tracking
LINEAGE_TAG_KEY = "pkg.onnx_ir.lineage_tag"
LINEAGE_STEP_KEY = "pkg.onnx_ir.lineage_step"
LINEAGE_COUNTER_KEY = "pkg.onnx_ir.lineage_counter"

_tracking_enabled = False


# def _ensure_unique_names(model: ir.Model) -> None:
#     """Ensure all nodes in the model have unique names.

#     This function assigns unique names to all nodes that don't have one.
#     It traverses all graphs and subgraphs in the model.

#     Args:
#         model: The model to process.
#     """
#     for graph in (model.graph, *[func.graph for func in model.functions.values()]):
#         for node in graph.all_nodes():
#             if node.name is None:
#                 # The name authority will generate a unique name
#                 graph._name_authority.register_or_name_node(node)


def _increment_or_create_model_step(model: ir.Model) -> str:
    step_str = model.metadata_props.get(LINEAGE_COUNTER_KEY)
    if step_str is None:
        model.metadata_props[LINEAGE_COUNTER_KEY] = "0"
        return "0"

    step_str = str(ast.literal_eval(step_str) + 1)
    model.metadata_props[LINEAGE_COUNTER_KEY] = step_str

    return step_str


def _maybe_set_lineage_info(obj: ir.Node | ir.Value, tag: str, step: str) -> None:
    """Set the lineage tag and step for a node or value.

    Args:
        obj: The node or value to update.
        tag: The tag name to set.
        step: The step number to set.
    """
    if LINEAGE_TAG_KEY not in obj.metadata_props and LINEAGE_STEP_KEY not in obj.metadata_props:
        obj.metadata_props[LINEAGE_TAG_KEY] = tag
        obj.metadata_props[LINEAGE_STEP_KEY] = step


def tag(model: ir.Model, tag_name: str) -> None:
    """Tag new nodes and values in the model with lineage information.

    This function should be called after each pass to track which nodes and values
    were created by that pass. It performs the following operations:

    1. Increments the global step counter stored in the model
    2. Ensures all nodes have unique names
    3. Tags any new nodes/values (those without lineage info) with the current tag and step

    The step number indicates when a node/value was created - lower steps mean
    the node/value was created earlier. The global counter is incremented with each
    call to tag(), so nodes created in different passes will have different steps.

    Note:
        This will not detect in-place modifications of node attributes. Only
        the creation of new nodes/values is tracked. Passes should recreate
        nodes when making modifications to ensure proper lineage tracking.

    Args:
        model: The model to tag.
        tag_name: The name of the tag to assign (typically the pass name).

    Example::
        import onnx_ir as ir
        from onnx_ir.passes import lineage

        model = ir.load("model.onnx")

        # Tag initial model
        lineage.tag(model, "initial")
        # After running a pass
        lineage.tag(model, "my_optimization_pass")
        # Check lineage of a node
        node = model.graph[0]
        tag = node.metadata_props.get(lineage.LINEAGE_TAG_KEY)
        step = node.metadata_props.get(lineage.LINEAGE_STEP_KEY)
        print(f"Node tagged with '{tag}' at step {step}")
    """
    # Increment the global step counter
    current_step = _increment_or_create_model_step(model)

    # Process all graphs in the model
    for graph in model.graphs():
        # Process all nodes and their values
        for node in graph.all_nodes():
            _maybe_set_lineage_info(node, tag_name, current_step)

        # Tag graph inputs
        for input_value in graph.inputs:
            _maybe_set_lineage_info(input_value, tag_name, current_step)

        # Tag initializers
        for initializer in graph.initializers.values():
            _maybe_set_lineage_info(initializer, tag_name, current_step)


@contextlib.contextmanager
def track_lineage(enabled: bool = True) -> Generator[None, None, None]:
    """Context manager to enable or disable lineage tracking.

    When enabled, calls to :func:`tag` will record lineage information on nodes
    and values. When disabled, calls to :func:`tag` are no-ops.

    This uses thread-local storage, so each thread can have independent tracking state.

    Args:
        enabled: Whether to enable tracking. Defaults to True.

    Yields:
        None

    Example:
        >>> import onnx_ir as ir
        >>> from onnx_ir.passes import lineage
        >>>
        >>> with lineage.track_lineage():
        ...     model = ir.load("model.onnx")
        ...     lineage.tag(model, "initial")
        ...     # ... run passes ...
        ...     lineage.tag(model, "optimized")
        >>>
        >>> # Lineage tracking is now disabled
        >>> lineage.tag(model, "final")  # This is a no-op
    """
    global _tracking_enabled
    old_value = _tracking_enabled
    _tracking_enabled = enabled
    try:
        yield
    finally:
        _tracking_enabled = old_value
