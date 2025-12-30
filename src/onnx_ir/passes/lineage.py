# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lineage tracking for ONNX IR passes.

This module provides utilities to track the lineage of nodes and values across
multiple transformation passes. Each node and value can be tagged with metadata
that includes a tag name and epoch number. A global counter in the model tracks
the current epoch, which is incremented each time tag() is called.

The epoch number indicates when a node/value was created - nodes created earlier
have lower epoch numbers. Visualization tools can use these epochs to color nodes
based on which pass created them.

Example:
    >>> import onnx_ir as ir
    >>> from onnx_ir.passes import lineage
    >>>
    >>> # Enable lineage tracking
    >>> with lineage.track_lineage():
    ...     # Load model
    ...     model = ir.load("model.onnx")
    ...
    ...     # Tag after first pass
    ...     lineage.tag(model, "constant_folding")
    ...     # ... run constant folding pass ...
    ...
    ...     # Tag after second pass
    ...     lineage.tag(model, "fusion")
    ...     # ... run fusion pass ...
"""

from __future__ import annotations

import contextlib
import itertools
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    import onnx_ir as ir

__all__ = [
    "tag",
    "track_lineage",
    "LINEAGE_TAG_KEY",
    "LINEAGE_EPOCH_KEY",
    "LINEAGE_COUNTER_KEY",
]

# Metadata keys for lineage tracking
LINEAGE_TAG_KEY = "pkg.onnx_ir.lineage_tag"
LINEAGE_EPOCH_KEY = "pkg.onnx_ir.lineage_epoch"
LINEAGE_COUNTER_KEY = "pkg.onnx_ir.lineage_counter"

_tracking_enabled = False


def _ensure_unique_names(model: ir.Model) -> None:
    """Ensure all nodes in the model have unique names.

    This function assigns unique names to all nodes that don't have one.
    It traverses all graphs and subgraphs in the model.

    Args:
        model: The model to process.
    """
    for graph in (model.graph, *[func.graph for func in model.functions.values()]):
        for node in graph.all_nodes():
            if node.name is None:
                # The name authority will generate a unique name
                graph._name_authority.register_or_name_node(node)


def _get_or_create_lineage_info(
    obj: ir.Node | ir.Value,
) -> tuple[str | None, int]:
    """Get the current lineage tag and epoch for a node or value.

    Args:
        obj: The node or value to query.

    Returns:
        A tuple of (tag, epoch). If no lineage info exists, returns (None, -1).
    """
    tag = obj.meta.get(LINEAGE_TAG_KEY)
    epoch = obj.meta.get(LINEAGE_EPOCH_KEY, -1)
    return tag, epoch


def _set_lineage_info(obj: ir.Node | ir.Value, tag: str, epoch: int) -> None:
    """Set the lineage tag and epoch for a node or value.

    Args:
        obj: The node or value to update.
        tag: The tag name to set.
        epoch: The epoch number to set.
    """
    obj.meta[LINEAGE_TAG_KEY] = tag
    obj.meta[LINEAGE_EPOCH_KEY] = epoch


def tag(model: ir.Model, tag_name: str) -> None:
    """Tag new nodes and values in the model with lineage information.

    This function should be called after each pass to track which nodes and values
    were created by that pass. It performs the following operations:

    1. Increments the global epoch counter stored in the model
    2. Ensures all nodes have unique names
    3. Tags any new nodes/values (those without lineage info) with the current tag and epoch

    The epoch number indicates when a node/value was created - lower epochs mean
    the node/value was created earlier. The global counter is incremented with each
    call to tag(), so nodes created in different passes will have different epochs.

    Note:
        This will not detect in-place modifications of node attributes. Only
        the creation of new nodes/values is tracked. Passes should recreate
        nodes when making modifications to ensure proper lineage tracking.

    Args:
        model: The model to tag.
        tag_name: The name of the tag to assign (typically the pass name).

    Example:
        >>> import onnx_ir as ir
        >>> from onnx_ir.passes import lineage
        >>>
        >>> model = ir.load("model.onnx")
        >>>
        >>> with lineage.track_lineage():
        ...     # Tag initial model
        ...     lineage.tag(model, "initial")
        ...
        ...     # After running a pass
        ...     lineage.tag(model, "my_optimization_pass")
        ...
        ...     # Check lineage of a node
        ...     node = model.graph[0]
        ...     tag = node.meta.get(lineage.LINEAGE_TAG_KEY)
        ...     epoch = node.meta.get(lineage.LINEAGE_EPOCH_KEY)
        ...     print(f"Node tagged with '{tag}' at epoch {epoch}")
    """
    # Increment the global epoch counter
    current_epoch = model.meta.get(LINEAGE_COUNTER_KEY, -1) + 1
    model.meta[LINEAGE_COUNTER_KEY] = current_epoch

    # Ensure all nodes have unique names
    _ensure_unique_names(model)

    # Process all graphs in the model
    for graph in model.graphs():
        # Process all nodes and their values
        for node in graph.all_nodes():
            current_tag, _ = _get_or_create_lineage_info(node)

            if current_tag is None:
                # New node - assign tag with current epoch
                _set_lineage_info(node, tag_name, current_epoch)

            # Tag all output values of the node
            for output in node.outputs:
                output_tag, _ = _get_or_create_lineage_info(output)

                if output_tag is None:
                    # New value - assign tag with current epoch
                    _set_lineage_info(output, tag_name, current_epoch)

        # Tag graph inputs
        for input_value in graph.inputs:
            input_tag, _ = _get_or_create_lineage_info(input_value)

            if input_tag is None:
                _set_lineage_info(input_value, tag_name, current_epoch)

        # Tag initializers
        for initializer in graph.initializers.values():
            init_tag, _ = _get_or_create_lineage_info(initializer)

            if init_tag is None:
                _set_lineage_info(initializer, tag_name, current_epoch)


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
